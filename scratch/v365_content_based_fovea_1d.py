import sys
import os
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Start time tracking for formatted logs
START_TIME = time.time()

def log_msg(msg):
    elapsed = time.time() - START_TIME
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    timestamp = f"[+{hours:02d}:{minutes:02d}:{seconds:05.2f}]"
    print(f"{timestamp} {msg}", flush=True)

# -----------------------------------------------------------------------------
# Dataset Sintético: MQAR 1D (Associative Recall T=128)
# -----------------------------------------------------------------------------
class MQARDataset(Dataset):
    def __init__(self, num_samples=3000, seq_len=128, num_kv_pairs=4, num_keys=16, num_values=16, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_kv_pairs = num_kv_pairs
        
        self.key_offset = 1
        self.val_offset = 1 + num_keys
        self.filler_offset = 1 + num_keys + num_values
        self.vocab_size = 1 + num_keys + num_values + 16
        
        self.sequences = torch.zeros(num_samples, seq_len, dtype=torch.long)
        self.targets = torch.zeros(num_samples, dtype=torch.long)
        
        for i in range(num_samples):
            seq = torch.randint(self.filler_offset, self.vocab_size, (seq_len,))
            
            keys = torch.randperm(num_keys)[:num_kv_pairs] + self.key_offset
            vals = torch.randperm(num_values)[:num_kv_pairs] + self.val_offset
            
            available_pos = list(range(0, seq_len - 10, 2))
            chosen_pos_indices = torch.randperm(len(available_pos))[:num_kv_pairs]
            
            for j in range(num_kv_pairs):
                pos = available_pos[chosen_pos_indices[j]]
                seq[pos] = keys[j]
                seq[pos + 1] = vals[j]
                
            query_pair_idx = torch.randint(0, num_kv_pairs, (1,)).item()
            target_key = keys[query_pair_idx]
            target_val = vals[query_pair_idx]
            
            seq[-1] = target_key
            self.sequences[i] = seq
            self.targets[i] = target_val

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# -----------------------------------------------------------------------------
# Core Layer: ContentBasedFoveaAttention1D (Atención Foveal 1D Dinámica)
# -----------------------------------------------------------------------------
class ContentBasedFoveaAttention1D(nn.Module):
    """
    Atención Foveal 1D Basada en Contenido (Content-Based Fovea 1D).
    
    A diferencia del cono estático, cada token X_i predice dinámicamente:
      - c_{i,k}: offset hacia el pasado en [0, i] (¿dónde está la clave pasada?)
      - r_{i,k}: radio de atención en el pasado
      
    Esto permite que cuando un token Query aparece, su representación prediga
    exactamente a cuántos pasos de distancia atrás está el dato relevante.
    """
    def __init__(self, d_model=64, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Redes independientes por cabeza para predecir centro y radio dinámicos
        self.offset_predictor = nn.Linear(d_model, num_heads)
        self.radius_predictor = nn.Linear(d_model, num_heads)
        self.amplitude = nn.Parameter(torch.randn(num_heads) * 0.5)
        
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.size()
        K = self.num_heads
        
        # 1. Matriz de deltas temporales causales [T, T] (i - j para j <= i)
        i_indices = torch.arange(T, device=x.device).unsqueeze(1) # [T, 1]
        j_indices = torch.arange(T, device=x.device).unsqueeze(0) # [1, T]
        delta_temporal = (i_indices - j_indices).float() # [T, T]
        causal_mask = (delta_temporal >= 0).float() # [T, T]
        
        # 2. Predicción DINÁMICA de offset y radio basada en el contenido de cada token
        # offset_ratio: [B, T, K] en [0, 1] (qué porcentaje del pasado i mirar hacia atrás)
        offset_ratio = torch.sigmoid(self.offset_predictor(x)) # [B, T, K]
        radius_ratio = torch.sigmoid(self.radius_predictor(x)) * 0.5 + 0.05 # [B, T, K] en [0.05, 0.55]
        
        # Escalar offsets al número de tokens pasados en la posición i
        # i_scale: [1, T, 1] (0, 1, 2, ..., T-1)
        i_scale = i_indices.float().unsqueeze(0) # [1, T, 1]
        
        # c_dyn: [B, K, T, 1] (offset dinámico en número de pasos atrás para cada posición i)
        c_dyn = (offset_ratio * i_scale).permute(0, 2, 1).unsqueeze(-1) # [B, K, T, 1]
        r_dyn = (radius_ratio * i_scale + 1e-2).permute(0, 2, 1).unsqueeze(-1) # [B, K, T, 1]
        a = self.amplitude.view(1, K, 1, 1)
        
        # 3. Construir ponderaciones foveales dinámicas [B, K, T, T]
        # delta_temporal: [1, 1, T, T]
        delta_exp = delta_temporal.unsqueeze(0).unsqueeze(0) # [1, 1, T, T]
        
        dist_from_center = torch.abs(delta_exp - c_dyn) # [B, K, T, T]
        base_weights = F.relu(1.0 - (dist_from_center / r_dyn)) * causal_mask.unsqueeze(0).unsqueeze(0) # [B, K, T, T]
        head_weights = base_weights * a # [B, K, T, T]
        
        # 4. Proyección V y agregación vectorizada
        d_head = D // K
        v = self.v_proj(x).view(B, T, K, d_head).permute(0, 2, 1, 3) # [B, K, T, d_head]
        
        attn_out = torch.matmul(head_weights, v) # [B, K, T, d_head]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, D) # [B, T, D]
        
        return self.out_proj(attn_out)

# -----------------------------------------------------------------------------
# Modelo ContentBasedFovea1DNet (V365)
# -----------------------------------------------------------------------------
class ContentBasedFovea1DNet(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(256, d_model)
        
        self.fovea_attn = ContentBasedFoveaAttention1D(d_model=d_model, num_heads=num_heads)
        self.norm = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, vocab_size)
        )

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_emb(pos)
        
        attn_out = self.fovea_attn(h)
        h = self.norm(h + attn_out)
        
        logits = self.mlp(h)
        return logits

# -----------------------------------------------------------------------------
# Baseline Causal Transformer (Softmax QK^T)
# -----------------------------------------------------------------------------
class BaselineCausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(256, d_model)
        
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            batch_first=True,
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_emb(pos)
        
        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        h = self.transformer(h, mask=causal_mask)
        logits = self.output(h)
        return logits

# -----------------------------------------------------------------------------
# Bucle de Entrenamiento
# -----------------------------------------------------------------------------
def run_experiment():
    log_msg("=========================================================================")
    log_msg("  EXPERIMENTO 2: Content-Based Fovea 1D vs Transformer (v365)")
    log_msg("=========================================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_msg(f"Metadatos: Python {sys.version.split()[0]} | PyTorch {torch.__version__} | Dispositivo: {device}")
    
    seed = 42
    torch.manual_seed(seed)
    
    seq_len = 128
    train_dataset = MQARDataset(num_samples=3000, seq_len=seq_len, num_kv_pairs=4, seed=seed)
    test_dataset = MQARDataset(num_samples=600, seq_len=seq_len, num_kv_pairs=4, seed=seed+1)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=600, shuffle=False)
    
    vocab_size = train_dataset.vocab_size
    log_msg(f"Configuración MQAR: VocabSize={vocab_size} | SeqLen={seq_len} | Parejas K-V=4")
    
    # Modelos
    fovea_model = ContentBasedFovea1DNet(vocab_size=vocab_size, d_model=64, num_heads=8).to(device)
    baseline_model = BaselineCausalTransformer(vocab_size=vocab_size, d_model=64, num_heads=4).to(device)
    
    fovea_params = sum(p.numel() for p in fovea_model.parameters() if p.requires_grad)
    baseline_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    
    log_msg("Inventario Arquitectura:")
    log_msg(f"  - ContentBasedFovea1DNet Parámetros:   {fovea_params}")
    log_msg(f"  - Baseline Causal Transformer Params:  {baseline_params}")
    
    opt_fovea = torch.optim.Adam(fovea_model.parameters(), lr=0.003)
    opt_base = torch.optim.Adam(baseline_model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 15
    history_fovea = []
    history_base = []
    
    log_msg("\n--- Iniciando Entrenamiento (FAST FEEDBACK en primeros 5 batches) ---")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        # 1. Entrenar Fóvea Dinámica 1D
        fovea_model.train()
        total_loss_f = 0.0
        for batch_idx, (seqs, targets) in enumerate(train_loader):
            seqs, targets = seqs.to(device), targets.to(device)
            opt_fovea.zero_grad()
            logits_f = fovea_model(seqs)
            pred_logits_f = logits_f[:, -1, :]
            loss_f = criterion(pred_logits_f, targets)
            loss_f.backward()
            opt_fovea.step()
            total_loss_f += loss_f.item()
            
            if epoch == 1 and batch_idx < 5:
                log_msg(f"  [FAST-FEEDBACK Epoch 1 Batch {batch_idx+1}] Fovea1D Loss: {loss_f.item():.4f}")
                
        # 2. Entrenar Baseline Transformer
        baseline_model.train()
        for seqs, targets in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            opt_base.zero_grad()
            logits_b = baseline_model(seqs)
            pred_logits_b = logits_b[:, -1, :]
            loss_b = criterion(pred_logits_b, targets)
            loss_b.backward()
            opt_base.step()
            
        # Eval
        fovea_model.eval()
        baseline_model.eval()
        
        f_correct, b_correct, total = 0, 0, 0
        with torch.no_grad():
            for seqs, targets in test_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                
                logits_f = fovea_model(seqs)[:, -1, :]
                pred_f = logits_f.argmax(dim=1)
                f_correct += (pred_f == targets).sum().item()
                
                logits_b = baseline_model(seqs)[:, -1, :]
                pred_b = logits_b.argmax(dim=1)
                b_correct += (pred_b == targets).sum().item()
                
                total += seqs.size(0)
                
        f_acc = 100.0 * f_correct / total
        b_acc = 100.0 * b_correct / total
        t_epoch = time.time() - t0
        
        log_msg(f"Epoch {epoch:02d}/{epochs:02d} | Fovea1D Acc: {f_acc:.2f}% | Baseline Transformer Acc: {b_acc:.2f}% | Tiempo: {t_epoch:.2f}s")
        history_fovea.append({"epoch": epoch, "acc": f_acc})
        history_base.append({"epoch": epoch, "acc": b_acc})

    # Resumen
    log_msg("\n=========================================================================")
    log_msg("  RESUMEN FINAL RESULTADOS EXPERIMENTO V365 (CONTENT-BASED FOVEA 1D)")
    log_msg("=========================================================================")
    log_msg(f"ContentBasedFovea1D -> Test Acc Final: {history_fovea[-1]['acc']:.2f}% (Params: {fovea_params})")
    log_msg(f"Baseline Transformer -> Test Acc Final: {history_base[-1]['acc']:.2f}% (Params: {baseline_params})")

    # Persistir JSON
    os.makedirs("results/raw", exist_ok=True)
    res_path = "results/raw/v365_copy_section_results.json"
    res_data = {
        "experiment_id": "v365_content_based_fovea_1d",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rigor_level": 1,
        "fovea_params": fovea_params,
        "baseline_params": baseline_params,
        "final_fovea_acc": history_fovea[-1]['acc'],
        "final_baseline_acc": history_base[-1]['acc'],
        "history_fovea": history_fovea,
        "history_base": history_base
    }
    with open(res_path, "w") as f:
        json.dump(res_data, f, indent=2)
    log_msg(f"Resultados guardados en: {res_path}")

    # Master Ledger
    ledger_path = "results/master_ledger.jsonl"
    ledger_entry = {
        "experiment_id": "v365",
        "fecha": time.strftime("%Y-%m-%d"),
        "familia": " geometrico_atencion_foveal ",
        "dataset": "MQAR Associative Recall 1D (T=128)",
        "n_eval": 600,
        "metric_name": "acc",
        "value": history_fovea[-1]['acc'],
        "SE": None,
        "params": fovea_params,
        "nivel_rigor": 1,
        "etiqueta": "ANCLA" if history_fovea[-1]['acc'] > history_base[-1]['acc'] else "SEÑAL"
    }
    with open(ledger_path, "a") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    log_msg(f"Registrado en Master Ledger: {ledger_path}")

if __name__ == '__main__':
    run_experiment()
