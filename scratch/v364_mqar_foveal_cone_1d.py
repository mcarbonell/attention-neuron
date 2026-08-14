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
# Dataset Sintético: Multi-Query Associative Recall (MQAR 1D)
# -----------------------------------------------------------------------------
class MQARDataset(Dataset):
    """
    Genera secuencias MQAR (Associative Recall):
    - Secuencia de longitud T=128.
    - Se insertan N_pairs parejas aleatorias de (Key, Value) en posiciones aleatorias de la secuencia.
    - Los demás tokens se rellenan con tokens de ruido/filler.
    - Al final de la secuencia se coloca el Query de una de las Keys insertadas.
    - El objetivo es predecir el Value asociado a dicha Key.
    """
    def __init__(self, num_samples=3000, seq_len=128, num_kv_pairs=4, num_keys=16, num_values=16, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_kv_pairs = num_kv_pairs
        
        # Mapeo de Vocabulario:
        # 0: Pad/BOS
        # 1 .. num_keys: Key Tokens
        # num_keys+1 .. num_keys+num_values: Value Tokens
        # num_keys+num_values+1 .. : Filler Noise Tokens (16 fillers)
        self.key_offset = 1
        self.val_offset = 1 + num_keys
        self.filler_offset = 1 + num_keys + num_values
        self.vocab_size = 1 + num_keys + num_values + 16
        
        self.sequences = torch.zeros(num_samples, seq_len, dtype=torch.long)
        self.targets = torch.zeros(num_samples, dtype=torch.long)
        
        for i in range(num_samples):
            # Rellenar con tokens de ruido
            seq = torch.randint(self.filler_offset, self.vocab_size, (seq_len,))
            
            # Elegir parejas (Key, Value) únicas
            keys = torch.randperm(num_keys)[:num_kv_pairs] + self.key_offset
            vals = torch.randperm(num_values)[:num_kv_pairs] + self.val_offset
            
            # Posiciones aleatorias sin solapamiento (con espacio para Key y Value contiguos)
            available_pos = list(range(0, seq_len - 10, 2))
            chosen_pos_indices = torch.randperm(len(available_pos))[:num_kv_pairs]
            
            for j in range(num_kv_pairs):
                pos = available_pos[chosen_pos_indices[j]]
                seq[pos] = keys[j]
                seq[pos + 1] = vals[j]
                
            # Seleccionar una pareja objetivo para el Query final
            query_pair_idx = torch.randint(0, num_kv_pairs, (1,)).item()
            target_key = keys[query_pair_idx]
            target_val = vals[query_pair_idx]
            
            # El último token es la Query Key
            seq[-1] = target_key
            self.sequences[i] = seq
            self.targets[i] = target_val

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# -----------------------------------------------------------------------------
# Core Layer: MultiHeadConeAttention1D (Atención Cono Causal 1D Vectorizada)
# -----------------------------------------------------------------------------
class MultiHeadConeAttention1D(nn.Module):
    """
    Atención Cono/Foveal 1D Causal Vectorizada.
    
    Cada una de las K cabezas atencionales define:
      - Centro de atención temporal: offset C_k (cuántos pasos hacia atrás mira)
      - Radio de atención temporal: R_k (ancho de ventana de decaimiento)
      - Amplitud A_k (excitación / inhibición)
      
    No calcula la matriz Q K^T de (T x T). En su lugar, aplica una ponderación
    de cono temporal causal vectorizada sobre la dimensión de secuencia T.
    """
    def __init__(self, d_model=64, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Parámetros entrenables de las K cabezas: centro offset C_k, radio R_k, amplitud A_k
        self.center_offset = nn.Parameter(torch.rand(num_heads) * 30.0) # inicializado en 0-30 pasos atrás
        self.radius = nn.Parameter(torch.ones(num_heads) * 10.0)        # ventana inicial de 10 tokens
        self.amplitude = nn.Parameter(torch.randn(num_heads) * 0.5)
        
        # Proyecciones lineales V y Out
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.size()
        K = self.num_heads
        
        # 1. Matriz de distancia temporal causal D_ij = i - j (para j <= i)
        # Forma: [T, T]
        i_indices = torch.arange(T, device=x.device).unsqueeze(1) # [T, 1]
        j_indices = torch.arange(T, device=x.device).unsqueeze(0) # [1, T]
        delta_temporal = i_indices - j_indices # [T, T] (pasos atrás de i a j)
        
        # Máscara causal: solo atiende al presente y pasado (delta_temporal >= 0)
        causal_mask = (delta_temporal >= 0).float()
        
        # 2. Calcular pesos de cono para cada una de las K cabezas [K, T, T]
        c = self.center_offset.view(K, 1, 1) # [K, 1, 1]
        r = F.softplus(self.radius).view(K, 1, 1) + 1e-3 # [K, 1, 1] (> 0)
        a = self.amplitude.view(K, 1, 1)
        
        # Distancia desde el centro de atención |(i - j) - c_k|
        dist_from_center = torch.abs(delta_temporal.unsqueeze(0) - c) # [K, T, T]
        
        # Cono de decaimiento lineal con ReLU
        base_weight = F.relu(1.0 - (dist_from_center / r)) * causal_mask.unsqueeze(0) # [K, T, T]
        head_weights = base_weight * a # [K, T, T]
        
        # 3. Proyección de V y agregación [B, T, D] -> [B, K, T, D/K]
        d_head = D // K
        v = self.v_proj(x).view(B, T, K, d_head).permute(0, 2, 1, 3) # [B, K, T, d_head]
        
        # Multiplicación tensorial vectorizada: [B, K, T, T] @ [B, K, T, d_head] -> [B, K, T, d_head]
        # Expandir head_weights a batch: [1, K, T, T]
        head_weights_b = head_weights.unsqueeze(0).repeat(B, 1, 1, 1) # [B, K, T, T]
        
        attn_out = torch.matmul(head_weights_b, v) # [B, K, T, d_head]
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, D) # [B, T, D]
        
        return self.out_proj(attn_out)

# -----------------------------------------------------------------------------
# Modelo MultiHeadCone1DNet (V364)
# -----------------------------------------------------------------------------
class MultiHeadCone1DNet(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(256, d_model)
        
        self.cone_attn = MultiHeadConeAttention1D(d_model=d_model, num_heads=num_heads)
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
        
        # Cono 1D residual
        attn_out = self.cone_attn(h)
        h = self.norm(h + attn_out)
        
        logits = self.mlp(h)
        return logits

# -----------------------------------------------------------------------------
# Baseline: Causal Transformer (1 Capa, Softmax QK^T)
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
        
        # Máscara Causal
        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        h = self.transformer(h, mask=causal_mask)
        logits = self.output(h)
        return logits

# -----------------------------------------------------------------------------
# Bucle Principal de Entrenamiento
# -----------------------------------------------------------------------------
def run_experiment():
    log_msg("=========================================================================")
    log_msg("  EXPERIMENTO 1: MQAR / Associative Recall 1D (v364_mqar_foveal_cone_1d)")
    log_msg("=========================================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_msg(f"Metadatos: Python {sys.version.split()[0]} | PyTorch {torch.__version__} | Dispositivo: {device}")
    
    seed = 42
    torch.manual_seed(seed)
    
    # Dataset MQAR T=128
    seq_len = 128
    train_dataset = MQARDataset(num_samples=3000, seq_len=seq_len, num_kv_pairs=4, seed=seed)
    test_dataset = MQARDataset(num_samples=600, seq_len=seq_len, num_kv_pairs=4, seed=seed+1)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=600, shuffle=False)
    
    vocab_size = train_dataset.vocab_size
    log_msg(f"Configuración MQAR: VocabSize={vocab_size} | SeqLen={seq_len} | Parejas K-V=4")
    
    # Instanciar modelos
    cone_model = MultiHeadCone1DNet(vocab_size=vocab_size, d_model=64, num_heads=8).to(device)
    baseline_model = BaselineCausalTransformer(vocab_size=vocab_size, d_model=64, num_heads=4).to(device)
    
    cone_params = sum(p.numel() for p in cone_model.parameters() if p.requires_grad)
    baseline_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    
    log_msg("Inventario Arquitectura:")
    log_msg(f"  - MultiHead Cone1DNet Parámetros:     {cone_params}")
    log_msg(f"  - Baseline Causal Transformer Params: {baseline_params}")
    
    opt_cone = torch.optim.Adam(cone_model.parameters(), lr=0.003)
    opt_base = torch.optim.Adam(baseline_model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 15
    history_cone = []
    history_base = []
    
    log_msg("\n--- Iniciando Entrenamiento (FAST FEEDBACK en primeros 5 batches) ---")
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        # 1. Entrenar Cone 1D
        cone_model.train()
        total_loss_c = 0.0
        for batch_idx, (seqs, targets) in enumerate(train_loader):
            seqs, targets = seqs.to(device), targets.to(device)
            opt_cone.zero_grad()
            logits_c = cone_model(seqs) # [B, T, Vocab]
            
            # La predicción es el último token de la secuencia (donde está la Query Key)
            pred_logits_c = logits_c[:, -1, :] # [B, Vocab]
            loss_c = criterion(pred_logits_c, targets)
            loss_c.backward()
            opt_cone.step()
            total_loss_c += loss_c.item()
            
            if epoch == 1 and batch_idx < 5:
                log_msg(f"  [FAST-FEEDBACK Epoch 1 Batch {batch_idx+1}] Cone1D Loss: {loss_c.item():.4f}")
                
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
        cone_model.eval()
        baseline_model.eval()
        
        c_correct, b_correct, total = 0, 0, 0
        with torch.no_grad():
            for seqs, targets in test_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                
                logits_c = cone_model(seqs)[:, -1, :]
                pred_c = logits_c.argmax(dim=1)
                c_correct += (pred_c == targets).sum().item()
                
                logits_b = baseline_model(seqs)[:, -1, :]
                pred_b = logits_b.argmax(dim=1)
                b_correct += (pred_b == targets).sum().item()
                
                total += seqs.size(0)
                
        c_acc = 100.0 * c_correct / total
        b_acc = 100.0 * b_correct / total
        t_epoch = time.time() - t0
        
        log_msg(f"Epoch {epoch:02d}/{epochs:02d} | Cone1D Acc: {c_acc:.2f}% | Baseline Transformer Acc: {b_acc:.2f}% | Tiempo: {t_epoch:.2f}s")
        history_cone.append({"epoch": epoch, "acc": c_acc})
        history_base.append({"epoch": epoch, "acc": b_acc})

    # Resumen
    log_msg("\n=========================================================================")
    log_msg("  RESUMEN FINAL RESULTADOS EXPERIMENTO 1 (MQAR 1D ASSOCIATIVE RECALL)")
    log_msg("=========================================================================")
    log_msg(f"MultiHead Cone1DNet  -> Test Acc Final: {history_cone[-1]['acc']:.2f}% (Params: {cone_params})")
    log_msg(f"Baseline Transformer -> Test Acc Final: {history_base[-1]['acc']:.2f}% (Params: {baseline_params})")

    # Persistir JSON
    os.makedirs("results/raw", exist_ok=True)
    res_path = "results/raw/v364_copy_section_results.json"
    res_data = {
        "experiment_id": "v364_mqar_foveal_cone_1d",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rigor_level": 1,
        "cone_params": cone_params,
        "baseline_params": baseline_params,
        "final_cone_acc": history_cone[-1]['acc'],
        "final_baseline_acc": history_base[-1]['acc'],
        "history_cone": history_cone,
        "history_base": history_base
    }
    with open(res_path, "w") as f:
        json.dump(res_data, f, indent=2)
    log_msg(f"Resultados guardados en: {res_path}")

    # Master Ledger
    ledger_path = "results/master_ledger.jsonl"
    ledger_entry = {
        "experiment_id": "v364",
        "fecha": time.strftime("%Y-%m-%d"),
        "familia": " geometrico_atencion_foveal ",
        "dataset": "MQAR Associative Recall 1D (T=128)",
        "n_eval": 600,
        "metric_name": "acc",
        "value": history_cone[-1]['acc'],
        "SE": None,
        "params": cone_params,
        "nivel_rigor": 1,
        "etiqueta": "ANCLA" if history_cone[-1]['acc'] > 80.0 else "SEÑAL"
    }
    with open(ledger_path, "a") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    log_msg(f"Registrado en Master Ledger: {ledger_path}")

if __name__ == '__main__':
    run_experiment()
