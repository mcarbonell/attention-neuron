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
# Core Layer: ContentBasedFoveaAttention1D (Escalable a K cabezas)
# -----------------------------------------------------------------------------
class ContentBasedFoveaAttention1D(nn.Module):
    def __init__(self, d_model=64, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.offset_predictor = nn.Linear(d_model, num_heads)
        self.radius_predictor = nn.Linear(d_model, num_heads)
        self.amplitude = nn.Parameter(torch.randn(num_heads) * 0.5)
        
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.size()
        K = self.num_heads
        
        i_indices = torch.arange(T, device=x.device).unsqueeze(1)
        j_indices = torch.arange(T, device=x.device).unsqueeze(0)
        delta_temporal = (i_indices - j_indices).float()
        causal_mask = (delta_temporal >= 0).float()
        
        offset_ratio = torch.sigmoid(self.offset_predictor(x)) # [B, T, K]
        radius_ratio = torch.sigmoid(self.radius_predictor(x)) * 0.5 + 0.05 # [B, T, K]
        
        i_scale = i_indices.float().unsqueeze(0)
        c_dyn = (offset_ratio * i_scale).permute(0, 2, 1).unsqueeze(-1) # [B, K, T, 1]
        r_dyn = (radius_ratio * i_scale + 1e-2).permute(0, 2, 1).unsqueeze(-1) # [B, K, T, 1]
        a = self.amplitude.view(1, K, 1, 1)
        
        delta_exp = delta_temporal.unsqueeze(0).unsqueeze(0)
        dist_from_center = torch.abs(delta_exp - c_dyn)
        base_weights = F.relu(1.0 - (dist_from_center / r_dyn)) * causal_mask.unsqueeze(0).unsqueeze(0)
        head_weights = base_weights * a # [B, K, T, T]
        
        d_head = D // K if D % K == 0 else D
        v = self.v_proj(x).view(B, T, K, -1).permute(0, 2, 1, 3)
        
        attn_out = torch.matmul(head_weights, v)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, D)
        
        return self.out_proj(attn_out)

# -----------------------------------------------------------------------------
# Modelo Escalable
# -----------------------------------------------------------------------------
class ScalableContentFovea1DNet(nn.Module):
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
# Barrido de Cabezas 1D (Sweep K = 2, 4, 8, 16, 32, 64)
# -----------------------------------------------------------------------------
def run_experiment():
    log_msg("=========================================================================")
    log_msg("  EXPERIMENTO: Barrido de Cabezas Fóvea 1D por Contenido (v366)")
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
    head_configs = [2, 4, 8, 16, 32, 64]
    epochs = 12
    sweep_results = {}
    
    for K in head_configs:
        log_msg(f"\n>>> Evaluando Configuración 1D: K = {K} Cabezas Foveales <<<")
        torch.manual_seed(seed)
        
        model = ScalableContentFovea1DNet(vocab_size=vocab_size, d_model=64, num_heads=K).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        criterion = nn.CrossEntropyLoss()
        
        k_history = []
        
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            
            for batch_idx, (seqs, targets) in enumerate(train_loader):
                seqs, targets = seqs.to(device), targets.to(device)
                optimizer.zero_grad()
                logits = model(seqs)
                pred_logits = logits[:, -1, :]
                loss = criterion(pred_logits, targets)
                loss.backward()
                optimizer.step()
                
                if epoch == 1 and batch_idx == 0:
                    log_msg(f"  [FAST-FEEDBACK K={K}] Epoch 1 Batch 1 Loss: {loss.item():.4f}")
                    
            # Eval
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for seqs, targets in test_loader:
                    seqs, targets = seqs.to(device), targets.to(device)
                    logits = model(seqs)[:, -1, :]
                    pred = logits.argmax(dim=1)
                    correct += (pred == targets).sum().item()
                    total += seqs.size(0)
                    
            acc = 100.0 * correct / total
            t_epoch = time.time() - t0
            
            log_msg(f"  K={K:02d} | Epoch {epoch:02d}/{epochs:02d} | Acc: {acc:.2f}% | Tiempo: {t_epoch:.2f}s")
            k_history.append({"epoch": epoch, "acc": acc})
            
        pei = k_history[-1]['acc'] / math.log10(total_params + 1)
        sweep_results[f"K_{K}"] = {
            "num_heads": K,
            "params": total_params,
            "final_acc": k_history[-1]['acc'],
            "pei": pei,
            "history": k_history
        }
        log_msg(f"=== RESULTADO K={K} -> Acc Final: {k_history[-1]['acc']:.2f}% | PEI: {pei:.2f} | Params: {total_params} ===")

    # Tabulación Final
    log_msg("\n=========================================================================")
    log_msg("  RESUMEN FINAL BARRIDO DE CABEZAS FOVEALES 1D (V366)")
    log_msg("=========================================================================")
    log_msg(f"{'Cabezas (K)':<12} | {'Parámetros':<12} | {'Test Acc (%)':<15} | {'PEI':<10}")
    log_msg("-" * 55)
    for K in head_configs:
        res = sweep_results[f"K_{K}"]
        log_msg(f"{res['num_heads']:<12} | {res['params']:<12} | {res['final_acc']:<15.2f} | {res['pei']:<10.2f}")

    # Persistir JSON
    os.makedirs("results/raw", exist_ok=True)
    res_path = "results/raw/v366_copy_section_results.json"
    with open(res_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    log_msg(f"Resultados guardados en: {res_path}")

    # Ledger
    best_k = max(head_configs, key=lambda k: sweep_results[f"K_{k}"]["final_acc"])
    best_res = sweep_results[f"K_{best_k}"]
    
    ledger_path = "results/master_ledger.jsonl"
    ledger_entry = {
        "experiment_id": "v366",
        "fecha": time.strftime("%Y-%m-%d"),
        "familia": " geometrico_atencion_foveal ",
        "dataset": "MQAR Associative Recall 1D (T=128)",
        "n_eval": 600,
        "metric_name": "acc",
        "value": best_res['final_acc'],
        "SE": None,
        "params": best_res['params'],
        "nivel_rigor": 1,
        "etiqueta": "ANCLA" if best_res['final_acc'] > 30.0 else "SEÑAL"
    }
    with open(ledger_path, "a") as f:
        f.write(json.dumps(ledger_entry) + "\n")
    log_msg(f"Registrado en Master Ledger (Mejor K={best_k}): {ledger_path}")

if __name__ == '__main__':
    run_experiment()
