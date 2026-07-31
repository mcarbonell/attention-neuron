"""
prototype_v299_capacity_frontier.py
======================================
V299: Capacity Frontier & Phase Hypothesis Benchmark in O(N).

Core Scientific Question:
  Does Complex Phase Delta Rule (C^{d_k x d_k}) provide superior memory capacity per float parameter
  compared to Real-Valued Delta Rule (R^{d_k x d_k}, DeltaNet Vanilla) under strictly equal state memory (iso-floats)?

Load Curve Evaluation:
  - KV Pairs: num_pairs in {8, 16, 32, 64}
  - Sequence Length: seq_len = 8 * num_pairs in {64, 128, 256, 512}

Iso-Floats Memory Budget (d_model=64, H=2):
  - Complex Delta Rule:  d_k = 32 -> 2 * 32^2 = 2048 floats per head (4096 floats total)
  - Real DeltaNet Vanilla: d_k = 45 -> 45^2     = 2025 floats per head (4050 floats total)

Models Compared:
  1. ComplexDeltaPhaseHolographic [Complex Delta, H=2, d_k=32]
  2. RealDeltaNetVanilla          [Real DeltaNet, H=2, d_k=45] (Iso-floats)
  3. ElementwiseComplexDelta      [Complex Diagonal, H=2, d_k=32]
  4. CausalAttentionMHA           [Softmax MHA + Conv1D O(N^2)] (Reference Ceiling)

Methodology:
  - Multi-Query MQAR Data Generator
  - LR Sweep Grid: [1e-3, 2e-3, 4e-3, 8e-3] per architecture and load setting
  - Results logged to results/raw/v299_capacity_frontier.json & results/master_ledger.jsonl
"""

import math
import time
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 1. Base Configuration ──────────────────────────────────────────────
CFG = {
    "exp_id": "v299_capacity_frontier",
    "num_pairs_list": [8, 16, 32, 64],
    "num_keys": 64, # Expanded vocabulary to accommodate larger pair loads
    "num_vals": 64,
    "batch_size": 32,
    "d_model": 64,
    "n_layers": 3,
    "epochs": 15,
    "steps_per_epoch": 60,
    "lr_grid": [1e-3, 2e-3, 4e-3, 8e-3],
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

torch.manual_seed(CFG["seed"])
device = torch.device(CFG["device"])

PAD_ID = 0
KEY_OFFSET = 1
VAL_OFFSET = 1 + CFG["num_keys"]
QUERY_MARKER = VAL_OFFSET + CFG["num_vals"]
VOCAB_SIZE = QUERY_MARKER + 1

# ── 2. Multi-Query MQAR Data Generator ──────────────────────────────────
def generate_mqar_batch(batch_size, num_pairs=8, seq_len=64, num_keys=64, num_vals=64, device=device):
    x = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    for b in range(batch_size):
        keys = torch.randperm(num_keys, device=device)[:num_pairs] + KEY_OFFSET
        vals = torch.randint(0, num_vals, (num_pairs,), device=device) + VAL_OFFSET
        
        kv_interleaved = torch.stack([keys, vals], dim=1).flatten()
        x[b, :len(kv_interleaved)] = kv_interleaved
        
        query_perm = torch.randperm(num_pairs, device=device)
        curr_pos = len(kv_interleaved) + 2
        
        for q_idx in query_perm:
            if curr_pos + 1 >= seq_len:
                break
            target_k = keys[q_idx]
            target_v = vals[q_idx]
            
            x[b, curr_pos] = QUERY_MARKER
            x[b, curr_pos + 1] = target_k
            y[b, curr_pos + 1] = target_v
            curr_pos += 2

    return x, y

# ── 3. Common Components ──────────────────────────────────────────────
class SinCosPE(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]

class ShortCausalConv1D(nn.Module):
    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model
        )
        self.act = nn.SiLU()

    def forward(self, x):
        B, L, D = x.shape
        x_t = x.transpose(1, 2)
        conv_out = self.conv(x_t)[:, :, :L].transpose(1, 2)
        return x + self.act(conv_out)

class FFN(nn.Module):
    def __init__(self, d_model, expand=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.SiLU(),
            nn.Linear(d_model * expand, d_model)
        )
    def forward(self, x):
        return self.net(x)

# ── Candidate 1: ComplexDeltaPhaseHolographicBlock (C^{d_k x d_k}, H=2, d_k=32) ──
class ComplexDeltaPhaseHolographicBlock(nn.Module):
    def __init__(self, d_model, n_heads=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        
        self.theta_k_proj = nn.Linear(d_model, d_model)
        self.theta_q_proj = nn.Linear(d_model, d_model)
        self.val_proj = nn.Linear(d_model, d_model)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x
        normed = self.norm1(x)
        conv_x = self.causal_conv(normed)
        B, L, D = conv_x.shape
        
        theta_k = self.theta_k_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        theta_q = self.theta_q_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        v = self.val_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        beta = torch.sigmoid(self.beta_proj(conv_x)).view(B, L, self.n_heads, 1, 1)
        
        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)
        
        M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=conv_x.device)
        out_retrieved = []
        inv_dk = 1.0 / float(self.d_k)
        
        for t in range(L):
            k_t = K[:, t]
            q_t = Q[:, t]
            v_t = v[:, t]
            beta_t = beta[:, t]
            
            k_conj = torch.conj(k_t)
            q_conj = torch.conj(q_t)
            
            v_old = torch.einsum('bhij,bhj->bhi', M, k_conj).real * inv_dk
            err = v_t - v_old
            
            update = torch.einsum('bhi,bhj->bhij', err.to(torch.complex64), k_t)
            M = M + beta_t * update
            
            ret = torch.einsum('bhij,bhj->bhi', M, q_conj).real * inv_dk
            out_retrieved.append(ret)
            
        retrieved = torch.stack(out_retrieved, dim=1).view(B, L, D)
        x = res + self.out_proj(retrieved)
        x = x + self.ffn(self.norm2(x))
        return x

# ── Candidate 2: RealDeltaNetVanillaBlock (R^{d_k x d_k}, H=2, d_k=45 Iso-Floats) ──
class RealDeltaNetVanillaBlock(nn.Module):
    """
    Real-Valued DeltaNet Vanilla (R^{d_k x d_k}):
    Iso-floats matched: d_k=45 -> 45^2 = 2025 floats per head (vs 2*32^2 = 2048 floats complex).
    """
    def __init__(self, d_model, n_heads=2, d_k_real=45):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k_real
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        
        self.k_proj = nn.Linear(d_model, n_heads * self.d_k)
        self.q_proj = nn.Linear(d_model, n_heads * self.d_k)
        self.val_proj = nn.Linear(d_model, n_heads * self.d_k)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(n_heads * self.d_k, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x
        normed = self.norm1(x)
        conv_x = self.causal_conv(normed)
        B, L, D = conv_x.shape
        
        # Real L2-normalized Key and Query vectors
        k_raw = self.k_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        q_raw = self.q_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        v = self.val_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        beta = torch.sigmoid(self.beta_proj(conv_x)).view(B, L, self.n_heads, 1, 1)
        
        K = F.normalize(k_raw, p=2, dim=-1)
        Q = F.normalize(q_raw, p=2, dim=-1)
        
        M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.float32, device=conv_x.device)
        out_retrieved = []
        
        for t in range(L):
            k_t = K[:, t] # [B, H, d_k]
            q_t = Q[:, t] # [B, H, d_k]
            v_t = v[:, t] # [B, H, d_k]
            beta_t = beta[:, t]
            
            # Real matrix-vector readout: v_old = M * k_t
            v_old = torch.einsum('bhij,bhj->bhi', M, k_t)
            err = v_t - v_old
            
            # Real outer product write: M_ij = err_i * k_j
            update = torch.einsum('bhi,bhj->bhij', err, k_t)
            M = M + beta_t * update
            
            ret = torch.einsum('bhij,bhj->bhi', M, q_t)
            out_retrieved.append(ret)
            
        retrieved = torch.stack(out_retrieved, dim=1).view(B, L, self.n_heads * self.d_k)
        x = res + self.out_proj(retrieved)
        x = x + self.ffn(self.norm2(x))
        return x

# ── Candidate 3: ElementwiseComplexDeltaBlock (C^{d_k}, H=2, d_k=32 Vector Memory) ──
class ElementwiseComplexDeltaBlock(nn.Module):
    def __init__(self, d_model, n_heads=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        
        self.theta_k_proj = nn.Linear(d_model, d_model)
        self.theta_q_proj = nn.Linear(d_model, d_model)
        self.val_proj = nn.Linear(d_model, d_model)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x
        normed = self.norm1(x)
        conv_x = self.causal_conv(normed)
        B, L, D = conv_x.shape
        
        theta_k = self.theta_k_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        theta_q = self.theta_q_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        v = self.val_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        beta = torch.sigmoid(self.beta_proj(conv_x)).view(B, L, self.n_heads, 1)
        
        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)
        
        M = torch.zeros(B, self.n_heads, self.d_k, dtype=torch.complex64, device=conv_x.device)
        out_retrieved = []
        for t in range(L):
            k_t = K[:, t]
            q_t = Q[:, t]
            v_t = v[:, t]
            beta_t = beta[:, t]
            
            v_old = (M * torch.conj(k_t)).real
            err = v_t - v_old
            M = M + beta_t * (k_t * err)
            
            ret = (M * torch.conj(q_t)).real
            out_retrieved.append(ret)
            
        retrieved = torch.stack(out_retrieved, dim=1).view(B, L, D)
        x = res + self.out_proj(retrieved)
        x = x + self.ffn(self.norm2(x))
        return x

# ── Baseline Control: Standard Causal Attention (Softmax MHA + Conv1D O(N^2)) ──────
class CausalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x
        normed = self.norm1(x)
        conv_x = self.causal_conv(normed)
        L = conv_x.shape[1]
        mask = torch.triu(torch.full((L, L), float('-inf'), device=conv_x.device), diagonal=1)
        attn_out, _ = self.mha(conv_x, conv_x, conv_x, attn_mask=mask, is_causal=True)
        x = res + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

# ── Full Model Container ────────────────────────────────────────────────
class MQARModel(nn.Module):
    def __init__(self, block_cls, vocab_size, d_model, n_layers, seq_len=64, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = SinCosPE(d_model, max_len=max(512, seq_len))
        self.blocks = nn.ModuleList([block_cls(d_model, **kwargs) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.pe(self.emb(x))
        for block in self.blocks:
            h = block(h)
        h = self.norm_f(h)
        return self.head(h)

# ── Evaluation Function ─────────────────────────────────────────────────
@torch.no_grad()
def evaluate_mqar(model, num_pairs, seq_len, num_batches=15, cfg=CFG):
    model.eval()
    correct = 0
    total = 0
    for _ in range(num_batches):
        x, y = generate_mqar_batch(cfg["batch_size"], num_pairs, seq_len, cfg["num_keys"], cfg["num_vals"], device)
        logits = model(x)
        
        valid_mask = (y != -100)
        if valid_mask.sum() > 0:
            preds = logits.argmax(dim=-1)
            correct += (preds[valid_mask] == y[valid_mask]).sum().item()
            total += valid_mask.sum().item()
            
    return (correct / total) * 100.0 if total > 0 else 0.0

# ── Benchmark Runner Across Capacity Load Curve ─────────────────────────
def run_model_benchmark_load_curve(name, block_cls, kwargs={}, cfg=CFG):
    print(f"\n==================================================")
    print(f" Running Load Curve Benchmark: {name}")
    print(f"==================================================")
    
    load_results = {}
    
    for num_pairs in cfg["num_pairs_list"]:
        seq_len = 8 * num_pairs
        print(f"\n --- Testing Load: N_pairs = {num_pairs} | Sequence Length = {seq_len} ---")
        
        best_lr_acc = -1.0
        best_lr_val = None
        
        for lr in cfg["lr_grid"]:
            torch.manual_seed(cfg["seed"])
            model = MQARModel(block_cls, VOCAB_SIZE, cfg["d_model"], cfg["n_layers"], seq_len, **kwargs).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"] * cfg["steps_per_epoch"])
            
            for epoch in range(1, cfg["epochs"] + 1):
                model.train()
                for step in range(1, cfg["steps_per_epoch"] + 1):
                    x, y = generate_mqar_batch(cfg["batch_size"], num_pairs, seq_len, cfg["num_keys"], cfg["num_vals"], device)
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=-100)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                
                val_acc = evaluate_mqar(model, num_pairs, seq_len, num_batches=10, cfg=cfg)
                if val_acc >= 99.5:
                    break
                    
            final_acc = evaluate_mqar(model, num_pairs, seq_len, num_batches=20, cfg=cfg)
            if final_acc > best_lr_acc:
                best_lr_acc = final_acc
                best_lr_val = lr
                
        print(f"  --> Best Acc for N_pairs={num_pairs}: {best_lr_acc:.2f}% (Best LR = {best_lr_val:.0e})")
        load_results[f"pairs_{num_pairs}"] = {
            "num_pairs": num_pairs,
            "seq_len": seq_len,
            "best_acc": best_lr_acc,
            "best_lr": best_lr_val
        }
        
    return {
        "model_name": name,
        "load_curve": load_results
    }

# ── Main ────────────────────────────────────────────────────────────────
def main():
    print(f"Starting V299 Capacity Frontier Benchmark on device: {device}")
    print(f"Load Curve Pairs: {CFG['num_pairs_list']} | LR Grid: {CFG['lr_grid']}")
    print(f"Iso-Floats Budget per Head: Complex (d_k=32 -> 2048 floats) vs Real (d_k=45 -> 2025 floats)")
    
    results = []
    
    # Candidate 1: Complex Delta Phase (Complex Matrix Delta, H=2, d_k=32)
    res_complex = run_model_benchmark_load_curve("ComplexDeltaPhaseHolographic [Complex Delta, d_k=32]", ComplexDeltaPhaseHolographicBlock, {"n_heads": 2}, CFG)
    results.append(res_complex)
    
    # Candidate 2: Real DeltaNet Vanilla (Real Matrix Delta, H=2, d_k=45 Iso-Floats)
    res_real = run_model_benchmark_load_curve("RealDeltaNetVanilla [Real DeltaNet, d_k=45 Iso-Floats]", RealDeltaNetVanillaBlock, {"n_heads": 2, "d_k_real": 45}, CFG)
    results.append(res_real)
    
    # Candidate 3: Elementwise Complex Delta (Diagonal Vector Memory, H=2, d_k=32)
    res_elem = run_model_benchmark_load_curve("ElementwiseComplexDelta [Complex Vector Memory]", ElementwiseComplexDeltaBlock, {"n_heads": 2}, CFG)
    results.append(res_elem)
    
    # Baseline Control: Causal Attention MHA (Softmax MHA + Conv1D O(N^2))
    res_attn = run_model_benchmark_load_curve("CausalAttentionMHA [Softmax MHA + Conv1D Control]", CausalAttentionBlock, {}, CFG)
    results.append(res_attn)
    
    os.makedirs("results/raw", exist_ok=True)
    raw_path = "results/raw/v299_capacity_frontier.json"
    with open(raw_path, "w") as f:
        json.dump({"config": CFG, "results": results}, f, indent=2)
    print(f"\nRaw results saved to {raw_path}")

    # Update Master Ledger with audit notes for v292-v297
    os.makedirs("results", exist_ok=True)
    ledger_path = "results/master_ledger.jsonl"
    with open(ledger_path, "a") as f:
        for r in results:
            entry = {
                "experiment_id": "v299_capacity_frontier",
                "fecha": time.strftime("%Y-%m-%d"),
                "familia": "capacity_frontier_complex_vs_real_delta",
                "dataset": "Multi-Query MQAR Capacity Stress Curve (N_pairs=8..64)",
                "metric_name": f"load_curve_{r['model_name'].split()[0]}",
                "value": r["load_curve"],
                "nivel_rigor": 2,
                "etiqueta": "ANCLA"
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Master ledger updated at {ledger_path}")

if __name__ == "__main__":
    main()
