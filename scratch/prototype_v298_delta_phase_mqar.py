"""
prototype_v298_delta_phase_mqar.py
======================================
V298: Corrected Delta Rule Phase Memory + Multi-Query MQAR + LR Sweep + Causal Conv1D in O(N).

Critical Fixes Applied (Audited by External Expert Agent):
  1. Outer Product Orientation Fix:
     - Old (BUG): M_ij = k_i * err_j  --> Readout gave key scaled by scalar (MSE > 250).
     - Fixed:     M_ij = err_i * k_j  --> Readout gives err * |k|^2 (Exact value retrieval!).
  2. Vector Norm Normalization (/ d_k):
     - Normalized update by d_k since sum_j |k_j|^2 = d_k.
  3. Head Dimension Capacity Margin (H=2, d_k=32):
     - Set H=2, d_k=32 for d_model=64. Provides 4x capacity margin for 8 KV pairs!
  4. Multi-Query MQAR Supervision (Standard Literature MQAR):
     - Queries multiple key-value pairs in the sequence (8x supervision signal per batch).
  5. LR Sweep Grid (1e-3, 2e-3, 4e-3, 8e-3):
     - Finds the optimal learning rate per architecture to eliminate LR calibration bias.

Models compared (~108k - 118k params):
  1. DeltaPhaseHolographic [CANDIDATE 1]: Corrected Matrix Delta Rule Phase Memory (H=2, d_k=32, O(N))
  2. ElementwiseDeltaPhaseHolographic [CANDIDATE 2]: Vector Delta Rule Phase Memory (H=2, d_k=32, O(N))
  3. PhaseSoftmaxHolographic [BASELINE 1]: Selective Scan + Match strength (v297)
  4. CausalAttentionMHA [BASELINE 2]: Causal Conv1D + Softmax MHA (O(N^2)) (Control >95%)
"""

import math
import time
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 1. Configuration ───────────────────────────────────────────────────
CFG = {
    "exp_id": "v298_delta_phase",
    "seq_len": 64,
    "num_kv_pairs": 8,
    "num_keys": 32,
    "num_vals": 32,
    "batch_size": 64,
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
def generate_mqar_batch(batch_size, seq_len=64, num_pairs=8, num_keys=32, num_vals=32, device=device):
    """
    Standard Multi-Query MQAR:
    KV insertion segment: K1 V1 K2 V2 ... K_n V_n
    Query segment: QUERY_MARKER Q_idx_1, QUERY_MARKER Q_idx_2, ...
    Target y contains V_idx_i at query token positions for multi-token supervision!
    """
    x = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    for b in range(batch_size):
        keys = torch.randperm(num_keys, device=device)[:num_pairs] + KEY_OFFSET
        vals = torch.randint(0, num_vals, (num_pairs,), device=device) + VAL_OFFSET
        
        # Interleave KV pairs at start of sequence
        kv_interleaved = torch.stack([keys, vals], dim=1).flatten()
        x[b, :len(kv_interleaved)] = kv_interleaved
        
        # Multi-query segment in second half of sequence
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
    def __init__(self, d_model, max_len=512):
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
    """
    Depthwise 1D Causal Convolution (kernel_size=4):
    Pairs Key_i and Value_{i+1} locally before memory state update.
    """
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

# ── Candidate 1: DeltaPhaseHolographicBlock (Corrected Matrix Delta Rule, H=2, d_k=32) ──
class DeltaPhaseHolographicBlock(nn.Module):
    """
    O(N) Matrix Delta Rule Phase Memory (H=2, d_k=32 for d_model=64):
      State M_t in C^{H x d_k x d_k}
      For each timestep t:
        1. k_conj_t = conj(K_t)
        2. v_old_t = Re( M_{t-1} * k_conj_t ) / d_k   [Readout value estimate]
        3. err_t   = V_t - v_old_t                     [Residual error signal]
        4. M_t     = M_{t-1} + beta_t * (err_t (x) K_t) [Corrected outer product write]
        5. R_t     = Re( M_t * conj(Q_t) ) / d_k      [Query readout]
    """
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
        
        K = torch.polar(torch.ones_like(theta_k), theta_k) # [B, L, H, d_k] complex
        Q = torch.polar(torch.ones_like(theta_q), theta_q) # [B, L, H, d_k] complex
        
        M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=conv_x.device)
        
        out_retrieved = []
        inv_dk = 1.0 / float(self.d_k)
        
        for t in range(L):
            k_t = K[:, t] # [B, H, d_k]
            q_t = Q[:, t] # [B, H, d_k]
            v_t = v[:, t] # [B, H, d_k] real
            beta_t = beta[:, t] # [B, H, 1, 1]
            
            k_conj = torch.conj(k_t)
            q_conj = torch.conj(q_t)
            
            # Readout prediction for k_t: M_{i,j} * k_conj_j -> sum_j err_i * k_j * k_conj_j = err_i * d_k
            # einsum: [B, H, i, j] * [B, H, j] -> [B, H, i]
            v_old = torch.einsum('bhij,bhj->bhi', M, k_conj).real * inv_dk
            
            # Compute residual error
            err = v_t - v_old # [B, H, d_k]
            
            # Correct outer product: M_ij = err_i * k_j
            # einsum: [B, H, i] * [B, H, j] -> [B, H, i, j]
            update = torch.einsum('bhi,bhj->bhij', err.to(torch.complex64), k_t)
            M = M + beta_t * update
            
            # Readout for query q_t
            ret = torch.einsum('bhij,bhj->bhi', M, q_conj).real * inv_dk
            out_retrieved.append(ret)
            
        retrieved = torch.stack(out_retrieved, dim=1).view(B, L, D)
        x = res + self.out_proj(retrieved)
        x = x + self.ffn(self.norm2(x))
        return x

# ── Candidate 2: ElementwiseDeltaPhaseHolographicBlock (Vector Delta Rule, H=2, d_k=32) ──
class ElementwiseDeltaPhaseHolographicBlock(nn.Module):
    """
    O(N) Vector Delta Rule Phase Memory (Diagonal Memory Contrast):
      State M_t in C^{H x d_k}
    """
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

# ── Baseline 1: PhaseSoftmaxHolographicBlock (v297 Winner) ─────────────
class PhaseSoftmaxHolographicBlock(nn.Module):
    def __init__(self, d_model, n_heads=8):
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
        self.lambda_proj = nn.Linear(d_model, d_model)
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
        lam = torch.sigmoid(self.lambda_proj(conv_x)).view(B, L, self.n_heads, self.d_k)
        
        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)
        bound = K * v
        
        M_v = torch.zeros(B, self.n_heads, self.d_k, dtype=torch.complex64, device=conv_x.device)
        M_k = torch.zeros(B, self.n_heads, self.d_k, dtype=torch.complex64, device=conv_x.device)
        M_m = torch.zeros(B, self.n_heads, self.d_k, dtype=torch.float32, device=conv_x.device)
        
        out_retrieved = []
        for t in range(L):
            lam_t = lam[:, t]
            M_v = lam_t * M_v + (1.0 - lam_t) * bound[:, t]
            M_k = lam_t * M_k + (1.0 - lam_t) * K[:, t]
            M_m = lam_t * M_m + (1.0 - lam_t) * 1.0
            
            raw = (torch.conj(Q[:, t]) * M_v).real
            match = (torch.conj(Q[:, t]) * M_k).abs()
            mass = M_m
            
            normalized = raw / (1e-4 + match / (1.0 + mass))
            out_retrieved.append(normalized)
            
        retrieved = torch.stack(out_retrieved, dim=1).view(B, L, D)
        x = res + self.out_proj(retrieved)
        x = x + self.ffn(self.norm2(x))
        return x

# ── Baseline 2: Standard Causal Attention (Softmax MHA + Conv1D O(N^2)) ──────────
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
        self.pe = SinCosPE(d_model, max_len=seq_len)
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
def evaluate_mqar(model, num_batches=15, cfg=CFG):
    model.eval()
    correct = 0
    total = 0
    for _ in range(num_batches):
        x, y = generate_mqar_batch(cfg["batch_size"], cfg["seq_len"], cfg["num_kv_pairs"], cfg["num_keys"], cfg["num_vals"], device)
        logits = model(x)
        
        # Evaluate accuracy ONLY on valid multi-query positions (where y != -100)
        valid_mask = (y != -100)
        if valid_mask.sum() > 0:
            preds = logits.argmax(dim=-1)
            correct += (preds[valid_mask] == y[valid_mask]).sum().item()
            total += valid_mask.sum().item()
            
    return (correct / total) * 100.0 if total > 0 else 0.0

# ── Benchmark Runner with LR Sweep ──────────────────────────────────────
def run_model_benchmark_with_lr_sweep(name, block_cls, kwargs={}, cfg=CFG):
    print(f"\n==================================================")
    print(f" Running Benchmark with LR Sweep: {name}")
    print(f"==================================================")
    
    best_overall = {
        "final_acc": -1.0,
        "best_lr": None,
        "wall_clock_time": 0.0,
        "eval_time": 0.0,
        "internal_overhead_time": 0.0,
        "params": 0
    }
    
    for lr in cfg["lr_grid"]:
        torch.manual_seed(cfg["seed"])
        model = MQARModel(block_cls, VOCAB_SIZE, cfg["d_model"], cfg["n_layers"], cfg["seq_len"], **kwargs).to(device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"] * cfg["steps_per_epoch"])
        
        start_time = time.time()
        train_time_accum = 0.0
        
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            total_loss = 0.0
            for step in range(1, cfg["steps_per_epoch"] + 1):
                x, y = generate_mqar_batch(cfg["batch_size"], cfg["seq_len"], cfg["num_kv_pairs"], cfg["num_keys"], cfg["num_vals"], device)
                
                t0 = time.time()
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=-100)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_time_accum += (time.time() - t0)
                
                total_loss += loss.item()
            
            avg_loss = total_loss / cfg["steps_per_epoch"]
            val_acc = evaluate_mqar(model, num_batches=10, cfg=cfg)
            
            if epoch == cfg["epochs"] or val_acc >= 99.5:
                print(f"  [LR = {lr:.0e}] Epoch {epoch:02d}/{cfg['epochs']:02d} | Train Loss: {avg_loss:.4f} | MQAR Target Acc: {val_acc:5.2f}%")
            
            if val_acc >= 99.5:
                print(f"   >>> Converged early at epoch {epoch} with {val_acc:.2f}% accuracy!")
                break
                
        wall_clock = time.time() - start_time
        final_acc = evaluate_mqar(model, num_batches=25, cfg=cfg)
        overhead = wall_clock - train_time_accum
        
        print(f" -> Result for LR={lr:.0e}: Final Acc = {final_acc:.2f}%")
        
        if final_acc > best_overall["final_acc"]:
            best_overall = {
                "model_name": name,
                "best_lr": lr,
                "final_acc": final_acc,
                "params": params,
                "wall_clock_time": round(wall_clock, 2),
                "train_time": round(train_time_accum, 2),
                "internal_overhead_time": round(overhead, 2),
                "epochs_run": epoch
            }
            
    print(f"--> BEST SUMMARY {name}: Best LR = {best_overall['best_lr']:.0e} | Final Acc = {best_overall['final_acc']:.2f}%")
    return best_overall

# ── Main ────────────────────────────────────────────────────────────────
def main():
    print(f"Starting V298 Corrected Delta Rule Phase Memory Benchmark on device: {device}")
    print(f"Sequence Length: {CFG['seq_len']} | KV Pairs: {CFG['num_kv_pairs']} | d_model: {CFG['d_model']} | Layers: {CFG['n_layers']} | LR Grid: {CFG['lr_grid']}")
    print(f"Chance level: 1/32 = 3.125%")
    
    results = []
    
    # REGLA DE ORO: CANDIDATES FIRST, BASELINES AFTER
    # Candidate 1: DeltaPhaseHolographic (Matrix Delta Rule, H=2, d_k=32)
    res_delta = run_model_benchmark_with_lr_sweep("DeltaPhaseHolographic [Candidate 1 - Corrected Matrix Delta Rule]", DeltaPhaseHolographicBlock, {"n_heads": 2}, CFG)
    results.append(res_delta)
    
    # Candidate 2: ElementwiseDeltaPhaseHolographic (Vector Delta Rule, H=2, d_k=32)
    res_elem_delta = run_model_benchmark_with_lr_sweep("ElementwiseDeltaPhaseHolographic [Candidate 2 - Vector Delta Rule]", ElementwiseDeltaPhaseHolographicBlock, {"n_heads": 2}, CFG)
    results.append(res_elem_delta)
    
    # Baseline 1: PhaseSoftmaxHolographic (v297 Baseline)
    res_ps = run_model_benchmark_with_lr_sweep("PhaseSoftmaxHolographic [Baseline 1 - Selective Scan]", PhaseSoftmaxHolographicBlock, {"n_heads": 8}, CFG)
    results.append(res_ps)
    
    # Baseline 2: Causal Attention MHA (Softmax MHA + Conv1D O(N^2))
    res_attn = run_model_benchmark_with_lr_sweep("CausalAttentionMHA [Baseline 2 - Softmax MHA + Conv1D O(N^2)]", CausalAttentionBlock, {}, CFG)
    results.append(res_attn)
    
    os.makedirs("results/raw", exist_ok=True)
    raw_path = "results/raw/v298_delta_phase.json"
    with open(raw_path, "w") as f:
        json.dump({"config": CFG, "chance_level": 3.125, "results": results}, f, indent=2)
    print(f"\nRaw results saved to {raw_path}")
    
    os.makedirs("results", exist_ok=True)
    ledger_path = "results/master_ledger.jsonl"
    with open(ledger_path, "a") as f:
        for r in results:
            entry = {
                "experiment_id": "v298_delta_phase",
                "fecha": time.strftime("%Y-%m-%d"),
                "familia": "corrected_delta_rule_phase vs attention",
                "dataset": f"Multi-Query MQAR (L={CFG['seq_len']}, pairs={CFG['num_kv_pairs']})",
                "n_eval": CFG["epochs"] * CFG["steps_per_epoch"] * CFG["batch_size"],
                "metric_name": f"target_acc_{r['model_name'].split()[0]}",
                "value": r["final_acc"],
                "best_lr": r["best_lr"],
                "SE": None,
                "params": r["params"],
                "nivel_rigor": 1,
                "etiqueta": "ANCLA" if r["final_acc"] > 90.0 else "SEÑAL"
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Master ledger updated at {ledger_path}")

if __name__ == "__main__":
    main()
