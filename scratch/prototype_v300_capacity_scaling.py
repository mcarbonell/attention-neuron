"""
prototype_v300_capacity_scaling.py
======================================
V300: Capacity Scaling Law Benchmark for Complex Phase Delta in O(N).

Core Scientific Question:
  How does the associative recall capacity frontier (max KV pairs stored with >95% accuracy)
  scale empirical capacity as a function of key dimension d_k in {32, 64, 128}?
  Does Complex Phase Delta (C^{d_k x d_k}) maintain its state density lead over
  Real-Valued DeltaNet Vanilla (R^{d_k x d_k}) under strict iso-floats state matching?

Dimension & Iso-Floats Setup (H=2 heads):
  - d_k_complex = 32  -> 2 * 32^2  = 2,048 floats/head  | Iso-Real d_k_real = 45  (45^2 = 2,025 floats)
  - d_k_complex = 64  -> 2 * 64^2  = 8,192 floats/head  | Iso-Real d_k_real = 90  (90^2 = 8,100 floats)
  - d_k_complex = 128 -> 2 * 128^2 = 32,768 floats/head | Iso-Real d_k_real = 181 (181^2 = 32,761 floats)

Load Curve Sweeps:
  - KV Pairs: num_pairs in {32, 64, 128, 256}
  - Sequence Lengths: seq_len = 8 * num_pairs in {256, 512, 1024, 2048}

Models Compared:
  1. ComplexDeltaPhaseHolographic [Complex Delta, H=2]
  2. RealDeltaNetVanilla          [Real DeltaNet, H=2] (Iso-floats per d_k)
  3. CausalAttentionMHA           [Softmax MHA + Conv1D O(N^2)] (Reference Ceiling)

Methodology:
  - Multi-Query MQAR Data Generator with expanded vocabulary (num_keys=256, num_vals=256)
  - LR Grid: [2e-3, 4e-3]
  - Results logged to results/raw/v300_capacity_scaling.json & results/master_ledger.jsonl
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
    "exp_id": "v300_capacity_scaling",
    "d_k_list": [32, 64, 128],
    "iso_floats_map": {
        32: {"dk_complex": 32, "dk_real": 45, "floats_c": 2048, "floats_r": 2025},
        64: {"dk_complex": 64, "dk_real": 90, "floats_c": 8192, "floats_r": 8100},
        128: {"dk_complex": 128, "dk_real": 181, "floats_c": 32768, "floats_r": 32761}
    },
    "num_pairs_list": [32, 64, 128, 256],
    "num_keys": 256,
    "num_vals": 256,
    "batch_size": 32,
    "n_layers": 3,
    "epochs": 15,
    "steps_per_epoch": 50,
    "lr_grid": [2e-3, 4e-3],
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
def generate_mqar_batch(batch_size, num_pairs=32, seq_len=256, num_keys=256, num_vals=256, device=device):
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
    def __init__(self, d_model, max_len=4096):
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

# ── Candidate 1: ComplexDeltaPhaseHolographicBlock ──
class ComplexDeltaPhaseHolographicBlock(nn.Module):
    def __init__(self, d_model, n_heads=2, d_k=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        
        self.theta_k_proj = nn.Linear(d_model, n_heads * d_k)
        self.theta_q_proj = nn.Linear(d_model, n_heads * d_k)
        self.val_proj = nn.Linear(d_model, n_heads * d_k)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(n_heads * d_k, d_model)
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
            
        retrieved = torch.stack(out_retrieved, dim=1).view(B, L, self.n_heads * self.d_k)
        x = res + self.out_proj(retrieved)
        x = x + self.ffn(self.norm2(x))
        return x

# ── Candidate 2: RealDeltaNetVanillaBlock (Iso-Floats Matched) ──
class RealDeltaNetVanillaBlock(nn.Module):
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
        
        k_raw = self.k_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        q_raw = self.q_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        v = self.val_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        beta = torch.sigmoid(self.beta_proj(conv_x)).view(B, L, self.n_heads, 1, 1)
        
        K = F.normalize(k_raw, p=2, dim=-1)
        Q = F.normalize(q_raw, p=2, dim=-1)
        
        M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.float32, device=conv_x.device)
        out_retrieved = []
        
        for t in range(L):
            k_t = K[:, t]
            q_t = Q[:, t]
            v_t = v[:, t]
            beta_t = beta[:, t]
            
            v_old = torch.einsum('bhij,bhj->bhi', M, k_t)
            err = v_t - v_old
            
            update = torch.einsum('bhi,bhj->bhij', err, k_t)
            M = M + beta_t * update
            
            ret = torch.einsum('bhij,bhj->bhi', M, q_t)
            out_retrieved.append(ret)
            
        retrieved = torch.stack(out_retrieved, dim=1).view(B, L, self.n_heads * self.d_k)
        x = res + self.out_proj(retrieved)
        x = x + self.ffn(self.norm2(x))
        return x

# ── Reference Ceiling: CausalAttentionBlock ──
class CausalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x
        normed = self.norm1(x)
        conv_x = self.causal_conv(normed)
        B, L, D = conv_x.shape
        
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=conv_x.device), diagonal=1)
        attn_out, _ = self.mha(conv_x, conv_x, conv_x, attn_mask=causal_mask, is_causal=False)
        
        x = res + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

# ── Full Model Wrapper ──────────────────────────────────────────────────
class SequenceModel(nn.Module):
    def __init__(self, block_cls, vocab_size, d_model, n_layers=3, block_kwargs=None):
        super().__init__()
        if block_kwargs is None:
            block_kwargs = {}
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = SinCosPE(d_model)
        self.layers = nn.ModuleList([block_cls(d_model=d_model, **block_kwargs) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.pe(self.emb(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)

# ── Training & Evaluation Loop ──────────────────────────────────────────
def train_and_eval(model, num_pairs, seq_len, lr, epochs=15, steps_per_epoch=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    start_time = time.time()

    for ep in range(epochs):
        model.train()
        for step in range(steps_per_epoch):
            x, y = generate_mqar_batch(CFG["batch_size"], num_pairs=num_pairs, seq_len=seq_len,
                                      num_keys=CFG["num_keys"], num_vals=CFG["num_vals"])
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # FAST FEEDBACK (GEMINI Rule: print progress in first 5 batches of Epoch 1)
            if ep == 0 and step < 5:
                print(f"      [Fast Feedback] Ep 1, Batch {step+1}: Loss = {loss.item():.4f}")

    eval_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(10): # 10 eval batches
            x, y = generate_mqar_batch(CFG["batch_size"], num_pairs=num_pairs, seq_len=seq_len,
                                      num_keys=CFG["num_keys"], num_vals=CFG["num_vals"])
            logits = model(x)
            preds = logits.argmax(dim=-1)
            mask = (y != -100)
            correct += (preds[mask] == y[mask]).sum().item()
            total += mask.sum().item()

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    return acc, eval_time

# ── Main Experiment Suite ───────────────────────────────────────────────
def run_experiment_suite():
    print("=" * 80)
    print(f"EXPERIMENT METADATA HEADER")
    print(f"  Exp ID: {CFG['exp_id']}")
    print(f"  Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (Torch DirectML/Native)'}")
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  Base File: scratch/prototype_v300_capacity_scaling.py")
    print(f"  d_k Complex Sweeps: {CFG['d_k_list']}")
    print(f"  KV Pair Sweeps: {CFG['num_pairs_list']}")
    print("=" * 80)

    results_matrix = {}

    for d_k in CFG["d_k_list"]:
        iso_info = CFG["iso_floats_map"][d_k]
        dk_c = iso_info["dk_complex"]
        dk_r = iso_info["dk_real"]
        d_model = 2 * dk_c # Scale d_model with head dimension (H=2)
        
        print(f"\n==========================================================================")
        print(f"--- SWEEP: d_k = {dk_c} (d_model = {d_model}) ---")
        print(f"    Complex Head: {dk_c}x{dk_c} (C) -> {iso_info['floats_c']} floats/head")
        print(f"    Iso-Real Head: {dk_r}x{dk_r} (R) -> {iso_info['floats_r']} floats/head")
        print(f"==========================================================================")

        results_matrix[d_k] = {}

        model_specs = [
            ("ComplexDeltaPhaseHolographic", ComplexDeltaPhaseHolographicBlock, {"d_k": dk_c}),
            ("RealDeltaNetVanilla", RealDeltaNetVanillaBlock, {"d_k_real": dk_r}),
            ("CausalAttentionMHA", CausalAttentionBlock, {})
        ]

        for num_pairs in CFG["num_pairs_list"]:
            seq_len = 8 * num_pairs
            print(f"\n  >>> Evaluating Load: {num_pairs} Pairs (Seq Len L={seq_len}) <<<")

            for name, block_cls, block_kwargs in model_specs:
                best_acc = -1.0
                best_lr = None
                total_time = 0.0

                for lr in CFG["lr_grid"]:
                    # Fresh model init
                    torch.manual_seed(CFG["seed"])
                    model = SequenceModel(block_cls, VOCAB_SIZE, d_model, CFG["n_layers"], block_kwargs).to(device)
                    
                    acc, eval_t = train_and_eval(model, num_pairs, seq_len, lr,
                                                 epochs=CFG["epochs"],
                                                 steps_per_epoch=CFG["steps_per_epoch"])
                    total_time += eval_t
                    if acc > best_acc:
                        best_acc = acc
                        best_lr = lr

                key_name = f"{name}_dk{d_k}"
                if key_name not in results_matrix[d_k]:
                    results_matrix[d_k][key_name] = {}

                results_matrix[d_k][key_name][num_pairs] = {
                    "best_acc": round(best_acc, 2),
                    "best_lr": best_lr,
                    "total_eval_time": round(total_time, 2)
                }
                print(f"      [{name:28s} | d_k={d_k}] Pairs={num_pairs:3d} (L={seq_len:4d}) -> Best Acc: {best_acc:6.2f}% (lr={best_lr})")

    # ── Save Results JSON & Master Ledger ───────────────────────────────
    os.makedirs("results/raw", exist_ok=True)
    raw_path = "results/raw/v300_capacity_scaling.json"
    with open(raw_path, "w") as f:
        json.dump({"config": CFG, "results": results_matrix}, f, indent=2)
    print(f"\nsaved: {raw_path}")

    # Append to master ledger
    master_ledger_line = {
        "experiment_id": "v300_capacity_scaling",
        "fecha": "2026-07-25",
        "familia": "fase_compleja",
        "dataset": "MQAR_scaling (pairs: 32-256, seq_len: 256-2048)",
        "n_eval": len(CFG["d_k_list"]) * len(CFG["num_pairs_list"]) * 3 * len(CFG["lr_grid"]),
        "metric_name": "recall_accuracy",
        "value": "scaling_matrix_logged",
        "SE": None,
        "params": "scaled_with_dk",
        "nivel_rigor": 2,
        "etiqueta": "ANCLA"
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/master_ledger.jsonl", "a") as f:
        f.write(json.dumps(master_ledger_line) + "\n")
    print("Logged execution to results/master_ledger.jsonl")
    print("=" * 80)

if __name__ == "__main__":
    run_experiment_suite()
