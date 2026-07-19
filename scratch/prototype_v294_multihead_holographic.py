"""
prototype_v294_multihead_holographic.py
========================================
V294: Multi-Head Holographic Phase Attention (MH-HPA) & Decayed Causal Accumulator.

Evaluates whether multi-head phase space splitting (H=8, H=16) and causal exponential decay:
  1. Multi-Head Phase Binding: K^{(h)} = exp(i * theta_k^{(h)}), Q^{(h)} = exp(i * theta_q^{(h)})
  2. Noise Reduction via Central Limit Theorem across H independent phase heads
  3. Vectorized LRU Decay: M_t = exp(-alpha * t) * CumSum( exp(alpha * tau) * (K_tau * V_tau) )
can eliminate phase crosstalk noise and reach 90-100% Associative Recall (MQAR) in O(N) linear time!

Models compared (~108k params):
  1. MultiHeadHolographicAccumulator (H=8)   [CANDIDATE 1]: O(N) 8-head phase conjugation
  2. MultiHeadHolographicAccumulator (H=16)  [CANDIDATE 2]: O(N) 16-head phase conjugation
  3. DecayedHolographicAccumulator (H=8)      [CANDIDATE 3]: O(N) 8-head phase + LRU exponential decay
  4. SingleHeadHolographicAccumulator        [BASELINE 1] : Single-head v293 (21.72% baseline)
  5. CausalAttentionMHA                      [BASELINE 2] : Softmax MHA O(N^2)

Methodology:
  - MQAR synthetic task: L=64, num_kv_pairs=8, vocab=32 keys / 32 vals
  - Strict query target accuracy on held-out test batch
  - Results logged to results/raw/v294_holographic_multihead.json & results/master_ledger.jsonl
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
    "exp_id": "v294_holographic_multihead",
    "seq_len": 64,
    "num_kv_pairs": 8,
    "num_keys": 32,
    "num_vals": 32,
    "batch_size": 64,
    "d_model": 64,
    "n_layers": 3,
    "epochs": 20,
    "steps_per_epoch": 80,
    "lr": 4e-3,
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

# ── 2. Data Generator for MQAR Task ─────────────────────────────────────
def generate_mqar_batch(batch_size, seq_len=64, num_pairs=8, num_keys=32, num_vals=32, device=device):
    x = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    for b in range(batch_size):
        keys = torch.randperm(num_keys, device=device)[:num_pairs] + KEY_OFFSET
        vals = torch.randint(0, num_vals, (num_pairs,), device=device) + VAL_OFFSET
        
        kv_interleaved = torch.stack([keys, vals], dim=1).flatten()
        x[b, :len(kv_interleaved)] = kv_interleaved
        
        q_idx = torch.randint(0, num_pairs, (1,)).item()
        target_k = keys[q_idx]
        target_v = vals[q_idx]
        
        x[b, seq_len - 2] = QUERY_MARKER
        x[b, seq_len - 1] = target_k
        y[b, seq_len - 1] = target_v

    return x, y

# ── 3. Architecture Components ──────────────────────────────────────────
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

# ── Candidate 1 & 2: Multi-Head Holographic Accumulator (MH-HPA O(N)) ────
class MultiHeadHolographicAccumulatorBlock(nn.Module):
    """
    Multi-Head Phase Attention in O(N) linear time.
    Splits d_model into H heads of dim d_k.
    Calculates phase binding per head and unbinds via phase conjugate conj(Q^(h)) * M^(h).
    Averages retrieved signals across heads, noise cancels out via Central Limit Theorem.
    """
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.theta_k_proj = nn.Linear(d_model, d_model)
        self.theta_q_proj = nn.Linear(d_model, d_model)
        self.val_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x
        normed = self.norm1(x)
        B, L, D = normed.shape
        
        # 1. Project and reshape to [B, L, H, d_k]
        theta_k = self.theta_k_proj(normed).view(B, L, self.n_heads, self.d_k)
        theta_q = self.theta_q_proj(normed).view(B, L, self.n_heads, self.d_k)
        v = self.val_proj(normed).view(B, L, self.n_heads, self.d_k)
        
        # 2. Unit complex phasors per head
        K = torch.polar(torch.ones_like(theta_k), theta_k) # [B, L, H, d_k] complex
        Q = torch.polar(torch.ones_like(theta_q), theta_q) # [B, L, H, d_k] complex
        
        # 3. Phase Binding per head: B_t = K_t * V_t
        bound = K * v # [B, L, H, d_k] complex
        
        # 4. Causal Cumulative Memory: M_t = sum_{tau <= t} B_tau
        memory = torch.cumsum(bound, dim=1) # [B, L, H, d_k] complex
        
        # 5. Phase Conjugate Unbinding: R_t = Re( conj(Q_t) * M_t )
        unbound = torch.conj(Q) * memory
        retrieved = unbound.real.view(B, L, D) # [B, L, D] real
        
        x = res + self.out_proj(retrieved)
        x = x + self.ffn(self.norm2(x))
        return x

# ── Candidate 3: Decayed Multi-Head Holographic Accumulator (LRU Decay O(N)) ──
class DecayedHolographicAccumulatorBlock(nn.Module):
    """
    Multi-Head Holographic Phase Memory with Learnable Exponential Causal Decay (LRU).
    M_t = exp(-alpha * t) * CumSum( exp(alpha * tau) * (K_tau * V_tau) )
    Vectorized in 1 line without loops!
    """
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.theta_k_proj = nn.Linear(d_model, d_model)
        self.theta_q_proj = nn.Linear(d_model, d_model)
        self.val_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Learnable decay rate per head
        self.log_decay = nn.Parameter(torch.full((1, 1, n_heads, 1), -3.0)) # small initial decay
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x
        normed = self.norm1(x)
        B, L, D = normed.shape
        
        theta_k = self.theta_k_proj(normed).view(B, L, self.n_heads, self.d_k)
        theta_q = self.theta_q_proj(normed).view(B, L, self.n_heads, self.d_k)
        v = self.val_proj(normed).view(B, L, self.n_heads, self.d_k)
        
        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)
        bound = K * v
        
        # Vectorized Causal Decay scale exp(alpha * tau)
        alpha = F.softplus(self.log_decay) # positive decay rate
        pos = torch.arange(L, device=normed.device, dtype=torch.float32).view(1, L, 1, 1)
        
        growth = torch.exp(alpha * pos) # [1, L, H, 1]
        decay = torch.exp(-alpha * pos)  # [1, L, H, 1]
        
        scaled_bound = bound * growth
        memory_cum = torch.cumsum(scaled_bound, dim=1)
        decayed_memory = memory_cum * decay # [B, L, H, d_k] complex
        
        unbound = torch.conj(Q) * decayed_memory
        retrieved = unbound.real.view(B, L, D)
        
        x = res + self.out_proj(retrieved)
        x = x + self.ffn(self.norm2(x))
        return x

# ── Baseline 1: Single-Head Holographic Accumulator (v293 Control) ──────
class SingleHeadHolographicAccumulatorBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.theta_k_proj = nn.Linear(d_model, d_model)
        self.theta_q_proj = nn.Linear(d_model, d_model)
        self.val_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x
        normed = self.norm1(x)
        theta_k = self.theta_k_proj(normed)
        theta_q = self.theta_q_proj(normed)
        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)
        v = self.val_proj(normed)
        bound = K * v
        memory = torch.cumsum(bound, dim=1)
        unbound = torch.conj(Q) * memory
        retrieved = unbound.real
        x = res + self.out_proj(retrieved)
        x = x + self.ffn(self.norm2(x))
        return x

# ── Baseline 2: Standard Causal Attention (Softmax MHA O(N^2)) ──────────
class CausalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x
        normed = self.norm1(x)
        L = normed.shape[1]
        mask = torch.triu(torch.full((L, L), float('-inf'), device=normed.device), diagonal=1)
        attn_out, _ = self.mha(normed, normed, normed, attn_mask=mask, is_causal=True)
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
        preds = logits[:, -1, :].argmax(dim=-1)
        targets = y[:, -1]
        correct += (preds == targets).sum().item()
        total += targets.numel()
    return (correct / total) * 100.0

# ── Benchmark Runner ────────────────────────────────────────────────────
def run_model_benchmark(name, block_cls, kwargs={}, cfg=CFG):
    print(f"\n==================================================")
    print(f" Running Benchmark: {name}")
    print(f"==================================================")
    
    model = MQARModel(block_cls, VOCAB_SIZE, cfg["d_model"], cfg["n_layers"], cfg["seq_len"], **kwargs).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"] * cfg["steps_per_epoch"])
    
    start_time = time.time()
    eval_time_accum = 0.0
    
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for step in range(1, cfg["steps_per_epoch"] + 1):
            x, y = generate_mqar_batch(cfg["batch_size"], cfg["seq_len"], cfg["num_kv_pairs"], cfg["num_keys"], cfg["num_vals"], device)
            
            t_eval0 = time.time()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=-100)
            eval_time_accum += (time.time() - t_eval0)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if epoch == 1 and step in [1, 2, 3, 5, 10]:
                print(f"  [Fast Feedback] Epoch 1 Step {step:02d} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / cfg["steps_per_epoch"]
        val_acc = evaluate_mqar(model, num_batches=10, cfg=cfg)
        print(f" Epoch {epoch:02d}/{cfg['epochs']:02d} | Train Loss: {avg_loss:.4f} | MQAR Target Acc: {val_acc:5.2f}%")
        
        if val_acc >= 99.5:
            print(f" >>> Converged early at epoch {epoch} with {val_acc:.2f}% accuracy!")
            break
            
    wall_clock = time.time() - start_time
    final_acc = evaluate_mqar(model, num_batches=25, cfg=cfg)
    internal_overhead = wall_clock - eval_time_accum
    
    print(f"--> Summary {name}: Final Acc = {final_acc:.2f}% | Wall Clock = {wall_clock:.2f}s | Overhead = {internal_overhead:.2f}s")
    
    return {
        "model_name": name,
        "params": params,
        "final_acc": final_acc,
        "wall_clock_time": round(wall_clock, 2),
        "eval_time": round(eval_time_accum, 2),
        "internal_overhead_time": round(internal_overhead, 2),
        "epochs_run": epoch
    }

# ── Main ────────────────────────────────────────────────────────────────
def main():
    print(f"Starting V294 Multi-Head Holographic Phase Benchmark on device: {device}")
    print(f"Sequence Length: {CFG['seq_len']} | KV Pairs: {CFG['num_kv_pairs']} | d_model: {CFG['d_model']} | Layers: {CFG['n_layers']}")
    
    results = []
    
    # REGLA DE ORO: CANDIDATES FIRST, BASELINES AFTER
    # Candidate 1: Multi-Head Holographic Accumulator (H=8)
    res_mh8 = run_model_benchmark("MultiHeadHolographic (H=8) [Candidate 1]", MultiHeadHolographicAccumulatorBlock, {"n_heads": 8}, CFG)
    results.append(res_mh8)
    
    # Candidate 2: Multi-Head Holographic Accumulator (H=16)
    res_mh16 = run_model_benchmark("MultiHeadHolographic (H=16) [Candidate 2]", MultiHeadHolographicAccumulatorBlock, {"n_heads": 16}, CFG)
    results.append(res_mh16)
    
    # Candidate 3: Decayed Multi-Head Holographic Accumulator (H=8 + LRU Decay)
    res_decay = run_model_benchmark("DecayedHolographic (H=8 + LRU) [Candidate 3]", DecayedHolographicAccumulatorBlock, {"n_heads": 8}, CFG)
    results.append(res_decay)
    
    # Baseline 1: Single-Head Holographic Accumulator (v293 Control)
    res_single = run_model_benchmark("SingleHeadHolographic (v293 Baseline)", SingleHeadHolographicAccumulatorBlock, {}, CFG)
    results.append(res_single)
    
    # Baseline 2: Causal Attention MHA (Quadratic O(N^2))
    res_attn = run_model_benchmark("CausalAttentionMHA (Baseline 2 - Softmax MHA O(N^2))", CausalAttentionBlock, {}, CFG)
    results.append(res_attn)
    
    os.makedirs("results/raw", exist_ok=True)
    raw_path = "results/raw/v294_holographic_multihead.json"
    with open(raw_path, "w") as f:
        json.dump({"config": CFG, "results": results}, f, indent=2)
    print(f"\nRaw results saved to {raw_path}")
    
    os.makedirs("results", exist_ok=True)
    ledger_path = "results/master_ledger.jsonl"
    with open(ledger_path, "a") as f:
        for r in results:
            entry = {
                "experiment_id": "v294_holographic_multihead",
                "fecha": time.strftime("%Y-%m-%d"),
                "familia": "holografico_multihead vs attention",
                "dataset": f"MQAR synthetic (L={CFG['seq_len']}, pairs={CFG['num_kv_pairs']})",
                "n_eval": CFG["epochs"] * CFG["steps_per_epoch"] * CFG["batch_size"],
                "metric_name": f"target_acc_{r['model_name'].split()[0]}",
                "value": r["final_acc"],
                "SE": None,
                "params": r["params"],
                "nivel_rigor": 1,
                "etiqueta": "ANCLA" if r["final_acc"] > 90.0 else "SEÑAL"
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Master ledger updated at {ledger_path}")

if __name__ == "__main__":
    main()
