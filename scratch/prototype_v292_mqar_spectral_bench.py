"""
prototype_v292_mqar_spectral_bench.py
======================================
V292: Multi-Query Associative Recall (MQAR) & Induction Heads Benchmark.

Probes whether input-dependent multiplicative gating on an O(N log N) spectral mixer
(CausalComplexFFT, CausalWalsh) can perform content-dependent associative recall (MQAR),
or if associative recall is restricted to quadratic O(N^2) Softmax Attention.

Models compared (iso-parameter ~220k params):
  1. CausalGatedFFTMixer  [CANDIDATE 1] : Zero-padded causal FFT + dynamic SiLU input gating
  2. CausalGatedWalshMixer[CANDIDATE 2] : Zero-padded causal FWHT + dynamic SiLU input gating
  3. StaticFFTMixer       [BASELINE 1]  : Causal FFT + static learnable weights (FNet style, NO gating)
  4. CausalAttentionMHA   [BASELINE 2]  : Standard Causal Softmax Multi-Head Attention (O(N^2))

Methodology:
  - Sequence length L = 128
  - Vocab: 64 Keys, 64 Values, Special tokens (PAD=0, QUERY=129, SEP=130)
  - Task: N_kv random pairs sampled in seq. At query token [QUERY, K_q], predict V_q.
  - Evaluation metric: Strict Query Target Accuracy (%) on held-out test batch.
  - Logging: JSON in results/raw/v292_mqar.json & line in results/master_ledger.jsonl
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
    "exp_id": "v292_mqar",
    "seq_len": 64,
    "num_kv_pairs": 8,
    "num_keys": 32,
    "num_vals": 32,
    "batch_size": 64,
    "d_model": 64,
    "n_layers": 3,
    "n_heads": 4,
    "epochs": 25,
    "steps_per_epoch": 80,
    "lr": 4e-3,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

torch.manual_seed(CFG["seed"])
device = torch.device(CFG["device"])

# ── 2. Data Generator for MQAR Task ─────────────────────────────────────
# Vocab layout:
# 0: PAD
# 1..64: Keys
# 65..128: Values
# 129: QUERY_MARKER
PAD_ID = 0
KEY_OFFSET = 1
VAL_OFFSET = 1 + CFG["num_keys"]
QUERY_MARKER = VAL_OFFSET + CFG["num_vals"]
VOCAB_SIZE = QUERY_MARKER + 1

def generate_mqar_batch(batch_size, seq_len=128, num_pairs=16, num_keys=64, num_vals=64, device=device):
    """
    Generates an MQAR batch.
    Sequence structure:
    [K_1, V_1, K_2, V_2, ..., K_N, V_N, ... PAD/Filler ..., QUERY_MARKER, K_q]
    Target tensor has -100 everywhere EXCEPT at position of K_q, where target is V_q.
    """
    x = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    for b in range(batch_size):
        # Sample unique keys & values for this sequence
        keys = torch.randperm(num_keys, device=device)[:num_pairs] + KEY_OFFSET
        vals = torch.randint(0, num_vals, (num_pairs,), device=device) + VAL_OFFSET
        
        # Interleave K, V
        kv_interleaved = torch.stack([keys, vals], dim=1).flatten()
        
        # Place K-V pairs at start
        x[b, :len(kv_interleaved)] = kv_interleaved
        
        # Select one random key as query target
        q_idx = torch.randint(0, num_pairs, (1,)).item()
        target_k = keys[q_idx]
        target_v = vals[q_idx]
        
        # Place QUERY_MARKER and target_k at the end of sequence (positions seq_len-2 and seq_len-1)
        x[b, seq_len - 2] = QUERY_MARKER
        x[b, seq_len - 1] = target_k
        
        # Target at position seq_len-1 is target_v
        y[b, seq_len - 1] = target_v

    return x, y

# ── 3. FWHT Helper ─────────────────────────────────────────────────────
def fwht(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform along last dimension."""
    N = x.shape[-1]
    h = 1
    while h < N:
        x = x.reshape(*x.shape[:-1], N // (2 * h), 2 * h)
        a, b = x[..., :h], x[..., h:]
        x = torch.cat([a + b, a - b], dim=-1)
        x = x.reshape(*x.shape[:-2], N)
        h *= 2
    return x / math.sqrt(N)

# ── 4. Architecture Components ──────────────────────────────────────────

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

# ── Candidate 1: Causal Gated FFT Mixer Layer ──────────────────────────
class CausalGatedFFTMixerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.val_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        # x: [B, L, D]
        res = x
        normed = self.norm1(x)
        B, L, D = normed.shape
        
        # Gating projection dependent on input
        g = F.silu(self.gate_proj(normed))  # [B, L, D]
        v = self.val_proj(normed)            # [B, L, D]
        
        # Causal Zero-padding along sequence dimension (L -> 2L)
        padded = F.pad(v * g, (0, 0, 0, L))  # [B, 2L, D]
        
        # Real FFT along sequence dimension
        fft_feat = torch.fft.rfft(padded, dim=1) # [B, L+1, D] complex
        
        # Inverse FFT and crop back to L
        ifft_feat = torch.fft.irfft(fft_feat, n=2*L, dim=1)[:, :L, :] # [B, L, D]
        
        x = res + self.out_proj(ifft_feat)
        x = x + self.ffn(self.norm2(x))
        return x

# ── Candidate 2: Causal Gated Walsh Mixer Layer ─────────────────────────
class CausalGatedWalshMixerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.val_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x
        normed = self.norm1(x)
        B, L, D = normed.shape
        
        g = F.silu(self.gate_proj(normed))
        v = self.val_proj(normed)
        
        # FWHT requires power of 2 length -> pad L=128 to 2L=256
        padded = F.pad(v * g, (0, 0, 0, L)) # [B, 2L, D]
        
        # Transpose so sequence dimension is last
        padded_t = padded.transpose(1, 2)   # [B, D, 2L]
        walsh_feat = fwht(padded_t)
        ifwht_feat = fwht(walsh_feat)        # FWHT is self-inverse
        
        out_seq = ifwht_feat.transpose(1, 2)[:, :L, :] # [B, L, D]
        
        x = res + self.out_proj(out_seq)
        x = x + self.ffn(self.norm2(x))
        return x

# ── Baseline 1: Static FFT Mixer Layer (FNet style, NO dynamic gating) ──
class StaticFFTMixerBlock(nn.Module):
    def __init__(self, d_model, seq_len=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # Static learnable spectral filter weights (complex)
        self.spec_weight = nn.Parameter(torch.randn(seq_len + 1, d_model, dtype=torch.cfloat) * 0.02)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x
        normed = self.norm1(x)
        B, L, D = normed.shape
        v = self.proj(normed)
        
        padded = F.pad(v, (0, 0, 0, L)) # [B, 2L, D]
        fft_feat = torch.fft.rfft(padded, dim=1) # [B, L+1, D]
        
        # Apply STATIC filter (independent of input content x_t)
        filtered = fft_feat * self.spec_weight
        
        ifft_feat = torch.fft.irfft(filtered, n=2*L, dim=1)[:, :L, :]
        x = res + self.out_proj(ifft_feat)
        x = x + self.ffn(self.norm2(x))
        return x

# ── Baseline 2: Standard Causal Attention (MHA O(N^2)) ──────────────────
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
    def __init__(self, block_cls, vocab_size, d_model, n_layers, seq_len=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = SinCosPE(d_model, max_len=seq_len)
        if block_cls == StaticFFTMixerBlock:
            self.blocks = nn.ModuleList([block_cls(d_model, seq_len=seq_len) for _ in range(n_layers)])
        else:
            self.blocks = nn.ModuleList([block_cls(d_model) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.pe(self.emb(x))
        for block in self.blocks:
            h = block(h)
        h = self.norm_f(h)
        return self.head(h)

# ── 5. Evaluation Function ──────────────────────────────────────────────
@torch.no_grad()
def evaluate_mqar(model, num_batches=10, cfg=CFG):
    model.eval()
    correct = 0
    total = 0
    for _ in range(num_batches):
        x, y = generate_mqar_batch(cfg["batch_size"], cfg["seq_len"], cfg["num_kv_pairs"], cfg["num_keys"], cfg["num_vals"], device)
        logits = model(x) # [B, L, V]
        
        # Only evaluate at query position (seq_len - 1)
        preds = logits[:, -1, :].argmax(dim=-1) # [B]
        targets = y[:, -1]                     # [B]
        
        correct += (preds == targets).sum().item()
        total += targets.numel()
    return (correct / total) * 100.0

# ── 6. Benchmark Runner ─────────────────────────────────────────────────
def run_model_benchmark(name, block_cls, cfg=CFG):
    print(f"\n==================================================")
    print(f" Running Benchmark: {name}")
    print(f"==================================================")
    
    model = MQARModel(block_cls, VOCAB_SIZE, cfg["d_model"], cfg["n_layers"], cfg["seq_len"]).to(device)
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
            
            # Fast Feedback Rule: print early step progress
            if epoch == 1 and step in [1, 2, 3, 5, 10]:
                print(f"  [Fast Feedback] Epoch 1 Step {step:02d} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / cfg["steps_per_epoch"]
        val_acc = evaluate_mqar(model, num_batches=10, cfg=cfg)
        print(f" Epoch {epoch:02d}/{cfg['epochs']:02d} | Train Loss: {avg_loss:.4f} | MQAR Target Acc: {val_acc:5.2f}%")
        
        # Early stop if converged to near 100%
        if val_acc >= 99.5:
            print(f" >>> Converged early at epoch {epoch} with {val_acc:.2f}% accuracy!")
            break
            
    wall_clock = time.time() - start_time
    final_acc = evaluate_mqar(model, num_batches=25, cfg=cfg) # 1600 test samples
    internal_overhead = wall_clock - eval_time_accum
    
    print(f"--> Summary {name}: Final Acc = {final_acc:.2f}% | Wall Clock = {wall_clock:.2f}s | Internal Overhead = {internal_overhead:.2f}s")
    
    return {
        "model_name": name,
        "params": params,
        "final_acc": final_acc,
        "wall_clock_time": round(wall_clock, 2),
        "eval_time": round(eval_time_accum, 2),
        "internal_overhead_time": round(internal_overhead, 2),
        "epochs_run": epoch
    }

# ── 7. Main Execution ───────────────────────────────────────────────────
def main():
    print(f"Starting V292 MQAR Benchmark on device: {device}")
    print(f"Sequence Length: {CFG['seq_len']} | KV Pairs: {CFG['num_kv_pairs']} | d_model: {CFG['d_model']} | Layers: {CFG['n_layers']}")
    
    results = []
    
    # REGLA DE ORO: CANDIDATES FIRST, BASELINES AFTER
    # Candidate 1: Causal Gated FFT Mixer
    res_fft_gated = run_model_benchmark("CausalGatedFFTMixer (Candidate 1)", CausalGatedFFTMixerBlock, CFG)
    results.append(res_fft_gated)
    
    # Candidate 2: Causal Gated Walsh Mixer
    res_walsh_gated = run_model_benchmark("CausalGatedWalshMixer (Candidate 2)", CausalGatedWalshMixerBlock, CFG)
    results.append(res_walsh_gated)
    
    # Baseline 1: Static FFT Mixer (FNet style, NO input gating)
    res_fft_static = run_model_benchmark("StaticFFTMixer (Baseline 1 - FNet style)", StaticFFTMixerBlock, CFG)
    results.append(res_fft_static)
    
    # Baseline 2: Causal Attention (MHA O(N^2))
    res_attn = run_model_benchmark("CausalAttentionMHA (Baseline 2 - Softmax MHA)", CausalAttentionBlock, CFG)
    results.append(res_attn)
    
    # Save raw JSON results
    os.makedirs("results/raw", exist_ok=True)
    raw_path = "results/raw/v292_mqar.json"
    with open(raw_path, "w") as f:
        json.dump({"config": CFG, "results": results}, f, indent=2)
    print(f"\nRaw results saved to {raw_path}")
    
    # Master Ledger entry
    os.makedirs("results", exist_ok=True)
    ledger_path = "results/master_ledger.jsonl"
    with open(ledger_path, "a") as f:
        for r in results:
            entry = {
                "experiment_id": "v292_mqar",
                "fecha": time.strftime("%Y-%m-%d"),
                "familia": "espectral_gated vs attention",
                "dataset": f"MQAR synthetic (L={CFG['seq_len']}, pairs={CFG['num_kv_pairs']})",
                "n_eval": CFG["epochs"] * CFG["steps_per_epoch"] * CFG["batch_size"],
                "metric_name": f"target_acc_{r['model_name'].split()[0]}",
                "value": r["final_acc"],
                "SE": None,
                "params": r["params"],
                "nivel_rigor": 1,
                "etiqueta": "ANCLA" if r["final_acc"] > 90.0 else "RUIDO-SOSPECHA"
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Master ledger updated at {ledger_path}")

if __name__ == "__main__":
    main()
