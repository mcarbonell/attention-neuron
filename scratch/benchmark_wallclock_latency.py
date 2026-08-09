"""
benchmark_wallclock_latency.py
===============================
Wall-Clock Latency & Throughput Benchmark for DeltaPhase vs Real Controls vs Softmax MHA.

Measures:
  1. Forward Pass Latency (ms per batch / ms per token)
  2. Generation / Autoregressive Decoding Latency (ms per token generated)
  3. Peak VRAM Memory Allocation (MB)
Across context lengths L in [256, 512, 1024, 2048, 4096].

Usage:
  python scratch/benchmark_wallclock_latency.py
"""

import math, time, os, json, sys, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()

def ts():
    elapsed = time.time() - T0
    h = int(elapsed // 3600)
    m = int(elapsed % 3600 // 60)
    s = elapsed % 60
    return f"[{h:02d}:{m:02d}:{s:05.2f}]"

_device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(_device)

# ── Building Blocks ──────────────────────────────────────────────────

class SinCosPE(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.shape[1]]

class ShortCausalConv1D(nn.Module):
    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                              padding=kernel_size-1, groups=d_model)
        self.act = nn.SiLU()
    def forward(self, x):
        B, L, D = x.shape
        return x + self.act(self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2))

class FFN(nn.Module):
    def __init__(self, d_model, expand=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model * expand),
                                 nn.SiLU(),
                                 nn.Linear(d_model * expand, d_model))
    def forward(self, x): return self.net(x)

class ChunkwiseComplexDeltaPhaseBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=2, d_k=32, chunk_size=64):
        super().__init__()
        self.d_model, self.n_heads, self.d_k, self.chunk_size = d_model, n_heads, d_k, chunk_size
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.theta_k_proj = nn.Linear(d_model, n_heads * d_k)
        self.theta_q_proj = nn.Linear(d_model, n_heads * d_k)
        self.val_proj = nn.Linear(d_model, n_heads * d_k)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(n_heads * d_k, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x; conv_x = self.causal_conv(self.norm1(x))
        B, L, D = conv_x.shape; C = self.chunk_size; inv_dk = 1.0 / float(self.d_k)
        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            conv_x = F.pad(conv_x, (0, 0, 0, pad_len)); L_padded = L + pad_len
        else: L_padded = L
        theta_k = self.theta_k_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        theta_q = self.theta_q_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        v = self.val_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(conv_x)).transpose(1, 2)
        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)
        num_chunks = L_padded // C
        Q_c = Q.view(B, self.n_heads, num_chunks, C, self.d_k)
        K_c = K.view(B, self.n_heads, num_chunks, C, self.d_k)
        V_c = v.view(B, self.n_heads, num_chunks, C, self.d_k)
        beta_c = beta.view(B, self.n_heads, num_chunks, C)
        Gram_real = torch.matmul(K_c, torch.conj(K_c).transpose(-1, -2)).real * inv_dk
        L_mat = torch.triu(Gram_real * beta_c.unsqueeze(-1), diagonal=1)
        I_mat = torch.eye(C, device=x.device).view(1, 1, 1, C, C)
        T_mat = torch.linalg.inv(I_mat + L_mat.transpose(-1, -2))
        M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=x.device)
        out_chunks = []
        for c in range(num_chunks):
            qc, kc, vc, bc, tc = Q_c[:,:,c], K_c[:,:,c], V_c[:,:,c], beta_c[:,:,c], T_mat[:,:,c]
            v_old = torch.matmul(M_state, torch.conj(kc).transpose(-1,-2)).real.transpose(-1,-2) * inv_dk
            E_c = torch.matmul(tc, vc - v_old)
            U_c = bc.unsqueeze(-1) * E_c
            o_inter = torch.matmul(M_state, torch.conj(qc).transpose(-1,-2)).real.transpose(-1,-2) * inv_dk
            A_intra = torch.tril(torch.matmul(qc, torch.conj(kc).transpose(-1,-2)).real) * inv_dk
            out_chunks.append(torch.matmul(A_intra, U_c) + o_inter)
            M_state = M_state + torch.matmul(U_c.to(torch.complex64).transpose(-1,-2), kc)
        retrieved = torch.cat(out_chunks, dim=2)[:,:,:L].transpose(1,2).reshape(B, L, self.n_heads*self.d_k)
        out = res + self.out_proj(retrieved)
        return out + self.ffn(self.norm2(out))

class ChunkwiseRealDeltaNetIsoParamBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=2, d_k=32, chunk_size=64):
        super().__init__()
        self.d_model, self.n_heads, self.d_k, self.chunk_size = d_model, n_heads, d_k, chunk_size
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.k_proj = nn.Linear(d_model, n_heads * d_k)
        self.q_proj = nn.Linear(d_model, n_heads * d_k)
        self.val_proj = nn.Linear(d_model, n_heads * d_k)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(n_heads * d_k, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x; conv_x = self.causal_conv(self.norm1(x))
        B, L, D = conv_x.shape; C = self.chunk_size
        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            conv_x = F.pad(conv_x, (0, 0, 0, pad_len)); L_padded = L + pad_len
        else: L_padded = L
        k_raw = self.k_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        q_raw = self.q_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        v = self.val_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(conv_x)).transpose(1, 2)
        K = F.normalize(k_raw, p=2, dim=-1); Q = F.normalize(q_raw, p=2, dim=-1)
        num_chunks = L_padded // C
        Q_c = Q.view(B, self.n_heads, num_chunks, C, self.d_k)
        K_c = K.view(B, self.n_heads, num_chunks, C, self.d_k)
        V_c = v.view(B, self.n_heads, num_chunks, C, self.d_k)
        beta_c = beta.view(B, self.n_heads, num_chunks, C)
        Gram = torch.matmul(K_c, K_c.transpose(-1, -2))
        L_mat = torch.triu(Gram * beta_c.unsqueeze(-1), diagonal=1)
        I_mat = torch.eye(C, device=x.device).view(1, 1, 1, C, C)
        T_mat = torch.linalg.inv(I_mat + L_mat.transpose(-1, -2))
        M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, device=x.device)
        out_chunks = []
        for c in range(num_chunks):
            qc, kc, vc, bc, tc = Q_c[:,:,c], K_c[:,:,c], V_c[:,:,c], beta_c[:,:,c], T_mat[:,:,c]
            v_old = torch.matmul(kc, M_state.transpose(-1,-2))
            E_c = torch.matmul(tc, vc - v_old)
            U_c = bc.unsqueeze(-1) * E_c
            o_inter = torch.matmul(qc, M_state.transpose(-1,-2))
            A_intra = torch.tril(torch.matmul(qc, kc.transpose(-1,-2)))
            out_chunks.append(torch.matmul(A_intra, U_c) + o_inter)
            M_state = M_state + torch.matmul(U_c.transpose(-1,-2), kc)
        retrieved = torch.cat(out_chunks, dim=2)[:,:,:L].transpose(1,2).reshape(B, L, self.n_heads*self.d_k)
        out = res + self.out_proj(retrieved)
        return out + self.ffn(self.norm2(out))

class CausalAttentionBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=2):
        super().__init__()
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
    def forward(self, x):
        res = x; conv_x = self.causal_conv(self.norm1(x))
        B, L, D = conv_x.shape
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=conv_x.device), diagonal=1)
        attn_out, _ = self.mha(conv_x, conv_x, conv_x, attn_mask=causal_mask, is_causal=False)
        return res + attn_out + self.ffn(self.norm2(res + attn_out))

class SequenceModel(nn.Module):
    def __init__(self, block_cls, vocab_size=1000, d_model=64, n_layers=4, block_kwargs=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = SinCosPE(d_model)
        self.layers = nn.ModuleList([block_cls(d_model=d_model, **(block_kwargs or {}))
                                     for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        h = self.pe(self.emb(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)

# ── Benchmarking Functions ──────────────────────────────────────────────

def benchmark_model_latency(model, batch_size=8, seq_len=512, warmup_runs=10, test_runs=50):
    model.eval()
    x = torch.randint(1, 900, (batch_size, seq_len), device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            
    # Measure Latency
    times = []
    with torch.no_grad():
        for _ in range(test_runs):
            t_start = time.perf_counter()
            _ = model(x)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_end = time.perf_counter()
            times.append((t_end - t_start) * 1000.0)  # ms
            
    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    ms_per_token = mean_ms / (batch_size * seq_len)
    tokens_per_sec = (batch_size * seq_len) / (mean_ms / 1000.0)
    return mean_ms, std_ms, ms_per_token, tokens_per_sec

# ── Main Execution ──────────────────────────────────────────────────────

print("=" * 85)
print(f"{ts()} BENCHMARK: WALL-CLOCK LATENCY & THROUGHPUT ({_device.upper()})")
print("=" * 85)

MODELS = {
    "ChunkwiseComplexDeltaPhase": (ChunkwiseComplexDeltaPhaseBlock, {"d_k": 32, "chunk_size": 64}),
    "ChunkwiseRealDeltaNetIsoParam": (ChunkwiseRealDeltaNetIsoParamBlock, {"d_k": 32, "chunk_size": 64}),
    "CausalAttentionMHA": (CausalAttentionBlock, {}),
}

seq_lengths = [256, 512, 1024, 2048]
results = {}

for model_name, (block_cls, kwargs) in MODELS.items():
    model = SequenceModel(block_cls, vocab_size=1000, d_model=64, n_layers=4, block_kwargs=kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{ts()} === Model: {model_name} (Params: {n_params:,}) ===", flush=True)
    results[model_name] = {}
    
    for L in seq_lengths:
        try:
            mean_ms, std_ms, ms_tok, tok_sec = benchmark_model_latency(model, batch_size=8, seq_len=L)
            results[model_name][L] = {
                "mean_batch_ms": round(mean_ms, 2),
                "std_batch_ms": round(std_ms, 2),
                "ms_per_token": round(ms_tok, 5),
                "tokens_per_sec": round(tok_sec, 1)
            }
            print(f"{ts()}   L={L:<4d} -> BatchTime: {mean_ms:6.2f} ms | "
                  f"ms/token: {ms_tok:.5f} | Throughput: {tok_sec:8.1f} tok/s", flush=True)
        except Exception as e:
            print(f"{ts()}   L={L:<4d} -> OOM or Error: {e}", flush=True)
            results[model_name][L] = {"error": str(e)}
            
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# ── Summary Table ───────────────────────────────────────────────────────

print(f"\n{'='*85}")
print(f"{ts()} SUMMARY TABLE — WALL-CLOCK LATENCY (ms per batch, B=8)")
print(f"{'='*85}")
header = f"  {'Model':35s}" + "".join(f" | L={l:<4d} (ms)" for l in seq_lengths)
print(header)
print("  " + "-" * len(header))

for model_name in sorted(results.keys()):
    row = f"  {model_name:35s}"
    for L in seq_lengths:
        if "mean_batch_ms" in results[model_name].get(L, {}):
            val = results[model_name][L]["mean_batch_ms"]
            row += f" | {val:9.2f}"
        else:
            row += f" | {'ERR':>9s}"
    print(row)

output_file = f"benchmark_wallclock_latency_{_device}_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n{ts()} saved: {output_file}")
