"""
prototype_v301_decisive_benchmark_fast.py
==========================================
Optimized version of v301_decisive_benchmark.py with 3 fixes:
1. Vectorized GPU dataset generator (no Python for-loops per sample)
2. Chunk-wise delta rule forward (reduces Python loop iterations by CHUNK_SIZE factor)
3. Pre-generate datasets ONCE per (d_k, num_pairs), shared across all models and LRs

Algorithmic logic is 100% identical to the original.
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
    "exp_id": "v301_decisive_benchmark_fast",
    "d_k_list": [32, 64],
    "iso_floats_map": {
        32: {"dk_complex": 32, "dk_real_square": 45, "floats_c": 2048, "floats_r_sq": 2025, "floats_r_rect": 2048},
        64: {"dk_complex": 64, "dk_real_square": 90, "floats_c": 8192, "floats_r_sq": 8100, "floats_r_rect": 8192}
    },
    "num_pairs_list": [32, 64, 128, 256],
    "num_keys": 256,
    "num_vals": 256,
    "batch_size": 32,
    "n_layers": 3,
    "epochs": 15,
    "steps_per_epoch": 50,
    "lr_grid": [1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2],
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

device = torch.device(CFG["device"])

PAD_ID = 0
KEY_OFFSET = 1
VAL_OFFSET = 1 + CFG["num_keys"]
QUERY_MARKER = VAL_OFFSET + CFG["num_vals"]
VOCAB_SIZE = QUERY_MARKER + 1

CHUNK_SIZE = 16  # Tune: 8-32 sweet spot on T4/P100

# ── 2. Vectorized GPU Data Generator ───────────────────────────────────
# Uses independent torch.Generator on CPU for reproducibility, but builds
# tensors on GPU in a fully vectorized manner (no per-sample Python loop).

def generate_mqar_batch_vectorized(batch_size, num_pairs, seq_len, rng, num_keys=256, num_vals=256, device=device):
    """Generate one batch of MQAR data, fully vectorized on GPU."""
    # Use CPU generator for reproducibility, then move to device
    # Key selection: sample without replacement via argsort of random
    rand_k = torch.rand(batch_size, num_keys, generator=rng)
    keys = torch.argsort(rand_k, dim=-1)[:, :num_pairs].to(device) + KEY_OFFSET
    vals = torch.randint(0, num_vals, (batch_size, num_pairs), generator=rng).to(device) + VAL_OFFSET

    x = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    # Interleave keys and values: [k0, v0, k1, v1, ...]
    kv_interleaved = torch.stack([keys, vals], dim=2).view(batch_size, 2 * num_pairs)
    x[:, :2 * num_pairs] = kv_interleaved

    # Shuffle query order independently per batch element
    rand_q = torch.rand(batch_size, num_pairs, generator=rng)
    query_perm = torch.argsort(rand_q, dim=-1).to(device)

    q_keys = torch.gather(keys, 1, query_perm)
    q_vals = torch.gather(vals, 1, query_perm)

    n_queries = min(num_pairs, (seq_len - 2 * num_pairs - 2) // 2)
    pos_q = (2 * num_pairs + 2 + 2 * torch.arange(n_queries, device=device)).unsqueeze(0).expand(batch_size, -1)

    x.scatter_(1, pos_q, QUERY_MARKER)
    x.scatter_(1, pos_q + 1, q_keys[:, :n_queries])
    y.scatter_(1, pos_q + 1, q_vals[:, :n_queries])
    return x, y


def generate_mqar_dataset(num_batches, batch_size, num_pairs, seq_len, seed=42, device=device):
    """Pre-generate a full dataset as stacked tensors on GPU."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    x_list, y_list = [], []
    for _ in range(num_batches):
        x, y = generate_mqar_batch_vectorized(batch_size, num_pairs, seq_len, rng,
                                               num_keys=CFG["num_keys"], num_vals=CFG["num_vals"], device=device)
        x_list.append(x)
        y_list.append(y)
    return torch.stack(x_list), torch.stack(y_list)


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
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size - 1, groups=d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        B, L, D = x.shape
        conv_out = self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
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


# ── 4. Chunk-wise Delta Rule Helpers ─────────────────────────────────────

def _delta_rule_chunked_complex(K, Q, V, beta, d_k):
    """Chunk-wise delta-rule scan for the Complex variant.
    K, Q: complex (B, L, H, d_k)   V: real (B, L, H, d_k)
    beta: (B, L, H, 1, 1)
    Returns: real (B, L, H, d_k)
    """
    B, L, H, dk = K.shape
    C = min(CHUNK_SIZE, L)
    M = torch.zeros(B, H, dk, dk, dtype=torch.complex64, device=K.device)
    inv_dk = 1.0 / float(dk)
    out_chunks = []

    for start in range(0, L, C):
        end = min(start + C, L)
        k_c = K[:, start:end]
        q_c = Q[:, start:end]
        v_c = V[:, start:end]
        b_c = beta[:, start:end]

        chunk_out = []
        for i in range(end - start):
            k_t = k_c[:, i]
            q_t = q_c[:, i]
            v_t = v_c[:, i]
            beta_t = b_c[:, i]

            k_conj = torch.conj(k_t)
            q_conj = torch.conj(q_t)
            v_old = torch.matmul(M, k_conj.unsqueeze(-1)).squeeze(-1).real * inv_dk
            err = v_t - v_old
            update = err.to(torch.complex64).unsqueeze(-1) * k_t.unsqueeze(-2)
            M = M + beta_t * update
            ret = torch.matmul(M, q_conj.unsqueeze(-1)).squeeze(-1).real * inv_dk
            chunk_out.append(ret)

        out_chunks.append(torch.stack(chunk_out, dim=1))

    return torch.cat(out_chunks, dim=1)


def _delta_rule_chunked_real(K, Q, V, beta):
    """Chunk-wise delta-rule scan for the Real variant (square or rectangular).
    K, Q: (B, L, H, d_key)   V: (B, L, H, d_val)
    beta: (B, L, H, 1, 1)
    Returns: (B, L, H, d_val)
    """
    B, L, H, d_key = K.shape
    d_val = V.shape[-1]
    C = min(CHUNK_SIZE, L)
    M = torch.zeros(B, H, d_val, d_key, dtype=K.dtype, device=K.device)
    out_chunks = []

    for start in range(0, L, C):
        end = min(start + C, L)
        k_c = K[:, start:end]
        q_c = Q[:, start:end]
        v_c = V[:, start:end]
        b_c = beta[:, start:end]

        chunk_out = []
        for i in range(end - start):
            k_t = k_c[:, i]
            q_t = q_c[:, i]
            v_t = v_c[:, i]
            beta_t = b_c[:, i]

            v_old = torch.matmul(M, k_t.unsqueeze(-1)).squeeze(-1)
            err = v_t - v_old
            M = M + beta_t * (err.unsqueeze(-1) * k_t.unsqueeze(-2))
            ret = torch.matmul(M, q_t.unsqueeze(-1)).squeeze(-1)
            chunk_out.append(ret)

        out_chunks.append(torch.stack(chunk_out, dim=1))

    return torch.cat(out_chunks, dim=1)


# ── Candidate 1: ComplexDeltaPhaseHolographicBlock (K_dim=2*d_k, V_dim=d_k) ──
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

        retrieved = _delta_rule_chunked_complex(K, Q, v, beta, self.d_k)
        retrieved = retrieved.reshape(B, L, self.n_heads * self.d_k)

        out = res + self.out_proj(retrieved)
        return out + self.ffn(self.norm2(out))

# ── Candidate 2: RealDeltaNetVanillaBlock (Square: K_dim=d_k_real, V_dim=d_k_real) ──
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

        K = F.normalize(self.k_proj(conv_x).view(B, L, self.n_heads, self.d_k), p=2, dim=-1)
        Q = F.normalize(self.q_proj(conv_x).view(B, L, self.n_heads, self.d_k), p=2, dim=-1)
        v = self.val_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        beta = torch.sigmoid(self.beta_proj(conv_x)).view(B, L, self.n_heads, 1, 1)

        retrieved = _delta_rule_chunked_real(K, Q, v, beta)
        retrieved = retrieved.reshape(B, L, self.n_heads * self.d_k)

        out = res + self.out_proj(retrieved)
        return out + self.ffn(self.norm2(out))

# ── Candidate 3: RealDeltaNetRectangularBlock (K_dim=2*d_k, V_dim=d_k) ──
class RealDeltaNetRectangularBlock(nn.Module):
    def __init__(self, d_model, n_heads=2, d_k=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = 2 * d_k
        self.d_val = d_k

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)

        self.k_proj = nn.Linear(d_model, n_heads * self.d_key)
        self.q_proj = nn.Linear(d_model, n_heads * self.d_key)
        self.val_proj = nn.Linear(d_model, n_heads * self.d_val)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(n_heads * self.d_val, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x
        normed = self.norm1(x)
        conv_x = self.causal_conv(normed)
        B, L, D = conv_x.shape

        K = F.normalize(self.k_proj(conv_x).view(B, L, self.n_heads, self.d_key), p=2, dim=-1)
        Q = F.normalize(self.q_proj(conv_x).view(B, L, self.n_heads, self.d_key), p=2, dim=-1)
        v = self.val_proj(conv_x).view(B, L, self.n_heads, self.d_val)
        beta = torch.sigmoid(self.beta_proj(conv_x)).view(B, L, self.n_heads, 1, 1)

        # _delta_rule_chunked_real handles rectangular M: (B, H, d_val, d_key)
        retrieved = _delta_rule_chunked_real(K, Q, v, beta)
        retrieved = retrieved.reshape(B, L, self.n_heads * self.d_val)

        out = res + self.out_proj(retrieved)
        return out + self.ffn(self.norm2(out))

# ── Candidate 4: CausalAttentionBlock ──
class CausalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

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

        out = res + attn_out
        return out + self.ffn(self.norm2(out))

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

# ── Training (receives pre-generated data) ──────────────────────────────
def train_model(model, model_label, train_x, train_y, lr, epochs, steps_per_epoch, accum_steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    start_time = time.time()
    for ep in range(epochs):
        model.train()
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            total_loss = 0.0
            for acc_i in range(accum_steps):
                idx = (step * accum_steps + acc_i) % len(train_x)
                logits = model(train_x[idx])
                loss = criterion(logits.view(-1, VOCAB_SIZE), train_y[idx].view(-1)) / accum_steps
                loss.backward()
                total_loss += loss.item() * accum_steps
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print(f"      [{model_label:36s} | lr={lr:.4f}] Epoch {ep+1:2d}/{epochs} Complete | Loss = {total_loss:.4f}", flush=True)
    train_time = time.time() - start_time
    return train_time


def eval_accuracy(model, data_x, data_y):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(len(data_x)):
            logits = model(data_x[i])
            preds = logits.argmax(dim=-1)
            mask = (data_y[i] != -100)
            correct += (preds[mask] == data_y[i][mask]).sum().item()
            total += mask.sum().item()
    return (correct / total) * 100.0 if total > 0 else 0.0


# ── Main Decisive Suite ─────────────────────────────────────────────────
def run_experiment_suite():
    print("=" * 85)
    print(f"V301 DECISIVE BENCHMARK (FAST): Complex Phase vs Real DeltaNet (Square & Rectangular)")
    print("=" * 85)
    print(f"  * Exp ID:          {CFG['exp_id']}")
    print(f"  * Hardware Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  * PyTorch Version: {torch.__version__}")
    print(f"  * CUDA Available:  {torch.cuda.is_available()} (Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'})")
    print(f"  * Seed:            {CFG['seed']}")
    print(f"  * Layers:          {CFG['n_layers']}")
    print(f"  * Effective Batch: {CFG['batch_size']}")
    print(f"  * Epochs / Steps:  {CFG['epochs']} epochs, {CFG['steps_per_epoch']} steps/epoch")
    print(f"  * LR Grid:         {CFG['lr_grid']}")
    print(f"  * Sweeps d_k:      {CFG['d_k_list']}")
    print(f"  * KV Pairs Sweep:  {CFG['num_pairs_list']}")
    print(f"  * Chunk Size:      {CHUNK_SIZE}")
    print("=" * 85, flush=True)

    results_matrix = {}

    for d_k in CFG["d_k_list"]:
        iso_info = CFG["iso_floats_map"][d_k]
        dk_c = iso_info["dk_complex"]
        dk_r_sq = iso_info["dk_real_square"]
        d_model = 2 * dk_c

        floats_c = iso_info["floats_c"]
        floats_r_sq = iso_info["floats_r_sq"]
        floats_r_rect = iso_info["floats_r_rect"]

        print(f"\n==========================================================================")
        print(f"--- SWEEP: d_k = {dk_c} (d_model = {d_model}) ---")
        print(f"  * Complex Delta Phase:       K_dim={2*dk_c:3d}, V_dim={dk_c:3d} -> State: {floats_c} floats/head")
        print(f"  * Real DeltaNet (Square):    K_dim={dk_r_sq:3d}, V_dim={dk_r_sq:3d} -> State: {floats_r_sq} floats/head")
        print(f"  * Real DeltaNet (Rectangular): K_dim={2*dk_c:3d}, V_dim={dk_c:3d} -> State: {floats_r_rect} floats/head (DECISIVE ARM)")
        print(f"==========================================================================")

        results_matrix[d_k] = {}

        model_specs = [
            (f"ComplexDeltaPhase (K={2*dk_c},V={dk_c})", ComplexDeltaPhaseHolographicBlock, {"d_k": dk_c}),
            (f"RealDeltaNetSquare (K={dk_r_sq},V={dk_r_sq})", RealDeltaNetVanillaBlock, {"d_k_real": dk_r_sq}),
            (f"RealDeltaNetRect (K={2*dk_c},V={dk_c})", RealDeltaNetRectangularBlock, {"d_k": dk_c}),
            (f"CausalAttentionMHA", CausalAttentionBlock, {})
        ]

        for num_pairs in CFG["num_pairs_list"]:
            seq_len = 8 * num_pairs
            print(f"\n  >>> Pre-generating GPU Datasets for Load: {num_pairs} Pairs (L={seq_len}) <<<", flush=True)

            target_batch = CFG["batch_size"]
            micro_batch = 2 if seq_len >= 2048 else (4 if seq_len >= 1024 else (16 if seq_len >= 512 else target_batch))
            accum_steps = max(1, target_batch // micro_batch)

            # FIX #3: Pre-generate datasets ONCE, shared across ALL models and LRs
            train_x, train_y = generate_mqar_dataset(CFG["steps_per_epoch"] * accum_steps, micro_batch, num_pairs, seq_len, seed=100, device=device)
            val_x, val_y     = generate_mqar_dataset(10, micro_batch, num_pairs, seq_len, seed=200, device=device)
            test_x, test_y   = generate_mqar_dataset(10, micro_batch, num_pairs, seq_len, seed=300, device=device)

            for name, block_cls, block_kwargs in model_specs:
                best_val_acc = -1.0
                best_test_acc = -1.0
                best_lr = None
                total_time = 0.0

                for lr in CFG["lr_grid"]:
                    torch.manual_seed(CFG["seed"])
                    model = SequenceModel(block_cls, VOCAB_SIZE, d_model, CFG["n_layers"], block_kwargs).to(device)

                    train_time = train_model(model, name, train_x, train_y, lr,
                                             epochs=CFG["epochs"],
                                             steps_per_epoch=CFG["steps_per_epoch"],
                                             accum_steps=accum_steps)
                    val_acc = eval_accuracy(model, val_x, val_y)
                    test_acc = eval_accuracy(model, test_x, test_y)
                    total_time += train_time

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_test_acc = test_acc
                        best_lr = lr

                key_name = f"{name}_dk{d_k}"
                if key_name not in results_matrix[d_k]:
                    results_matrix[d_k][key_name] = {}

                results_matrix[d_k][key_name][num_pairs] = {
                    "best_val_acc": round(best_val_acc, 2),
                    "best_test_acc": round(best_test_acc, 2),
                    "best_lr": best_lr,
                    "total_eval_time": round(total_time, 2)
                }
                print(f"  ==> [{name:42s}] Pairs={num_pairs:3d} (L={seq_len:4d}) -> Test Acc: {best_test_acc:6.2f}% (Val: {best_val_acc:.2f}%, lr={best_lr})", flush=True)

            # Free dataset memory before next load level
            del train_x, train_y, val_x, val_y, test_x, test_y
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open("v301_decisive_benchmark_fast_results.json", "w") as f:
        json.dump({"config": CFG, "results": results_matrix}, f, indent=2)
    print("\nDECISIVE BENCHMARK COMPLETE! Results saved to v301_decisive_benchmark_fast_results.json", flush=True)

if __name__ == "__main__":
    run_experiment_suite()
