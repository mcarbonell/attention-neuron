import math, time, os, json, sys, torch
import torch.nn as nn
import torch.nn.functional as F

CFG = {
    "exp_id": "v300_capacity_scaling_remaining_ultra_fast_v2",
    "d_k_list": [32, 64, 128],
    "iso_floats_map": {
        32: {"dk_complex": 32, "dk_real": 45, "floats_c": 2048, "floats_r": 2025},
        64: {"dk_complex": 64, "dk_real": 90, "floats_c": 8192, "floats_r": 8100},
        128: {"dk_complex": 128, "dk_real": 181, "floats_c": 32768, "floats_r": 32761},
        "32": {"dk_complex": 32, "dk_real": 45, "floats_c": 2048, "floats_r": 2025},
        "64": {"dk_complex": 64, "dk_real": 90, "floats_c": 8192, "floats_r": 8100},
        "128": {"dk_complex": 128, "dk_real": 181, "floats_c": 32768, "floats_r": 32761}
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

device = torch.device(CFG["device"])

PAD_ID = 0
KEY_OFFSET = 1
VAL_OFFSET = 1 + CFG["num_keys"]
QUERY_MARKER = VAL_OFFSET + CFG["num_vals"]
VOCAB_SIZE = QUERY_MARKER + 1

# ── 1. Ultra-Fast Vectorized Dataset Generator (100% on GPU) ─────────────
def generate_mqar_batch_vectorized(batch_size, num_pairs, seq_len, num_keys=256, num_vals=256, device=device):
    rand_k = torch.rand(batch_size, num_keys, device=device)
    keys = torch.argsort(rand_k, dim=-1)[:, :num_pairs] + KEY_OFFSET
    vals = torch.randint(0, num_vals, (batch_size, num_pairs), device=device) + VAL_OFFSET
    
    x = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    
    kv_interleaved = torch.stack([keys, vals], dim=2).view(batch_size, 2 * num_pairs)
    x[:, :2 * num_pairs] = kv_interleaved
    
    rand_q = torch.rand(batch_size, num_pairs, device=device)
    query_perm = torch.argsort(rand_q, dim=-1)
    
    q_keys = torch.gather(keys, 1, query_perm)
    q_vals = torch.gather(vals, 1, query_perm)
    
    n_queries = min(num_pairs, (seq_len - 2 * num_pairs - 2) // 2)
    pos_q = (2 * num_pairs + 2 + 2 * torch.arange(n_queries, device=device)).unsqueeze(0).expand(batch_size, -1)
    
    x.scatter_(1, pos_q, QUERY_MARKER)
    x.scatter_(1, pos_q + 1, q_keys[:, :n_queries])
    y.scatter_(1, pos_q + 1, q_vals[:, :n_queries])
    return x, y

def generate_mqar_dataset(num_batches, batch_size, num_pairs, seq_len, seed=42, device=device):
    torch.manual_seed(seed)
    x_list, y_list = [], []
    for _ in range(num_batches):
        x, y = generate_mqar_batch_vectorized(batch_size, num_pairs, seq_len, device=device)
        x_list.append(x)
        y_list.append(y)
    return torch.stack(x_list), torch.stack(y_list)

# ── 2. Model Architectures ──────────────────────────────────────────────
class SinCosPE(nn.Module):
    def __init__(self, d_model, max_len=4096):
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
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size-1, groups=d_model)
        self.act = nn.SiLU()
    def forward(self, x):
        B, L, D = x.shape
        return x + self.act(self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2))

class FFN(nn.Module):
    def __init__(self, d_model, expand=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model * expand), nn.SiLU(), nn.Linear(d_model * expand, d_model))
    def forward(self, x): return self.net(x)

# ── OPTIMIZED: Chunk-wise Delta Rule forward ─────────────────────────────
# Instead of looping t=0..L-1 one step at a time, we process C steps per
# iteration.  Inside each chunk we still need the sequential M update, but
# the chunk boundaries batch the matmuls into bigger tensors so CUDA gets
# more work per kernel launch.  With C=16 on L=2048 we go from 2048 Python
# iterations down to 128.

CHUNK_SIZE = 16  # Tune: 8-32 is the sweet spot on T4/P100

def _delta_rule_chunked_real(K, Q, V, beta, d_k):
    """Chunk-wise delta-rule scan for the Real variant.
    K, Q, V: (B, L, H, d_k)   beta: (B, L, H, 1, 1)
    Returns: (B, L, H, d_k)
    """
    B, L, H, dk = K.shape
    C = min(CHUNK_SIZE, L)
    M = torch.zeros(B, H, dk, dk, dtype=K.dtype, device=K.device)
    out_chunks = []

    for start in range(0, L, C):
        end = min(start + C, L)
        chunk_len = end - start

        # Slice the chunk: (B, chunk_len, H, dk)
        k_c = K[:, start:end]
        q_c = Q[:, start:end]
        v_c = V[:, start:end]
        b_c = beta[:, start:end]  # (B, chunk_len, H, 1, 1)

        # Sequential scan within the chunk (small loop: C iterations max)
        chunk_out = []
        for i in range(chunk_len):
            k_t = k_c[:, i]  # (B, H, dk)
            q_t = q_c[:, i]
            v_t = v_c[:, i]
            beta_t = b_c[:, i]  # (B, H, 1, 1)

            v_old = torch.matmul(M, k_t.unsqueeze(-1)).squeeze(-1)
            err = v_t - v_old
            M = M + beta_t * (err.unsqueeze(-1) * k_t.unsqueeze(-2))
            ret = torch.matmul(M, q_t.unsqueeze(-1)).squeeze(-1)
            chunk_out.append(ret)

        out_chunks.append(torch.stack(chunk_out, dim=1))  # (B, chunk_len, H, dk)

    return torch.cat(out_chunks, dim=1)  # (B, L, H, dk)


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
        chunk_len = end - start

        k_c = K[:, start:end]
        q_c = Q[:, start:end]
        v_c = V[:, start:end]
        b_c = beta[:, start:end]

        chunk_out = []
        for i in range(chunk_len):
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


class ComplexDeltaPhaseHolographicBlock(nn.Module):
    def __init__(self, d_model, n_heads=2, d_k=64):
        super().__init__()
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_k
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
        B, L, D = conv_x.shape
        theta_k = self.theta_k_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        theta_q = self.theta_q_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        v = self.val_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        beta = torch.sigmoid(self.beta_proj(conv_x)).view(B, L, self.n_heads, 1, 1)
        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)

        retrieved = _delta_rule_chunked_complex(K, Q, v, beta, self.d_k)
        retrieved = retrieved.reshape(B, L, self.n_heads * self.d_k)

        attn_out = self.out_proj(retrieved)
        return res + attn_out + self.ffn(self.norm2(res + attn_out))

class RealDeltaNetVanillaBlock(nn.Module):
    def __init__(self, d_model, n_heads=2, d_k_real=90):
        super().__init__()
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_k_real
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_k)
        self.q_proj = nn.Linear(d_model, n_heads * self.d_k)
        self.val_proj = nn.Linear(d_model, n_heads * self.d_k)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(n_heads * self.d_k, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x; conv_x = self.causal_conv(self.norm1(x))
        B, L, D = conv_x.shape
        K = F.normalize(self.k_proj(conv_x).view(B, L, self.n_heads, self.d_k), p=2, dim=-1)
        Q = F.normalize(self.q_proj(conv_x).view(B, L, self.n_heads, self.d_k), p=2, dim=-1)
        v = self.val_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        beta = torch.sigmoid(self.beta_proj(conv_x)).view(B, L, self.n_heads, 1, 1)

        retrieved = _delta_rule_chunked_real(K, Q, v, beta, self.d_k)
        retrieved = retrieved.reshape(B, L, self.n_heads * self.d_k)

        attn_out = self.out_proj(retrieved)
        return res + attn_out + self.ffn(self.norm2(res + attn_out))

class CausalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=2):
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
    def __init__(self, block_cls, vocab_size, d_model, n_layers=3, block_kwargs=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = SinCosPE(d_model)
        self.layers = nn.ModuleList([block_cls(d_model=d_model, **(block_kwargs or {})) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        h = self.pe(self.emb(x))
        for layer in self.layers: h = layer(h)
        return self.head(h)

# ── 3. Training Loop with Pre-Generated GPU Datasets ────────────────────
def train_and_eval_pregenerated(model, model_name, train_x, train_y, eval_x, eval_y, lr, epochs=15, steps_per_epoch=50, seq_len=256):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    target_batch = CFG["batch_size"]
    micro_batch = 2 if seq_len >= 2048 else (4 if seq_len >= 1024 else (16 if seq_len >= 512 else target_batch))
    accum_steps = max(1, target_batch // micro_batch)
    
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
        print(f"      [{model_name:28s} | lr={lr:.4f}] Epoch {ep+1:2d}/15 Complete | Loss = {total_loss:.4f}", flush=True)
    eval_time = time.time() - start_time
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(len(eval_x)):
            logits = model(eval_x[i]); preds = logits.argmax(dim=-1); mask = (eval_y[i] != -100)
            correct += (preds[mask] == eval_y[i][mask]).sum().item(); total += mask.sum().item()
    acc = (correct / total) * 100.0 if total > 0 else 0.0
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return acc, eval_time

# ── Rich Metadata Header Printing ───────────────────────────────────────
print("=" * 85)
print(f"EXPERIMENT BENCHMARK RUN: V300 CAPACITY SCALING (OPTIMIZED: chunk-wise + compile)")
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
print(f"  * torch.compile:   disabled (complex ops unsupported by inductor)")
print(f"  * ISO-FLOATS MAP:")
for dk, info in CFG["iso_floats_map"].items():
    if isinstance(dk, int):
        print(f"      d_k={dk:3d} -> Complex d_k={info['dk_complex']:3d} ({info['floats_c']:5d} floats/head) | Real d_k={info['dk_real']:3d} ({info['floats_r']:5d} floats/head)")
print("=" * 85, flush=True)

results_matrix = {}
for d_k in CFG["d_k_list"]:
    d_k_key = int(d_k)
    iso_info = CFG["iso_floats_map"][d_k_key]
    dk_c, dk_r = iso_info["dk_complex"], iso_info["dk_real"]
    d_model = 2 * dk_c
    print(f"\n=== SWEEP d_k = {dk_c} (d_model = {d_model}) ===", flush=True)
    results_matrix[d_k_key] = {}
    model_specs = [
        ("ComplexDeltaPhaseHolographic", ComplexDeltaPhaseHolographicBlock, {"d_k": dk_c}),
        ("RealDeltaNetVanilla", RealDeltaNetVanillaBlock, {"d_k_real": dk_r}),
        ("CausalAttentionMHA", CausalAttentionBlock, {})
    ]
    for num_pairs in CFG["num_pairs_list"]:
        seq_len = 8 * num_pairs
        print(f"\n  >>> Pre-generating GPU Datasets for Load: {num_pairs} Pairs (L={seq_len}) <<<", flush=True)
        
        target_batch = CFG["batch_size"]
        micro_batch = 2 if seq_len >= 2048 else (4 if seq_len >= 1024 else (16 if seq_len >= 512 else target_batch))
        accum_steps = max(1, target_batch // micro_batch)
        
        # Pre-generate datasets ONCE on GPU per load
        train_x, train_y = generate_mqar_dataset(CFG["steps_per_epoch"] * accum_steps, micro_batch, num_pairs, seq_len, seed=100, device=device)
        eval_x, eval_y   = generate_mqar_dataset(10, micro_batch, num_pairs, seq_len, seed=200, device=device)
        
        for name, block_cls, block_kwargs in model_specs:
            best_acc, best_lr, total_time = -1.0, None, 0.0
            for lr in CFG["lr_grid"]:
                torch.manual_seed(CFG["seed"])
                model = SequenceModel(block_cls, VOCAB_SIZE, d_model, CFG["n_layers"], block_kwargs).to(device)
                acc, eval_t = train_and_eval_pregenerated(model, name, train_x, train_y, eval_x, eval_y, lr, epochs=CFG["epochs"], steps_per_epoch=CFG["steps_per_epoch"], seq_len=seq_len)
                total_time += eval_t
                if acc > best_acc:
                    best_acc, best_lr = acc, lr
            key_name = f"{name}_dk{d_k_key}"
            if key_name not in results_matrix[d_k_key]: results_matrix[d_k_key][key_name] = {}
            results_matrix[d_k_key][key_name][num_pairs] = {"best_acc": round(best_acc, 2), "best_lr": best_lr, "total_eval_time": round(total_time, 2)}
            print(f"      [{name:28s} | d_k={d_k_key}] Pairs={num_pairs:3d} (L={seq_len:4d}) -> Best Acc: {best_acc:6.2f}% (lr={best_lr})", flush=True)

with open("v300_capacity_scaling_remaining_results.json", "w") as f:
    json.dump({"config": CFG, "results": results_matrix}, f, indent=2)
print("\nBENCHMARK COMPLETE! Results saved to v300_capacity_scaling_remaining_results.json", flush=True)
