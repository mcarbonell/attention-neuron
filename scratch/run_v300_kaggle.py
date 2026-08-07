"""
run_v300_kaggle.py
============================
Ultra-Fast Vectorized Version of V300 Benchmark for Kaggle GPU:
- Enriched Metadata Header for self-documenting logs.
- Fixed 2D pos_q expansion for scatter across all batch rows.
- Reduces CPU launch/synchronization overhead while keeping batches on GPU.
- Supports a small smoke-test sweep before launching the full benchmark.
"""

import argparse
import math, time, os, json, sys, torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_START = time.perf_counter()


def elapsed_timestamp():
    """Elapsed wall-clock time since this process was started (HH:MM:SS)."""
    elapsed = int(time.perf_counter() - SCRIPT_START)
    return f"+{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"


def log(message=""):
    print(f"[{elapsed_timestamp()}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="V300 complex phase-memory benchmark")
    parser.add_argument(
        "--mode", choices=("full", "lite"), default="lite",
        help="'lite' is a quick smoke test; the default runs the full benchmark.",
    )
    parser.add_argument("--d-k", nargs="+", type=int, dest="d_k_list", help="Override d_k sweep values.")
    parser.add_argument("--pairs", nargs="+", type=int, dest="num_pairs_list", help="Override KV-pair sweep values.")
    parser.add_argument("--output", help="Path for the JSON results file.")
    # Colab/Kaggle/Jupyter insert kernel arguments (normally ``-f ...``) into
    # sys.argv when this file is pasted into a cell. Ignore only those unknown
    # arguments; normal command-line options above are still honored.
    args, _unknown = parser.parse_known_args()
    return args

CFG = {
    "exp_id": "v300_capacity_scaling_all",
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
    "compile_non_complex": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

args = parse_args()
if args.mode == "lite":
    # Fast sanity check: 9 train/eval runs instead of 72 in the full sweep.
    CFG.update({"d_k_list": [64], "num_pairs_list": [32, 64, 128], "epochs": 5,
                "steps_per_epoch": 20, "lr_grid": [2e-3]})
if args.d_k_list is not None:
    CFG["d_k_list"] = args.d_k_list
if args.num_pairs_list is not None:
    CFG["num_pairs_list"] = args.num_pairs_list
output_path = args.output or (
    "v300_capacity_scaling_remaining_results_lite.json"
    if args.mode == "lite" else "v300_capacity_scaling_remaining_results.json"
)

if not set(CFG["d_k_list"]).issubset({32, 64, 128}):
    raise ValueError("--d-k must contain only values from: 32, 64, 128")
if not set(CFG["num_pairs_list"]).issubset(set(range(1, CFG["num_keys"] + 1))):
    raise ValueError(f"--pairs must be in [1, {CFG['num_keys']}]")

device = torch.device(CFG["device"])

if device.type == "cuda":
    # Use Tensor Cores for the real-valued projections/matmuls. Complex kernels
    # retain their native precision.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # A sweep creates several static graphs; avoid recompiling after Dynamo's
    # conservative default cache limit is reached.
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.suppress_errors = True

PAD_ID = 0
KEY_OFFSET = 1
VAL_OFFSET = 1 + CFG["num_keys"]
QUERY_MARKER = VAL_OFFSET + CFG["num_vals"]
VOCAB_SIZE = QUERY_MARKER + 1

# â”€â”€ 1. Ultra-Fast Vectorized Dataset Generator (100% on GPU) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    # Generate the complete dataset in one GPU operation instead of issuing
    # Python-driven kernels once per minibatch.
    x, y = generate_mqar_batch_vectorized(
        num_batches * batch_size, num_pairs, seq_len, device=device
    )
    return x.reshape(num_batches, batch_size, seq_len), y.reshape(num_batches, batch_size, seq_len)

# â”€â”€ 2. Model Architectures â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=conv_x.device)
        out_retrieved = []; inv_dk = 1.0 / float(self.d_k)
        for t in range(L):
            k_t, q_t, v_t, beta_t = K[:, t], Q[:, t], v[:, t], beta[:, t]
            k_conj, q_conj = torch.conj(k_t), torch.conj(q_t)
            # Fuse M@k* and M@q* into one matrix multiplication.  The value
            # returned after the rank-one update follows directly from
            # M_new@q* = M_old@q* + beta*err*(k@q*).
            kq_conj = torch.stack((k_conj, q_conj), dim=-1)
            projections = torch.matmul(M, kq_conj)
            v_old = projections[..., 0].real * inv_dk
            err = v_t - v_old
            err_complex = err.to(torch.complex64)
            update = err_complex.unsqueeze(-1) * k_t.unsqueeze(-2)
            M = M + beta_t * update
            beta_scalar = beta_t.squeeze(-1).squeeze(-1).unsqueeze(-1)
            ret = (projections[..., 1] + beta_scalar * err_complex *
                   torch.sum(k_t * q_conj, dim=-1).unsqueeze(-1)).real * inv_dk
            out_retrieved.append(ret)
        retrieved = torch.stack(out_retrieved, dim=1).view(B, L, self.n_heads * self.d_k)
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
        M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.float32, device=conv_x.device)
        out_retrieved = []
        for t in range(L):
            k_t, q_t, v_t, beta_t = K[:, t], Q[:, t], v[:, t], beta[:, t]
            kq = torch.stack((k_t, q_t), dim=-1)
            projections = torch.matmul(M, kq)
            v_old = projections[..., 0]
            err = v_t - v_old
            M = M + beta_t * (err.unsqueeze(-1) * k_t.unsqueeze(-2))
            beta_scalar = beta_t.squeeze(-1).squeeze(-1).unsqueeze(-1)
            ret = projections[..., 1] + beta_scalar * err * torch.sum(k_t * q_t, dim=-1).unsqueeze(-1)
            out_retrieved.append(ret)
        retrieved = torch.stack(out_retrieved, dim=1).view(B, L, self.n_heads * self.d_k)
        attn_out = self.out_proj(retrieved)
        return res + attn_out + self.ffn(self.norm2(res + attn_out))

class CausalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=2):
        super().__init__()
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
        max_len = max(4 * pairs + 2 for pairs in CFG["num_pairs_list"])
        self.register_buffer("causal_mask", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1))
    def forward(self, x):
        res = x; conv_x = self.causal_conv(self.norm1(x))
        B, L, D = conv_x.shape
        attn_out, _ = self.mha(conv_x, conv_x, conv_x, attn_mask=self.causal_mask[:L, :L], is_causal=False)
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

# â”€â”€ 3. Training Loop with Pre-Generated GPU Datasets â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def train_and_eval_pregenerated(model, model_name, train_x, train_y, eval_x, eval_y, lr, epochs=15, steps_per_epoch=50, seq_len=256):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    target_batch = CFG["batch_size"]
    micro_batch = 2 if seq_len >= 2048 else (4 if seq_len >= 1024 else (16 if seq_len >= 512 else target_batch))
    accum_steps = max(1, target_batch // micro_batch)

    # Inductor removes much of the Python/kernel-launch overhead for the real
    # scan and MHA. Complex kernels are deliberately left eager because
    # Inductor's complex support is still incomplete.
    if device.type == "cuda" and CFG["compile_non_complex"] and "Complex" not in model_name:
        model = torch.compile(model)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    for ep in range(epochs):
        model.train()
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            total_loss = torch.zeros((), dtype=torch.float32, device=device)
            for acc_i in range(accum_steps):
                idx = (step * accum_steps + acc_i) % len(train_x)
                logits = model(train_x[idx])
                loss = criterion(logits.view(-1, VOCAB_SIZE), train_y[idx].view(-1)) / accum_steps
                loss.backward()
                total_loss = total_loss + loss.detach() * accum_steps
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        # .item() deliberately happens only once per epoch; it also ensures
        # that this timestamp represents completed GPU work.
        log(f"      [{model_name:28s} | lr={lr:.4f}] Epoch {ep+1:2d}/{epochs} Complete | Loss = {total_loss.item():.4f}")

    model.eval()
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = torch.zeros((), dtype=torch.long, device=device)
    with torch.no_grad():
        for i in range(len(eval_x)):
            logits = model(eval_x[i]); preds = logits.argmax(dim=-1); mask = (eval_y[i] != -100)
            correct += (preds[mask] == eval_y[i][mask]).sum()
            total += mask.sum()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    total_items = total.item()
    acc = (correct.item() / total_items) * 100.0 if total_items > 0 else 0.0
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return acc, elapsed

evaluated_lengths = [4 * num_pairs + 2 for num_pairs in CFG["num_pairs_list"]]
log("=" * 85)
log("EXPERIMENT BENCHMARK RUN: V300 CAPACITY SCALING (GPU-OPTIMIZED)")
log("=" * 85)
log(f"  * Run mode:        {args.mode}")
log(f"  * Exp ID:          {CFG['exp_id']}")
log(f"  * Hardware Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
log(f"  * PyTorch Version: {torch.__version__}")
log(f"  * CUDA Available:  {torch.cuda.is_available()} (Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'})")
log(f"  * Seed:            {CFG['seed']}")
log(f"  * Layers:          {CFG['n_layers']}")
log(f"  * Effective Batch: {CFG['batch_size']}")
log(f"  * Epochs / Steps:  {CFG['epochs']} epochs, {CFG['steps_per_epoch']} steps/epoch")
log(f"  * LR Grid:         {CFG['lr_grid']}")
log(f"  * Sweeps d_k:      {CFG['d_k_list']}")
log(f"  * KV Pairs Sweep:  {CFG['num_pairs_list']}")
log(f"  * Sequence L:      {evaluated_lengths}")
log(f"  * torch.compile:   {'real delta + MHA' if CFG['compile_non_complex'] and device.type == 'cuda' else 'disabled'}")
log(f"  * Results file:    {output_path}")
log(f"  * ISO-FLOATS MAP:")
for dk, info in CFG["iso_floats_map"].items():
    if isinstance(dk, int):
        log(f"      d_k={dk:3d} -> Complex d_k={info['dk_complex']:3d} ({info['floats_c']:5d} floats/head) | Real d_k={info['dk_real']:3d} ({info['floats_r']:5d} floats/head)")
log("=" * 85)

results_matrix = {}
for d_k in CFG["d_k_list"]:
    d_k_key = int(d_k)
    iso_info = CFG["iso_floats_map"][d_k_key]
    dk_c, dk_r = iso_info["dk_complex"], iso_info["dk_real"]
    d_model = 2 * dk_c
    log(f"\n=== SWEEP d_k = {dk_c} (d_model = {d_model}) ===")
    results_matrix[d_k_key] = {}
    model_specs = [
        ("ComplexDeltaPhaseHolographic", ComplexDeltaPhaseHolographicBlock, {"d_k": dk_c}),
        ("RealDeltaNetVanilla", RealDeltaNetVanillaBlock, {"d_k_real": dk_r}),
        ("CausalAttentionMHA", CausalAttentionBlock, {})
    ]
    for num_pairs in CFG["num_pairs_list"]:
        # KV tokens (2P), separator (2), and P query/key pairs (2P): all
        # positions are useful. The previous 8P setting spent half the scan
        # on trailing PAD tokens without adding a single target.
        seq_len = 4 * num_pairs + 2
        log(f"\n  >>> Pre-generating GPU Datasets for Load: {num_pairs} Pairs (L={seq_len}) <<<")

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
            log(f"      [{name:28s} | d_k={d_k_key}] Pairs={num_pairs:3d} (L={seq_len:4d}) -> Best Acc: {best_acc:6.2f}% (lr={best_lr})")

with open(output_path, "w") as f:
    json.dump({"config": CFG, "results": results_matrix}, f, indent=2)
log(f"\nBENCHMARK COMPLETE! Results saved to {output_path}")
