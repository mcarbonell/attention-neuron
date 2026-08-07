
"""
run_v300_kaggle.py
==================
V300 capacity scaling — memory/runtime hardened:
- Chunked gradient checkpointing over the Delta scan (fixes OOM from unrolled M_t)
- Micro-batch sizing depends on complex vs real, d_k, L
- Optional on-the-fly batch gen (less GPU dataset residency)
- Model teardown between runs
"""

import argparse
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SCRIPT_START = time.perf_counter()


def elapsed_timestamp():
    elapsed = int(time.perf_counter() - SCRIPT_START)
    return f"+{elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}"


def log(message=""):
    print(f"[{elapsed_timestamp()}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="V300 complex phase-memory benchmark")
    parser.add_argument("--mode", choices=("full", "lite"), default="lite")
    parser.add_argument("--d-k", nargs="+", type=int, dest="d_k_list")
    parser.add_argument("--pairs", nargs="+", type=int, dest="num_pairs_list")
    parser.add_argument("--output", help="Path for the JSON results file.")
    parser.add_argument("--scan-chunk", type=int, default=32,
                        help="Timesteps per grad-checkpoint chunk (16–64 sweet spot).")
    parser.add_argument("--on-the-fly-data", action="store_true", default=True,
                        help="Generate MQAR batches each step (saves GPU RAM).")
    args, _unknown = parser.parse_known_args()
    return args


CFG = {
    "exp_id": "v300_capacity_scaling_all",
    "d_k_list": [32, 64, 128],
    "iso_floats_map": {
        32: {"dk_complex": 32, "dk_real": 45, "floats_c": 2048, "floats_r": 2025},
        64: {"dk_complex": 64, "dk_real": 90, "floats_c": 8192, "floats_r": 8100},
        128: {"dk_complex": 128, "dk_real": 181, "floats_c": 32768, "floats_r": 32761},
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
    "scan_chunk_size": 32,
    "on_the_fly_data": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

args = parse_args()
if args.mode == "lite":
    CFG.update({
        "d_k_list": [64],
        "num_pairs_list": [32, 64, 128],
        "epochs": 5,
        "steps_per_epoch": 20,
        "lr_grid": [2e-3],
    })
if args.d_k_list is not None:
    CFG["d_k_list"] = args.d_k_list
if args.num_pairs_list is not None:
    CFG["num_pairs_list"] = args.num_pairs_list
CFG["scan_chunk_size"] = args.scan_chunk
CFG["on_the_fly_data"] = args.on_the_fly_data

output_path = args.output or (
    "v300_capacity_scaling_remaining_results_lite.json"
    if args.mode == "lite"
    else "v300_capacity_scaling_remaining_results.json"
)

if not set(CFG["d_k_list"]).issubset({32, 64, 128}):
    raise ValueError("--d-k must contain only values from: 32, 64, 128")
if not set(CFG["num_pairs_list"]).issubset(set(range(1, CFG["num_keys"] + 1))):
    raise ValueError(f"--pairs must be in [1, {CFG['num_keys']}]")

device = torch.device(CFG["device"])

if device.type == "cuda":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.suppress_errors = True

PAD_ID = 0
KEY_OFFSET = 1
VAL_OFFSET = 1 + CFG["num_keys"]
QUERY_MARKER = VAL_OFFSET + CFG["num_vals"]
VOCAB_SIZE = QUERY_MARKER + 1


# ── micro-batch heuristic ───────────────────────────────────────────────
def recommend_micro_batch(seq_len, d_k, is_complex, target_batch, n_layers=3, n_heads=2):
    """
    Rough VRAM budget for checkpointed scan:
      resid ≈ 2 * B * H * chunk * d_k^2 * bytes * n_layers
    Keep B small when d_k or L is large; accumulate to target_batch.
    """
    bytes_m = 8 if is_complex else 4
    # empirical safety table for ~14–16 GB class GPUs (T4/P100)
    if is_complex:
        if d_k >= 128 and seq_len >= 500:
            mb = 2
        elif d_k >= 128:
            mb = 4
        elif d_k >= 64 and seq_len >= 500:
            mb = 4
        elif d_k >= 64:
            mb = 8
        else:
            mb = 16
    else:
        if d_k >= 180 and seq_len >= 500:
            mb = 4
        elif seq_len >= 1000:
            mb = 8
        else:
            mb = min(32, target_batch)

    mb = min(mb, target_batch)
    # ensure divides target for clean accum (not required, but nice)
    while target_batch % mb != 0 and mb > 1:
        mb -= 1
    return max(1, mb)


# ── data ────────────────────────────────────────────────────────────────
def generate_mqar_batch_vectorized(batch_size, num_pairs, seq_len,
                                   num_keys=256, num_vals=256, device=device):
    rand_k = torch.rand(batch_size, num_keys, device=device)
    keys = torch.argsort(rand_k, dim=-1)[:, :num_pairs] + KEY_OFFSET
    vals = torch.randint(0, num_vals, (batch_size, num_pairs), device=device) + VAL_OFFSET

    x = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    kv_interleaved = torch.stack([keys, vals], dim=2).view(batch_size, 2 * num_pairs)
    x[:, : 2 * num_pairs] = kv_interleaved

    rand_q = torch.rand(batch_size, num_pairs, device=device)
    query_perm = torch.argsort(rand_q, dim=-1)
    q_keys = torch.gather(keys, 1, query_perm)
    q_vals = torch.gather(vals, 1, query_perm)

    n_queries = min(num_pairs, max(0, (seq_len - 2 * num_pairs - 2) // 2))
    if n_queries == 0:
        return x, y

    pos_q = (2 * num_pairs + 2 + 2 * torch.arange(n_queries, device=device))
    pos_q = pos_q.unsqueeze(0).expand(batch_size, -1)

    x.scatter_(1, pos_q, QUERY_MARKER)
    x.scatter_(1, pos_q + 1, q_keys[:, :n_queries])
    y.scatter_(1, pos_q + 1, q_vals[:, :n_queries])
    return x, y


# ── building blocks ─────────────────────────────────────────────────────
class SinCosPE(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]]


class ShortCausalConv1D(nn.Module):
    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size - 1, groups=d_model,
        )
        self.act = nn.SiLU()

    def forward(self, x):
        B, L, D = x.shape
        y = self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        return x + self.act(y)


class FFN(nn.Module):
    def __init__(self, d_model, expand=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.SiLU(),
            nn.Linear(d_model * expand, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ── checkpointed scans ──────────────────────────────────────────────────
def _complex_scan_chunk(M, k_c, q_c, v_c, beta_c, inv_dk):
    """
    M:      (B,H,D,D) complex
    k_c..:  (B,T,H,D) complex / float / beta
    returns M_out, ret (B,T,H,D) float
    """
    B, T, H, D = v_c.shape
    outs = []
    for t in range(T):
        k_t = k_c[:, t]          # (B,H,D)
        q_t = q_c[:, t]
        v_t = v_c[:, t]
        beta_t = beta_c[:, t]    # (B,H,1,1)

        k_conj = torch.conj(k_t)
        q_conj = torch.conj(q_t)
        kq = torch.stack((k_conj, q_conj), dim=-1)          # (B,H,D,2)
        proj = torch.matmul(M, kq)                          # (B,H,D,2)

        v_old = proj[..., 0].real * inv_dk
        err = v_t - v_old
        err_c = err.to(dtype=M.dtype)
        M = M + beta_t * (err_c.unsqueeze(-1) * k_t.unsqueeze(-2))

        # ret = Re(M_new @ q*) / d_k, algebraically without extra matmul:
        beta_s = beta_t.squeeze(-1).squeeze(-1).unsqueeze(-1)  # (B,H,1)
        kq_dot = torch.sum(k_t * q_conj, dim=-1, keepdim=True)
        ret = (proj[..., 1] + beta_s * err_c * kq_dot).real * inv_dk
        outs.append(ret)

    return M, torch.stack(outs, dim=1)


def _real_scan_chunk(M, k_c, q_c, v_c, beta_c):
    B, T, H, D = v_c.shape
    outs = []
    for t in range(T):
        k_t = k_c[:, t]
        q_t = q_c[:, t]
        v_t = v_c[:, t]
        beta_t = beta_c[:, t]

        kq = torch.stack((k_t, q_t), dim=-1)
        proj = torch.matmul(M, kq)
        v_old = proj[..., 0]
        err = v_t - v_old
        M = M + beta_t * (err.unsqueeze(-1) * k_t.unsqueeze(-2))

        beta_s = beta_t.squeeze(-1).squeeze(-1).unsqueeze(-1)
        kq_dot = torch.sum(k_t * q_t, dim=-1, keepdim=True)
        ret = proj[..., 1] + beta_s * err * kq_dot
        outs.append(ret)

    return M, torch.stack(outs, dim=1)


def checkpointed_delta_scan(scan_fn, M, K, Q, V, beta, chunk_size, *extra):
    """
    K,Q: (B,L,H,D)   V: (B,L,H,D)   beta: (B,L,H,1,1)
    Recomputes each chunk on backward → stores O(L/chunk) boundaries, not O(L).
    """
    B, L, H, D = V.shape
    rets = []
    use_ckpt = torch.is_grad_enabled() and M.requires_grad is False
    # M is a leaf zeros state; inputs K,Q,V,beta need grad. Checkpoint on chunks.
    use_ckpt = torch.is_grad_enabled()

    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        k_c = K[:, start:end]
        q_c = Q[:, start:end]
        v_c = V[:, start:end]
        b_c = beta[:, start:end]

        if use_ckpt and (end - start) > 1:
            M, ret = checkpoint(
                scan_fn, M, k_c, q_c, v_c, b_c, *extra,
                use_reentrant=False,
            )
        else:
            M, ret = scan_fn(M, k_c, q_c, v_c, b_c, *extra)
        rets.append(ret)

    return torch.cat(rets, dim=1)


class ComplexDeltaPhaseHolographicBlock(nn.Module):
    def __init__(self, d_model, n_heads=2, d_k=64, scan_chunk=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.scan_chunk = scan_chunk
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.theta_k_proj = nn.Linear(d_model, n_heads * d_k)
        self.theta_q_proj = nn.Linear(d_model, n_heads * d_k)
        self.val_proj = nn.Linear(d_model, n_heads * d_k)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(n_heads * d_k, d_model)
        self.ffn = FFN(d_model)
        self.inv_dk = 1.0 / float(d_k)

    def forward(self, x):
        res = x
        conv_x = self.causal_conv(self.norm1(x))
        B, L, _ = conv_x.shape
        H, D = self.n_heads, self.d_k

        theta_k = self.theta_k_proj(conv_x).view(B, L, H, D)
        theta_q = self.theta_q_proj(conv_x).view(B, L, H, D)
        v = self.val_proj(conv_x).view(B, L, H, D)
        beta = torch.sigmoid(self.beta_proj(conv_x)).view(B, L, H, 1, 1)

        # unit-modulus phasors
        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)

        M = torch.zeros(B, H, D, D, dtype=torch.complex64, device=conv_x.device)
        retrieved = checkpointed_delta_scan(
            _complex_scan_chunk, M, K, Q, v, beta, self.scan_chunk, self.inv_dk
        )
        attn_out = self.out_proj(retrieved.reshape(B, L, H * D))
        h = res + attn_out
        return h + self.ffn(self.norm2(h))


class RealDeltaNetVanillaBlock(nn.Module):
    def __init__(self, d_model, n_heads=2, d_k_real=90, scan_chunk=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k_real
        self.scan_chunk = scan_chunk
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
        conv_x = self.causal_conv(self.norm1(x))
        B, L, _ = conv_x.shape
        H, D = self.n_heads, self.d_k

        K = F.normalize(self.k_proj(conv_x).view(B, L, H, D), p=2, dim=-1)
        Q = F.normalize(self.q_proj(conv_x).view(B, L, H, D), p=2, dim=-1)
        v = self.val_proj(conv_x).view(B, L, H, D)
        beta = torch.sigmoid(self.beta_proj(conv_x)).view(B, L, H, 1, 1)

        M = torch.zeros(B, H, D, D, dtype=torch.float32, device=conv_x.device)
        retrieved = checkpointed_delta_scan(
            _real_scan_chunk, M, K, Q, v, beta, self.scan_chunk
        )
        attn_out = self.out_proj(retrieved.reshape(B, L, H * D))
        h = res + attn_out
        return h + self.ffn(self.norm2(h))


class CausalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=2, scan_chunk=32):  # scan_chunk ignored
        super().__init__()
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
        max_len = max(4 * p + 2 for p in CFG["num_pairs_list"])
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1),
        )

    def forward(self, x):
        res = x
        conv_x = self.causal_conv(self.norm1(x))
        B, L, D = conv_x.shape
        attn_out, _ = self.mha(
            conv_x, conv_x, conv_x,
            attn_mask=self.causal_mask[:L, :L],
            is_causal=False,
        )
        h = res + attn_out
        return h + self.ffn(self.norm2(h))


class SequenceModel(nn.Module):
    def __init__(self, block_cls, vocab_size, d_model, n_layers=3, block_kwargs=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = SinCosPE(d_model)
        kw = dict(block_kwargs or {})
        self.layers = nn.ModuleList(
            [block_cls(d_model=d_model, **kw) for _ in range(n_layers)]
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.pe(self.emb(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


# ── train / eval ────────────────────────────────────────────────────────
def train_and_eval(
    model,
    model_name,
    num_pairs,
    seq_len,
    lr,
    epochs,
    steps_per_epoch,
    micro_batch,
    accum_steps,
    is_complex,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    if (
        device.type == "cuda"
        and CFG["compile_non_complex"]
        and not is_complex
    ):
        model = torch.compile(model)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for ep in range(epochs):
        model.train()
        last_loss = None
        for step in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            for acc_i in range(accum_steps):
                # on-the-fly data: no giant GPU-resident corpus
                xb, yb = generate_mqar_batch_vectorized(
                    micro_batch, num_pairs, seq_len, device=device
                )
                logits = model(xb)
                loss = criterion(
                    logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1)
                ) / accum_steps
                loss.backward()
                last_loss = loss.detach()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        loss_val = float(last_loss.item() * accum_steps) if last_loss is not None else float("nan")
        log(f"      [{model_name:28s} | lr={lr:.4f}] Epoch {ep+1:2d}/{epochs} | Loss≈{loss_val:.4f}")

    # eval
    model.eval()
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = torch.zeros((), dtype=torch.long, device=device)
    n_eval_batches = 10
    with torch.no_grad():
        for _ in range(n_eval_batches):
            xb, yb = generate_mqar_batch_vectorized(
                micro_batch, num_pairs, seq_len, device=device
            )
            preds = model(xb).argmax(dim=-1)
            mask = yb != -100
            correct += (preds[mask] == yb[mask]).sum()
            total += mask.sum()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    tot = int(total.item())
    acc = (float(correct.item()) / tot) * 100.0 if tot > 0 else 0.0
    return acc, elapsed


def free_model(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── main sweep ──────────────────────────────────────────────────────────
evaluated_lengths = [4 * n + 2 for n in CFG["num_pairs_list"]]
log("=" * 85)
log("EXPERIMENT BENCHMARK RUN: V300 CAPACITY SCALING (CHECKPOINT HARDENED)")
log("=" * 85)
log(f"  * Run mode:        {args.mode}")
log(f"  * Device:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
log(f"  * scan_chunk:      {CFG['scan_chunk_size']}")
log(f"  * on_the_fly_data: {CFG['on_the_fly_data']}")
log(f"  * d_k sweep:       {CFG['d_k_list']}")
log(f"  * pairs sweep:     {CFG['num_pairs_list']}  (L={evaluated_lengths})")
log(f"  * epochs/steps:    {CFG['epochs']} x {CFG['steps_per_epoch']}")
log(f"  * lr grid:         {CFG['lr_grid']}")
log(f"  * results:         {output_path}")
log("=" * 85)

results_matrix = {}
scan_chunk = CFG["scan_chunk_size"]

for d_k in CFG["d_k_list"]:
    d_k_key = int(d_k)
    iso = CFG["iso_floats_map"][d_k_key]
    dk_c, dk_r = iso["dk_complex"], iso["dk_real"]
    d_model = 2 * dk_c
    log(f"\n=== SWEEP d_k={dk_c} (d_model={d_model}) ===")
    results_matrix[d_k_key] = {}

    model_specs = [
        ("ComplexDeltaPhaseHolographic", ComplexDeltaPhaseHolographicBlock,
         {"d_k": dk_c, "scan_chunk": scan_chunk}, True, dk_c),
        ("RealDeltaNetVanilla", RealDeltaNetVanillaBlock,
         {"d_k_real": dk_r, "scan_chunk": scan_chunk}, False, dk_r),
        ("CausalAttentionMHA", CausalAttentionBlock,
         {"scan_chunk": scan_chunk}, False, dk_c),
    ]

    for num_pairs in CFG["num_pairs_list"]:
        seq_len = 4 * num_pairs + 2
        log(f"\n  >>> Load: {num_pairs} pairs (L={seq_len}) <<<")

        for name, block_cls, block_kwargs, is_complex, dk_eff in model_specs:
            micro = recommend_micro_batch(
                seq_len, dk_eff, is_complex, CFG["batch_size"], CFG["n_layers"]
            )
            accum = max(1, CFG["batch_size"] // micro)
            log(f"      plan {name}: micro_batch={micro}, accum={accum}, chunk={scan_chunk}")

            best_acc, best_lr, total_time = -1.0, None, 0.0
            for lr in CFG["lr_grid"]:
                torch.manual_seed(CFG["seed"])
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(CFG["seed"])

                model = SequenceModel(
                    block_cls, VOCAB_SIZE, d_model, CFG["n_layers"], block_kwargs
                ).to(device)

                try:
                    acc, eval_t = train_and_eval(
                        model, name, num_pairs, seq_len, lr,
                        epochs=CFG["epochs"],
                        steps_per_epoch=CFG["steps_per_epoch"],
                        micro_batch=micro,
                        accum_steps=accum,
                        is_complex=is_complex,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        log(f"      OOM at {name} lr={lr} micro={micro} — skipping LR")
                        free_model(model)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise

                total_time += eval_t
                if acc > best_acc:
                    best_acc, best_lr = acc, lr
                free_model(model)

            key_name = f"{name}_dk{d_k_key}"
            results_matrix[d_k_key].setdefault(key_name, {})
            results_matrix[d_k_key][key_name][num_pairs] = {
                "best_acc": round(best_acc, 2),
                "best_lr": best_lr,
                "total_eval_time": round(total_time, 2),
                "micro_batch": micro,
                "scan_chunk": scan_chunk,
            }
            log(
                f"      [{name:28s} | d_k={d_k_key}] Pairs={num_pairs:3d} "
                f"(L={seq_len:4d}) -> Best Acc: {best_acc:6.2f}% (lr={best_lr})"
            )

with open(output_path, "w") as f:
    json.dump({"config": CFG, "results": results_matrix}, f, indent=2)
log(f"\nBENCHMARK COMPLETE! Results saved to {output_path}")