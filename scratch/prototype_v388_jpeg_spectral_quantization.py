"""
Prototype v388: Adaptive Spectral JPEG-style Quantization in Autoregressive Transformers
Hypothesis:
  H1: Because Topographic Regularization (TCR, Dirichlet coupling) concentrates weight energy
      into the lowest spatial frequencies (DC and fundamental harmonics of 2D-DCT),
      allocating variable bit-width according to radial frequency bands
      (e.g., 8-bit for DC/low, 4-bit for mid, 2-bit for high-mid, 0-bit/truncated for ultra-high)
      achieves sub-2-bit average precision (~1.68 bits/parameter, ~19x compression vs FP32)
      with minimal increase in validation perplexity.
  H2: On a standard Transformer (AdamW, unregularized), where weight energy is distributed
      uniformly across high frequencies, the same JPEG-style spectral quantization induces
      catastrophic information destruction and severe perplexity degradation.
  H3: Compared to spatial uniform quantization (standard INT4 and INT2), JPEG-style adaptive
      spectral quantization achieves a superior rate-distortion frontier on Topographic Transformers,
      outperforming 2-bit spatial quantization while using fewer bits per parameter (1.68 vs 2.0 bpp).

Rigour Level: Level 1 (Exploratory multi-seed Transformer benchmark, 3 seeds per condition)
"""

import sys
import os
import time
import math
import json
import platform
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Logging and Timing Contract
# ---------------------------------------------------------
START_TIME = time.time()

def log(msg):
    elapsed = time.time() - START_TIME
    mins = int(elapsed // 60)
    secs = elapsed % 60
    print(f"[+{mins:02d}:{secs:05.2f}] {msg}", flush=True)

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

# ---------------------------------------------------------
# Orthonormal DCT-II Basis Generator
# ---------------------------------------------------------
def get_dct_basis(N, device="cpu"):
    n = torch.arange(N, device=device).unsqueeze(1)
    k = torch.arange(N, device=device).unsqueeze(0)
    D = torch.cos(math.pi * (n + 0.5) * k / N)
    D[:, 0] *= 1.0 / math.sqrt(N)
    D[:, 1:] *= math.sqrt(2.0 / N)
    return D.T  # (N, N) orthonormal basis matrix

# ---------------------------------------------------------
# Topographic Metrics & Regularizers
# ---------------------------------------------------------
def dirichlet_energy_2d(W):
    """Vectorized 2D Dirichlet coupling energy."""
    diff_i = W[1:, :] - W[:-1, :]
    diff_j = W[:, 1:] - W[:, :-1]
    return torch.sum(diff_i ** 2) + torch.sum(diff_j ** 2)

def adjacent_cosine_similarity_2d(W):
    """Adjacent cosine similarity along rows and columns."""
    r1, r2 = W[:-1, :], W[1:, :]
    cos_rows = F.cosine_similarity(r1, r2, dim=-1).mean().item()
    c1, c2 = W[:, :-1], W[:, 1:]
    cos_cols = F.cosine_similarity(c1, c2, dim=0).mean().item()
    return 0.5 * (cos_rows + cos_cols)

# ---------------------------------------------------------
# Quantization Operators (Spatial Uniform & Spectral JPEG)
# ---------------------------------------------------------
def quantize_spatial_uniform(W, bits):
    """
    Symmetric uniform per-tensor quantization directly in spatial coordinates.
    For bits=4: 15 levels [-7, 7]. For bits=2: 3 levels [-1, 1] (ternary).
    """
    if bits == 0:
        return torch.zeros_like(W)
    qmax = (2 ** (bits - 1)) - 1
    s = W.abs().max()
    if s < 1e-8:
        return torch.zeros_like(W)
    q = torch.clamp(torch.round((W / s) * qmax), -qmax, qmax)
    return q * (s / qmax)

def quantize_spectral_uniform(W, D_out, D_in, bits):
    """Symmetric uniform per-tensor quantization in 2D-DCT domain."""
    W_dct = D_out @ W @ D_in.T
    W_dct_q = quantize_spatial_uniform(W_dct, bits)
    return D_out.T @ W_dct_q @ D_in

def quantize_spectral_jpeg(W, D_out, D_in, b_edges, bits):
    """
    JPEG-style adaptive spectral quantization:
    Transforms W to 2D-DCT: W_dct = D_out @ W @ D_in.T
    Partitions coefficients into concentric radial frequency bands:
      rho(u, v) = sqrt((u/M)^2 + (v/N)^2) / sqrt(2) in [0, 1]
    Each band k is quantized with symmetric uniform bitdepth bits[k].
    Coefficients beyond the last edge are truncated to 0 bits (zeroed out).
    Reconstructs: W_recon = D_out.T @ W_dct_q @ D_in
    """
    M, N = W.shape
    u = torch.arange(M, device=W.device, dtype=W.dtype).unsqueeze(1) / M
    v = torch.arange(N, device=W.device, dtype=W.dtype).unsqueeze(0) / N
    rho = torch.sqrt(u**2 + v**2) / math.sqrt(2.0)

    W_dct = D_out @ W @ D_in.T
    W_dct_q = torch.zeros_like(W_dct)

    prev_edge = 0.0
    for edge, b in zip(b_edges, bits):
        mask = (rho >= prev_edge) & (rho < edge)
        if mask.any() and b > 0:
            x = W_dct[mask]
            s = x.abs().max()
            if s > 1e-8:
                qmax = (2 ** (b - 1)) - 1
                q = torch.clamp(torch.round((x / s) * qmax), -qmax, qmax)
                W_dct_q[mask] = q * (s / qmax)
        prev_edge = edge

    return D_out.T @ W_dct_q @ D_in

def compute_profile_bpp(M, N, b_edges, bits):
    """Calculates exact average bits per parameter (bpp) for a given matrix size and JPEG profile."""
    u = np.arange(M)[:, None] / M
    v = np.arange(N)[None, :] / N
    rho = np.sqrt(u**2 + v**2) / np.sqrt(2.0)

    total_bits = 0.0
    prev_edge = 0.0
    for edge, b in zip(b_edges, bits):
        mask = (rho >= prev_edge) & (rho < edge)
        total_bits += mask.mean() * b
        prev_edge = edge
    # Beyond last edge is 0 bits
    return float(total_bits)

# ---------------------------------------------------------
# Transformer Architecture with Topographic Hooks
# ---------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4, max_len=128):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1),
            persistent=False
        )

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o(y)

    def get_dirichlet_energy(self):
        return (dirichlet_energy_2d(self.q.weight) +
                dirichlet_energy_2d(self.k.weight) +
                dirichlet_energy_2d(self.v.weight) +
                dirichlet_energy_2d(self.o.weight))

    def get_mean_adj_cos(self):
        return (adjacent_cosine_similarity_2d(self.q.weight) +
                adjacent_cosine_similarity_2d(self.k.weight) +
                adjacent_cosine_similarity_2d(self.v.weight) +
                adjacent_cosine_similarity_2d(self.o.weight)) / 4.0


class TopographicFFN(nn.Module):
    def __init__(self, d_model=128, ffn_dim=256):
        super().__init__()
        self.w_in = nn.Linear(d_model, ffn_dim, bias=False)
        self.w_out = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x):
        return self.w_out(F.gelu(self.w_in(x)))

    def get_dirichlet_energy(self):
        return dirichlet_energy_2d(self.w_in.weight) + dirichlet_energy_2d(self.w_out.weight)

    def get_mean_adj_cos(self):
        return (adjacent_cosine_similarity_2d(self.w_in.weight) +
                adjacent_cosine_similarity_2d(self.w_out.weight)) / 2.0


class FullTransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, max_len=128, ffn_dim=256):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = TopographicFFN(d_model, ffn_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class FullTopographicLM(nn.Module):
    def __init__(self, vocab_size=65, d_model=128, n_heads=4, n_layers=2, max_len=128, ffn_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        self.blocks = nn.ModuleList([
            FullTransformerBlock(d_model, n_heads, max_len, ffn_dim)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def get_attn_dirichlet_energy(self):
        return sum(block.attn.get_dirichlet_energy() for block in self.blocks)

    def get_ffn_dirichlet_energy(self):
        return sum(block.ffn.get_dirichlet_energy() for block in self.blocks)

    def get_mean_attn_adj_cos(self):
        return float(np.mean([block.attn.get_mean_adj_cos() for block in self.blocks]))

    def get_mean_ffn_adj_cos(self):
        return float(np.mean([block.ffn.get_mean_adj_cos() for block in self.blocks]))

# ---------------------------------------------------------
# Dataset: Tiny Shakespeare Loader
# ---------------------------------------------------------
class TinyShakespeareDataset:
    def __init__(self, data_path="data/tinyshakespeare.txt", seq_len=128):
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n_train = int(len(data) * 0.9)
        self.train_data = data[:n_train]
        self.val_data = data[n_train:]
        self.seq_len = seq_len

    def get_batch(self, split="train", batch_size=32, device="cpu"):
        d = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(d) - self.seq_len, (batch_size,))
        x = torch.stack([d[i:i + self.seq_len] for i in ix]).to(device)
        y = torch.stack([d[i + 1:i + 1 + self.seq_len] for i in ix]).to(device)
        return x, y

# ---------------------------------------------------------
# Evaluation Helpers (SE across Independent Sequences)
# ---------------------------------------------------------
def evaluate_language_model(model, dataset, num_batches=20, batch_size=32, device="cpu"):
    """
    Evaluates Validation Loss and Perplexity on independent sequences.
    Calculates Standard Error (SE) per sequence according to GEMINI.md contract.
    """
    model.eval()
    all_seq_losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = dataset.get_batch(split="val", batch_size=batch_size, device=device)
            logits = model(x)
            loss_token = F.cross_entropy(
                logits.view(-1, model.vocab_size), y.view(-1), reduction="none"
            ).view(x.shape[0], x.shape[1])
            seq_loss = loss_token.mean(dim=1)
            all_seq_losses.extend(seq_loss.cpu().tolist())

    seq_losses_arr = np.array(all_seq_losses)
    mean_loss = float(np.mean(seq_losses_arr))
    n_seq = len(seq_losses_arr)
    se_loss = float(np.std(seq_losses_arr) / math.sqrt(n_seq))
    ppl = float(math.exp(mean_loss))
    return mean_loss, se_loss, ppl, n_seq

# ---------------------------------------------------------
# Model Quantization Sweep Evaluator
# ---------------------------------------------------------
def evaluate_quantization_modes(model, dataset, quant_modes, D_128, D_256, num_batches=20, batch_size=32, device="cpu"):
    """
    Evaluates model across multiple quantization modes applied to all 12 linear projection matrices.
    Computes validation loss, SE, PPL, and average weight reconstruction relative error.
    Restores original weights after each evaluation.
    """
    target_projections = [".q.weight", ".k.weight", ".v.weight", ".o.weight", ".w_in.weight", ".w_out.weight"]
    orig_weights = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if any(proj in name for proj in target_projections)
    }

    results = {}
    for mode in quant_modes:
        mode_id = mode["id"]
        mode_type = mode["type"]
        rel_errors = []

        with torch.no_grad():
            for name, w_orig in orig_weights.items():
                if w_orig.shape == (128, 128):
                    d_out, d_in = D_128, D_128
                elif w_orig.shape == (256, 128):
                    d_out, d_in = D_256, D_128
                elif w_orig.shape == (128, 256):
                    d_out, d_in = D_128, D_256
                else:
                    d_out, d_in = None, None

                if mode_type == "fp32":
                    w_q = w_orig.clone()
                elif mode_type == "spatial_uniform":
                    w_q = quantize_spatial_uniform(w_orig, mode["bits"])
                elif mode_type == "spectral_uniform":
                    w_q = quantize_spectral_uniform(w_orig, d_out, d_in, mode["bits"])
                elif mode_type == "spectral_jpeg":
                    w_q = quantize_spectral_jpeg(w_orig, d_out, d_in, mode["b_edges"], mode["bits"])
                else:
                    raise ValueError(f"Unknown mode type: {mode_type}")

                # Reconstruction relative error
                rel_err = float((w_orig - w_q).norm() / (w_orig.norm() + 1e-8))
                rel_errors.append(rel_err)
                dict(model.named_parameters())[name].data.copy_(w_q)

        val_loss, val_se, val_ppl, n_seq = evaluate_language_model(
            model, dataset, num_batches=num_batches, batch_size=batch_size, device=device
        )
        mean_rel_err = float(np.mean(rel_errors))

        results[mode_id] = {
            "mode_name": mode["name"],
            "bpp": mode["bpp"],
            "compression_ratio": 32.0 / mode["bpp"],
            "val_loss": val_loss,
            "val_se": val_se,
            "val_ppl": val_ppl,
            "n_seq": n_seq,
            "mean_rel_err": mean_rel_err
        }

        # Restore original weights immediately
        with torch.no_grad():
            for name, w_orig in orig_weights.items():
                dict(model.named_parameters())[name].data.copy_(w_orig)

    return results

# ---------------------------------------------------------
# Training Function for Base Transformer
# ---------------------------------------------------------
def train_base_model(cond_cfg, seed, dataset, max_steps=600, lr=2e-3, batch_size=32, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FullTopographicLM(
        vocab_size=dataset.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_len=128,
        ffn_dim=256
    ).to(device)

    eps_attn = cond_cfg["eps_attn"]
    eps_ffn = cond_cfg["eps_ffn"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_params = sum(p.numel() for p in model.parameters())
    target_params = sum(p.numel() for n, p in model.named_parameters() if any(
        proj in n for proj in [".q.weight", ".k.weight", ".v.weight", ".o.weight", ".w_in.weight", ".w_out.weight"]
    ))

    log(f"  Training [{cond_cfg['name']}, Seed {seed}]: Total={total_params:,}, LinearLayers={target_params:,}, eps_attn={eps_attn}, eps_ffn={eps_ffn}")

    t0 = time.time()
    for step in range(1, max_steps + 1):
        model.train()
        x, y = dataset.get_batch(split="train", batch_size=batch_size, device=device)
        optimizer.zero_grad()
        logits = model(x)
        task_loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

        reg_loss = 0.0
        if eps_attn > 0.0:
            reg_loss = reg_loss + eps_attn * model.get_attn_dirichlet_energy()
        if eps_ffn > 0.0:
            reg_loss = reg_loss + eps_ffn * model.get_ffn_dirichlet_energy()

        total_loss = task_loss + reg_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Fast feedback on first 5 steps of Run 1
        if step <= 5 and seed == 42 and "Topographic" in cond_cfg["name"]:
            log(f"  [Fast Feedback Step {step}/5] TaskLoss = {task_loss.item():.4f} | RegLoss = {float(reg_loss):.4f}")

        if step % 200 == 0 or step == max_steps:
            val_loss, val_se, val_ppl, _ = evaluate_language_model(
                model, dataset, num_batches=15, batch_size=batch_size, device=device
            )
            adj_attn = model.get_mean_attn_adj_cos()
            adj_ffn = model.get_mean_ffn_adj_cos()
            st_s = step / (time.time() - t0)
            log(f"  Step {step:3d}/{max_steps} | Val Loss: {val_loss:.4f} (SE: {val_se:.4f}) | Val PPL: {val_ppl:5.2f} | AdjCos Attn={adj_attn:.3f} FFN={adj_ffn:.3f} | {st_s:.1f} st/s")

    wall_time = time.time() - t0
    final_adj_attn = model.get_mean_attn_adj_cos()
    final_adj_ffn = model.get_mean_ffn_adj_cos()

    return model, {
        "wall_time": wall_time,
        "final_adj_attn": final_adj_attn,
        "final_adj_ffn": final_adj_ffn
    }

# ---------------------------------------------------------
# Main Suite Execution
# ---------------------------------------------------------
def main():
    log("=========================================================================================")
    log(" EXPERIMENT v388: Adaptive Spectral JPEG-style Quantization in Transformers")
    log("=========================================================================================")
    log(f"Commit: {get_git_commit()} | Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    log(f"Platform: {platform.platform()} | Device: CPU")

    dataset_path = "data/tinyshakespeare.txt"
    if not os.path.exists(dataset_path):
        log(f"ERROR: Dataset not found at {dataset_path}!")
        sys.exit(1)

    dataset = TinyShakespeareDataset(data_path=dataset_path, seq_len=128)
    log(f"Dataset Loaded: Tiny Shakespeare | Vocab Size: {dataset.vocab_size} | Total Chars: {len(dataset.train_data) + len(dataset.val_data):,}")

    SEEDS = [42, 100, 2026]
    MAX_STEPS = 600
    BATCH_SIZE = 32
    LR = 2e-3
    D_MODEL = 128
    FFN_DIM = 256

    D_128 = get_dct_basis(D_MODEL, device="cpu")
    D_256 = get_dct_basis(FFN_DIM, device="cpu")

    # Calculate exact weighted bpp across 12 linear projection matrices
    # (8 matrices of 128x128 = 131,072 params, 4 matrices of 256x128 / 128x256 = 131,072 params)
    def calc_weighted_bpp(b_edges, bits):
        bpp_128 = compute_profile_bpp(128, 128, b_edges, bits)
        bpp_256 = compute_profile_bpp(256, 128, b_edges, bits)
        return float(0.5 * bpp_128 + 0.5 * bpp_256)

    bpp_jpeg_adapt = calc_weighted_bpp([0.15, 0.35, 0.60], [8, 4, 2])
    bpp_jpeg_aggr = calc_weighted_bpp([0.10, 0.25, 0.45], [8, 4, 2])

    log(f"Profile JPEG-Adaptive Bitrate: {bpp_jpeg_adapt:.3f} bpp ({32.0/bpp_jpeg_adapt:.1f}x compression vs FP32)")
    log(f"Profile JPEG-Aggressive Bitrate: {bpp_jpeg_aggr:.3f} bpp ({32.0/bpp_jpeg_aggr:.1f}x compression vs FP32)")

    # Define Quantization Modes to Sweep
    quant_modes = [
        {
            "id": "fp32",
            "name": "FP32 (Unquantized)",
            "type": "fp32",
            "bpp": 32.0,
            "bits": 32
        },
        {
            "id": "jpeg_adaptive_1.7bpp",
            "name": "JPEG Adaptive (8b/4b/2b/0b)",
            "type": "spectral_jpeg",
            "bpp": bpp_jpeg_adapt,
            "b_edges": [0.15, 0.35, 0.60],
            "bits": [8, 4, 2]
        },
        {
            "id": "jpeg_aggressive_0.9bpp",
            "name": "JPEG Aggressive (8b/4b/2b/0b)",
            "type": "spectral_jpeg",
            "bpp": bpp_jpeg_aggr,
            "b_edges": [0.10, 0.25, 0.45],
            "bits": [8, 4, 2]
        },
        {
            "id": "spatial_int4",
            "name": "Spatial Uniform INT4",
            "type": "spatial_uniform",
            "bpp": 4.0,
            "bits": 4
        },
        {
            "id": "spatial_int2",
            "name": "Spatial Uniform INT2",
            "type": "spatial_uniform",
            "bpp": 2.0,
            "bits": 2
        },
        {
            "id": "spectral_uniform_int4",
            "name": "Spectral Uniform INT4",
            "type": "spectral_uniform",
            "bpp": 4.0,
            "bits": 4
        }
    ]

    # Models to train (Regla de Oro: Primero el Candidato)
    model_conditions = [
        {
            "name": "Topographic_Full_eps2e-3",
            "desc": "Full Topographic Transformer (Candidate)",
            "eps_attn": 2e-3,
            "eps_ffn": 2e-3
        },
        {
            "name": "Standard_AdamW_Baseline",
            "desc": "Standard Transformer Baseline (AdamW, eps=0)",
            "eps_attn": 0.0,
            "eps_ffn": 0.0
        }
    ]

    all_results = {}
    total_runs = len(model_conditions) * len(SEEDS)
    current_run = 0

    log(f"\nBeginning Suite: {len(model_conditions)} Models x {len(SEEDS)} Seeds = {total_runs} Base Trainings.")
    log(f"Each base model will be evaluated across {len(quant_modes)} quantization modes on 640 validation sequences.\n")

    suite_t0 = time.time()

    for cond in model_conditions:
        cond_name = cond["name"]
        all_results[cond_name] = {
            "config": cond,
            "runs": {},
            "quant_summary": {}
        }
        log(f"=========================================================================================")
        log(f" MODEL FAMILY: {cond['desc']}")
        log(f"=========================================================================================")

        for seed in SEEDS:
            current_run += 1
            run_t0 = time.time()
            log(f">>> Run [{current_run}/{total_runs}] Model: {cond_name} | Seed: {seed} <<<")

            model, train_stats = train_base_model(
                cond, seed, dataset, max_steps=MAX_STEPS, lr=LR, batch_size=BATCH_SIZE, device="cpu"
            )

            # Evaluate Quantization Modes on this trained model
            log(f"  Evaluating {len(quant_modes)} quantization modes on 640 validation sequences...")
            q_res = evaluate_quantization_modes(
                model, dataset, quant_modes, D_128, D_256, num_batches=20, batch_size=32, device="cpu"
            )

            for qid, qdata in q_res.items():
                log(f"    [{qdata['mode_name']:<28}] bpp={qdata['bpp']:<4.2f} ({qdata['compression_ratio']:<4.1f}x) | Val PPL = {qdata['val_ppl']:<5.2f} (SE: {qdata['val_se']:.4f}) | RelErr = {qdata['mean_rel_err']*100:<4.1f}%")

            all_results[cond_name]["runs"][seed] = {
                "train_stats": train_stats,
                "quant_results": q_res
            }

            elapsed_suite = time.time() - suite_t0
            avg_run_time = elapsed_suite / current_run
            remaining_runs = total_runs - current_run
            eta_suite = remaining_runs * avg_run_time
            log(f"  Run Complete in {time.time() - run_t0:.1f}s | Suite Progress: {current_run}/{total_runs} ({(current_run/total_runs)*100:.1f}%) | Suite ETA: {eta_suite:.1f}s\n")

    # Aggregate Statistics across Seeds for each Condition and Quantization Mode
    for cond in model_conditions:
        cond_name = cond["name"]
        summary = {}
        for qmode in quant_modes:
            qid = qmode["id"]
            ppls = [all_results[cond_name]["runs"][s]["quant_results"][qid]["val_ppl"] for s in SEEDS]
            losses = [all_results[cond_name]["runs"][s]["quant_results"][qid]["val_loss"] for s in SEEDS]
            ses = [all_results[cond_name]["runs"][s]["quant_results"][qid]["val_se"] for s in SEEDS]
            rel_errs = [all_results[cond_name]["runs"][s]["quant_results"][qid]["mean_rel_err"] for s in SEEDS]

            summary[qid] = {
                "mode_name": qmode["name"],
                "bpp": qmode["bpp"],
                "compression_ratio": 32.0 / qmode["bpp"],
                "mean_ppl": float(np.mean(ppls)),
                "std_ppl": float(np.std(ppls)),
                "mean_loss": float(np.mean(losses)),
                "std_loss": float(np.std(losses)),
                "mean_se": float(np.mean(ses)),
                "mean_rel_err": float(np.mean(rel_errs))
            }
        all_results[cond_name]["quant_summary"] = summary

    # ---------------------------------------------------------
    # Final Reporting & Comparison Table
    # ---------------------------------------------------------
    log("\n" + "=" * 115)
    log(" FINAL CONSOLIDATED SUMMARY: SPECTRAL JPEG vs SPATIAL QUANTIZATION")
    log("=" * 115)
    header = f"{'Quantization Mode':<28} | {'Bits/Param':<10} | {'Comp Ratio':<10} | {'Topographic PPL':<18} | {'Standard PPL':<18} | {'Topographic RelErr':<18}"
    log(header)
    log("-" * 115)

    topo_sum = all_results["Topographic_Full_eps2e-3"]["quant_summary"]
    std_sum = all_results["Standard_AdamW_Baseline"]["quant_summary"]

    for qmode in quant_modes:
        qid = qmode["id"]
        t_data = topo_sum[qid]
        s_data = std_sum[qid]
        t_ppl_str = f"{t_data['mean_ppl']:.2f} +/- {t_data['std_ppl']:.2f}"
        s_ppl_str = f"{s_data['mean_ppl']:.2f} +/- {s_data['std_ppl']:.2f}"
        t_err_str = f"{t_data['mean_rel_err']*100:.1f}%"

        line = f"{t_data['mode_name']:<28} | {t_data['bpp']:<10.2f} | {t_data['compression_ratio']:<10.1f}x | {t_ppl_str:<18} | {s_ppl_str:<18} | {t_err_str:<18}"
        log(line)

    log("-" * 115)

    # ---------------------------------------------------------
    # Visualization & Figures
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Panel 1: Rate-Distortion Curve (PPL vs bpp)
    ax1 = axes[0, 0]
    bpps_topo = [topo_sum[m["id"]]["bpp"] for m in quant_modes]
    ppls_topo = [topo_sum[m["id"]]["mean_ppl"] for m in quant_modes]
    errs_topo = [topo_sum[m["id"]]["std_ppl"] for m in quant_modes]

    bpps_std = [std_sum[m["id"]]["bpp"] for m in quant_modes]
    ppls_std = [min(std_sum[m["id"]]["mean_ppl"], 100.0) for m in quant_modes]  # Cap for plotting
    errs_std = [std_sum[m["id"]]["std_ppl"] for m in quant_modes]

    ax1.errorbar(bpps_topo, ppls_topo, yerr=errs_topo, fmt='o-', color='#1f77b4', lw=2.5, capsize=4, label='Topographic Transformer (Ours)')
    ax1.errorbar(bpps_std, ppls_std, yerr=errs_std, fmt='s--', color='#d62728', lw=2.0, capsize=4, label='Standard AdamW Transformer')

    for m in quant_modes:
        qid = m["id"]
        ax1.annotate(m["name"].split(" (")[0], (topo_sum[qid]["bpp"], topo_sum[qid]["mean_ppl"]),
                     textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8, color='#1f77b4')

    ax1.set_xlabel("Bits Per Parameter (bpp, Linear Weights)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Validation Perplexity (Tiny Shakespeare)", fontsize=11, fontweight="bold")
    ax1.set_title("Rate-Distortion Frontier: PPL vs Bit-Budget", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    # Panel 2: 2D-DCT Power Spectrum & JPEG Band Partition
    ax2 = axes[0, 1]
    # Recompute sample 2D-DCT spectrum from first block Q weight of trained Topographic model
    sample_w = model.blocks[0].attn.q.weight.detach().cpu()
    w_dct_sample = (D_128 @ sample_w @ D_128.T).abs().numpy()
    im = ax2.imshow(np.log10(w_dct_sample + 1e-6), cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax2, label="Log10 |DCT Amplitude|")

    # Draw radial contours for JPEG bands
    theta = np.linspace(0, np.pi/2, 100)
    for r_norm, col, lbl in [(0.15, "cyan", "8-bit (r<=0.15)"), (0.35, "yellow", "4-bit (r<=0.35)"), (0.60, "lime", "2-bit (r<=0.60)")]:
        r_pixel = r_norm * 128 * np.sqrt(2.0)
        ax2.plot(r_pixel * np.sin(theta), r_pixel * np.cos(theta), color=col, lw=2.0, label=lbl)

    ax2.set_xlim(0, 127)
    ax2.set_ylim(127, 0)
    ax2.set_title("2D-DCT Spectrum with JPEG Band Allocation", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Frequency Index u (Columns)", fontsize=10)
    ax2.set_ylabel("Frequency Index v (Rows)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)

    # Panel 3: Weight Reconstruction Relative Error (%)
    ax3 = axes[1, 0]
    modes_labels = [m["name"].split(" (")[0] for m in quant_modes if m["id"] != "fp32"]
    modes_keys = [m["id"] for m in quant_modes if m["id"] != "fp32"]

    x = np.arange(len(modes_labels))
    width = 0.35

    err_t = [topo_sum[k]["mean_rel_err"] * 100 for k in modes_keys]
    err_s = [std_sum[k]["mean_rel_err"] * 100 for k in modes_keys]

    ax3.bar(x - width/2, err_t, width, label='Topographic', color='#1f77b4', alpha=0.85)
    ax3.bar(x + width/2, err_s, width, label='Standard AdamW', color='#d62728', alpha=0.85)

    ax3.set_ylabel("Reconstruction Relative Error (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Weight Reconstruction Distortion Across Schemes", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(modes_labels, rotation=25, ha="right", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax3.legend(fontsize=10)

    # Panel 4: Physical Memory Footprint (Linear Weights, KB)
    ax4 = axes[1, 1]
    mem_labels = [m["name"].split(" (")[0] for m in quant_modes]
    mem_kb = [262144 * (m["bpp"] / 8.0) / 1024.0 for m in quant_modes]
    colors = ['#7f7f7f', '#2ca02c', '#17becf', '#ff7f0e', '#bcbd22', '#9467bd']

    bars = ax4.bar(mem_labels, mem_kb, color=colors, alpha=0.85)
    for bar, kb, m in zip(bars, mem_kb, quant_modes):
        ratio = 32.0 / m["bpp"]
        ax4.text(bar.get_x() + bar.get_width()/2, kb + 15, f"{kb:.1f} KB\n({ratio:.1f}x)",
                 ha='center', va='bottom', fontsize=8, fontweight="bold")

    ax4.set_ylabel("Physical Footprint (KB)", fontsize=11, fontweight="bold")
    ax4.set_title("Linear Projection Weight Footprint (262,144 params)", fontsize=12, fontweight="bold")
    ax4.set_xticks(range(len(mem_labels)))
    ax4.set_xticklabels(mem_labels, rotation=25, ha="right", fontsize=9)
    ax4.set_ylim(0, 1200)
    ax4.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    figure_path = "results/figures/v388_jpeg_spectral_quantization.png"
    plt.savefig(figure_path, dpi=200)
    plt.close()
    log(f"Figure saved to: {figure_path}")

    # ---------------------------------------------------------
    # Persist Structured JSON Results
    # ---------------------------------------------------------
    os.makedirs("results/raw", exist_ok=True)
    raw_path = "results/raw/v388_jpeg_spectral_quantization.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    log(f"Raw results saved to: {raw_path}")
    log(f"Experiment v388 execution completed successfully in {time.time() - START_TIME:.2f}s total.")


if __name__ == "__main__":
    main()
