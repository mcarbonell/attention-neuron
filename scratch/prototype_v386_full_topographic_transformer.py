"""
Prototype v386: Full Topographic Transformer (Attention + FFN) in Autoregressive Language Modeling
Hypothesis:
  H1: Extending 2D Dirichlet Topographic Cellular Regularization (TCR) to both Attention
      (W_q, W_k, W_v, W_o) AND Feed-Forward projections (W_in, W_out) induces macroscopic
      cortical smoothness across 100% of the Transformer's linear weight layers (AdjCos > 0.85).
  H2: Joint attention-FFN spatial continuity maintains stable autoregressive training,
      preserving competitive validation perplexity on Tiny Shakespeare.
  H3: A Full Topographic Transformer enables global 2D-DCT parametric compression
      (pruning both attention routing AND feed-forward memories at 4x and 10x compression)
      preventing the catastrophic perplexity collapse observed in standard Transformers.

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
    return D.T  # (N, N) where D[k, :] is k-th basis vector

# ---------------------------------------------------------
# Topographic Metrics & Regularizers
# ---------------------------------------------------------
def dirichlet_energy_2d(W):
    """Vectorized 2D Dirichlet coupling energy (finite difference sum of squares)."""
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

def prune_dct_2d_general(W, sparsity, D_out, D_in):
    """
    Orthonormal 2D-DCT magnitude pruning of a general (M, N) matrix W.
    W_dct = D_out @ W @ D_in.T
    Keep top (1 - sparsity) largest magnitude coefficients.
    Reconstruct: D_out.T @ W_dct_sparse @ D_in
    """
    if sparsity <= 0.0:
        return W.clone()
    W_dct = D_out @ W @ D_in.T
    k_keep = max(1, int(W.numel() * (1.0 - sparsity)))
    threshold = torch.kthvalue(W_dct.abs().flatten(), W.numel() - k_keep + 1).values
    mask = (W_dct.abs() >= threshold).float()
    W_dct_sparse = W_dct * mask
    return D_out.T @ W_dct_sparse @ D_in

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

        # Independent projection matrices
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
    """Feed-Forward Network with independent linear projections for TCR coupling."""
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

def evaluate_global_dct_pruning_sweep(model, dataset, sparsity_levels, D_128, D_256, num_batches=15, batch_size=32, device="cpu"):
    """
    Evaluates validation loss & PPL under 2D-DCT magnitude pruning on ALL 12 linear matrices
    (both Attention and FFN projections across all layers).
    Restores original weights after evaluation.
    """
    results = {}
    orig_weights = {}

    target_projections = [".q.weight", ".k.weight", ".v.weight", ".o.weight", ".w_in.weight", ".w_out.weight"]
    for name, param in model.named_parameters():
        if any(proj in name for proj in target_projections):
            orig_weights[name] = param.data.clone()

    for s in sparsity_levels:
        with torch.no_grad():
            for name, w_orig in orig_weights.items():
                if w_orig.shape == (128, 128):
                    pruned = prune_dct_2d_general(w_orig, s, D_128, D_128)
                elif w_orig.shape == (256, 128):
                    pruned = prune_dct_2d_general(w_orig, s, D_256, D_128)
                elif w_orig.shape == (128, 256):
                    pruned = prune_dct_2d_general(w_orig, s, D_128, D_256)
                else:
                    pruned = w_orig.clone()
                dict(model.named_parameters())[name].data.copy_(pruned)

        val_loss, val_se, val_ppl, _ = evaluate_language_model(
            model, dataset, num_batches=num_batches, batch_size=batch_size, device=device
        )
        results[s] = {
            "val_loss": val_loss,
            "val_se": val_se,
            "val_ppl": val_ppl
        }

    # Restore exact unpruned weights
    with torch.no_grad():
        for name, w_orig in orig_weights.items():
            dict(model.named_parameters())[name].data.copy_(w_orig)

    return results

# ---------------------------------------------------------
# Training Function for One Condition / Seed
# ---------------------------------------------------------
def train_condition(cond_cfg, seed, dataset, D_128, D_256, max_steps=600, lr=2e-3, batch_size=32, device="cpu"):
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

    log(f"  Start Training [{cond_cfg['name']}, Seed {seed}]: Params Total={total_params:,}, LinearLayers={target_params:,}, eps_attn={eps_attn}, eps_ffn={eps_ffn}")

    t0 = time.time()
    step_history = []

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
        if step <= 5 and seed == 42 and "eps2e-3" in cond_cfg["name"]:
            log(f"  [Fast Feedback Step {step}/5] TaskLoss = {task_loss.item():.4f} | RegLoss = {float(reg_loss):.4f}")

        # Periodic logging every 150 steps
        if step % 150 == 0 or step == max_steps:
            val_loss, val_se, val_ppl, n_seq = evaluate_language_model(
                model, dataset, num_batches=15, batch_size=batch_size, device=device
            )
            adj_attn = model.get_mean_attn_adj_cos()
            adj_ffn = model.get_mean_ffn_adj_cos()
            st_s = step / (time.time() - t0)
            eta_s = (max_steps - step) / st_s if st_s > 0 else 0
            log(f"  Step {step:3d}/{max_steps} | Val Loss: {val_loss:.4f} (SE: {val_se:.4f}) | Val PPL: {val_ppl:5.2f} | AdjCos Attn={adj_attn:.3f} FFN={adj_ffn:.3f} | {st_s:.1f} st/s (ETA: {eta_s:.1f}s)")
            step_history.append({
                "step": step,
                "val_loss": val_loss,
                "val_se": val_se,
                "val_ppl": val_ppl,
                "attn_adj_cos": adj_attn,
                "ffn_adj_cos": adj_ffn
            })

    wall_time = time.time() - t0
    final_adj_attn = model.get_mean_attn_adj_cos()
    final_adj_ffn = model.get_mean_ffn_adj_cos()
    final_val_loss, final_val_se, final_val_ppl, _ = evaluate_language_model(
        model, dataset, num_batches=20, batch_size=batch_size, device=device
    )

    # Global 2D-DCT Sparsity sweep evaluation (100% of linear layers)
    sparsity_levels = [0.0, 0.5, 0.75, 0.90]
    log(f"  Evaluating Global 2D-DCT magnitude pruning across ALL linear projections (0%, 50%, 75%, 90%)...")
    pruning_res = evaluate_global_dct_pruning_sweep(
        model, dataset, sparsity_levels, D_128, D_256, num_batches=15, batch_size=batch_size, device=device
    )

    # Sample matrices for visualization
    sample_wq = model.blocks[0].attn.q.weight.detach().cpu().numpy()
    sample_win = model.blocks[0].ffn.w_in.weight.detach().cpu().numpy()

    return {
        "final_val_loss": final_val_loss,
        "final_val_se": final_val_se,
        "final_val_ppl": final_val_ppl,
        "final_adj_attn": final_adj_attn,
        "final_adj_ffn": final_adj_ffn,
        "wall_time": wall_time,
        "step_history": step_history,
        "pruning_res": pruning_res,
        "sample_wq": sample_wq,
        "sample_win": sample_win
    }

# ---------------------------------------------------------
# Main Suite
# ---------------------------------------------------------
def main():
    log("=========================================================================================")
    log(" EXPERIMENT v386: Full Topographic Transformer (Attention + FFN)")
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

    # -----------------------------------------------------
    # Conditions definition (Regla de Oro: Primero el Candidato)
    # -----------------------------------------------------
    conditions = [
        # Candidate 1: Full Topographic (Attn + FFN) with eps = 2e-3
        {
            "name": "Full_Topographic_eps2e-3",
            "eps_attn": 0.002,
            "eps_ffn": 0.002,
            "desc": "Full Topographic Transformer (Attn eps=2e-3, FFN eps=2e-3)"
        },
        # Candidate 2: Full Topographic (Attn + FFN) with eps = 5e-4
        {
            "name": "Full_Topographic_eps5e-4",
            "eps_attn": 0.0005,
            "eps_ffn": 0.0005,
            "desc": "Full Topographic Transformer (Attn eps=5e-4, FFN eps=5e-4)"
        },
        # Candidate 3: Attn-Only Topographic (Ablation control from v385)
        {
            "name": "AttnOnly_Topographic_eps2e-3",
            "eps_attn": 0.002,
            "eps_ffn": 0.0,
            "desc": "Attn-Only Topographic Transformer (Attn eps=2e-3, FFN unregularized)"
        },
        # Baseline 1: Standard AdamW (eps = 0.0)
        {
            "name": "Baseline_Standard_AdamW",
            "eps_attn": 0.0,
            "eps_ffn": 0.0,
            "desc": "Standard Transformer (eps=0.0, uncoupled attention and FFN)"
        }
    ]

    log(f"Total Conditions: {len(conditions)} | Seeds per Condition: {len(SEEDS)} | Total Runs: {len(conditions) * len(SEEDS)}")
    log("Architecture Inventory:")
    log("  FullTopographicLM (2 layers, d_model=128, n_heads=4, head_dim=32, ffn_dim=256, seq_len=128)")
    log("  Attn Projections: W_q, W_k, W_v, W_o (128x128 each) -> 131,072 params (50.0% of linear layers)")
    log("  FFN Projections: W_in (256x128), W_out (128x256) -> 131,072 params (50.0% of linear layers)")
    log("  Total Linear Layer Parameters subjected to TCR & DCT pruning: 262,144 params (90.7% of total model)")

    all_results = []
    total_runs = len(conditions) * len(SEEDS)
    current_run = 0

    sample_heatmaps = {}

    for cond in conditions:
        cond_name = cond["name"]
        log(f"\n>>> Running Condition: {cond_name} ({cond['desc']}) <<<")
        seed_runs = []

        for seed in SEEDS:
            current_run += 1
            progress_pct = (current_run / total_runs) * 100.0
            log(f"[Progress: {progress_pct:.1f}% | Run {current_run}/{total_runs}] Condition: {cond_name} | Seed: {seed}")

            res = train_condition(
                cond_cfg=cond,
                seed=seed,
                dataset=dataset,
                D_128=D_128,
                D_256=D_256,
                max_steps=MAX_STEPS,
                lr=LR,
                batch_size=BATCH_SIZE,
                device="cpu"
            )
            res["seed"] = seed
            res["cond_name"] = cond_name
            seed_runs.append(res)

            if cond_name not in sample_heatmaps:
                sample_heatmaps[cond_name] = {
                    "wq": res["sample_wq"],
                    "win": res["sample_win"]
                }

        all_results.append({
            "cond_name": cond_name,
            "cond_cfg": cond,
            "runs": seed_runs
        })

    # -----------------------------------------------------
    # Phase 2: Consolidation & Table
    # -----------------------------------------------------
    log("\n=========================================================================================")
    log(" PHASE 2: CONSOLIDATED RESULTS (Global Perplexity, AdjCos & 100% DCT Pruning)")
    log("=========================================================================================")

    summary_rows = []
    log(f"{'Condition':<28} | {'Val Loss':<14} | {'Val PPL':<11} | {'Attn AdjCos':<11} | {'FFN AdjCos':<11} | {'DCT 50% PPL':<11} | {'DCT 75% PPL':<11} | {'DCT 90% PPL':<11}")
    log("-" * 125)

    for cond_res in all_results:
        cname = cond_res["cond_name"]
        runs = cond_res["runs"]
        val_losses = [r["final_val_loss"] for r in runs]
        val_ppls = [r["final_val_ppl"] for r in runs]
        val_ses = [r["final_val_se"] for r in runs]
        adj_attns = [r["final_adj_attn"] for r in runs]
        adj_ffns = [r["final_adj_ffn"] for r in runs]

        m_loss = float(np.mean(val_losses))
        sd_loss = float(np.std(val_losses))
        m_ppl = float(np.mean(val_ppls))
        sd_ppl = float(np.std(val_ppls))
        m_se = float(np.mean(val_ses))
        m_adj_attn = float(np.mean(adj_attns))
        sd_adj_attn = float(np.std(adj_attns))
        m_adj_ffn = float(np.mean(adj_ffns))
        sd_adj_ffn = float(np.std(adj_ffns))

        # DCT pruning averages across seeds
        dct_ppl_50 = float(np.mean([r["pruning_res"][0.5]["val_ppl"] for r in runs]))
        dct_ppl_75 = float(np.mean([r["pruning_res"][0.75]["val_ppl"] for r in runs]))
        dct_ppl_90 = float(np.mean([r["pruning_res"][0.90]["val_ppl"] for r in runs]))

        summary_rows.append({
            "cond_name": cname,
            "mean_loss": m_loss,
            "std_loss": sd_loss,
            "mean_se": m_se,
            "mean_ppl": m_ppl,
            "std_ppl": sd_ppl,
            "mean_adj_attn": m_adj_attn,
            "std_adj_attn": sd_adj_attn,
            "mean_adj_ffn": m_adj_ffn,
            "std_adj_ffn": sd_adj_ffn,
            "dct_ppl_0": m_ppl,
            "dct_ppl_50": dct_ppl_50,
            "dct_ppl_75": dct_ppl_75,
            "dct_ppl_90": dct_ppl_90
        })

        log(f"{cname:<28} | {m_loss:5.3f} +/- {sd_loss:5.3f} | {m_ppl:5.2f} +/- {sd_ppl:4.2f} | {m_adj_attn:5.4f}     | {m_adj_ffn:5.4f}     | {dct_ppl_50:10.2f} | {dct_ppl_75:10.2f} | {dct_ppl_90:10.2f}")

    # -----------------------------------------------------
    # Phase 3: Artifact Generation (Figures & JSON)
    # -----------------------------------------------------
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    json_path = "results/raw/v386_full_topographic_transformer.json"
    raw_payload = {
        "experiment_id": "v386",
        "date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "commit_hash": get_git_commit(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": "cpu",
        "hyperparameters": {
            "max_steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "d_model": D_MODEL,
            "ffn_dim": FFN_DIM,
            "n_heads": 4,
            "n_layers": 2,
            "seq_len": 128,
            "dataset": "Tiny Shakespeare (char-level)",
            "seeds": SEEDS
        },
        "summary": summary_rows,
        "detailed_results": [
            {
                "cond_name": cr["cond_name"],
                "runs": [
                    {
                        "seed": r["seed"],
                        "final_val_loss": r["final_val_loss"],
                        "final_val_se": r["final_val_se"],
                        "final_val_ppl": r["final_val_ppl"],
                        "final_adj_attn": r["final_adj_attn"],
                        "final_adj_ffn": r["final_adj_ffn"],
                        "wall_time": r["wall_time"],
                        "pruning_res": r["pruning_res"]
                    }
                    for r in cr["runs"]
                ]
            }
            for cr in all_results
        ]
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_payload, f, indent=2)
    log(f"\nRaw results saved to {json_path}")

    # Generate Figures
    fig = plt.figure(figsize=(16, 5))

    # Subplot 1: Global Perplexity vs DCT Sparsity
    ax1 = fig.add_subplot(1, 3, 1)
    sparsities = [0.0, 50.0, 75.0, 90.0]
    colors = {
        "Full_Topographic_eps2e-3": "#2ecc71",
        "Full_Topographic_eps5e-4": "#3498db",
        "AttnOnly_Topographic_eps2e-3": "#9b59b6",
        "Baseline_Standard_AdamW": "#e74c3c"
    }

    for s in summary_rows:
        cname = s["cond_name"]
        ppls = [s["dct_ppl_0"], s["dct_ppl_50"], s["dct_ppl_75"], s["dct_ppl_90"]]
        ax1.plot(sparsities, ppls, marker="o", label=cname.replace("_", " "),
                 color=colors.get(cname, "#34495e"), linewidth=2.0)

    ax1.set_title("Global 2D-DCT Pruning (Attn + FFN) vs Val PPL", fontsize=11, weight="bold")
    ax1.set_xlabel("Global Linear Layer Sparsity (%)", fontsize=10)
    ax1.set_ylabel("Validation Perplexity (lower is better)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8)

    # Subplot 2: Adjacent Cosine Similarity in Attention vs FFN
    ax2 = fig.add_subplot(1, 3, 2)
    cnames = [s["cond_name"].replace("_Topographic_", " ").replace("Baseline_", "") for s in summary_rows]
    x_pos = np.arange(len(cnames))
    width = 0.35

    attn_vals = [s["mean_adj_attn"] for s in summary_rows]
    ffn_vals = [s["mean_adj_ffn"] for s in summary_rows]

    ax2.bar(x_pos - width/2, attn_vals, width, label="Attention AdjCos", color="#3498db", edgecolor="black", alpha=0.85)
    ax2.bar(x_pos + width/2, ffn_vals, width, label="FFN AdjCos", color="#e67e22", edgecolor="black", alpha=0.85)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(cnames, rotation=15, ha="right", fontsize=8)
    ax2.set_ylabel("Adjacent Cosine Similarity", fontsize=10)
    ax2.set_title("Cortical Smoothness: Attention vs FFN", fontsize=11, weight="bold")
    ax2.set_ylim(-0.05, 1.0)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8)

    # Subplot 3: Heatmap of FFN Weight (W_in, 256x128)
    ax3 = fig.add_subplot(1, 3, 3)
    full_win = sample_heatmaps.get("Full_Topographic_eps2e-3", {}).get("win", np.zeros((256, 128)))
    base_win = sample_heatmaps.get("Baseline_Standard_AdamW", {}).get("win", np.zeros((256, 128)))
    composite = np.vstack([full_win[:64, :64], base_win[:64, :64]])
    im = ax3.imshow(composite, cmap="magma", aspect="auto")
    ax3.axhline(64, color="white", linewidth=2.0, linestyle="--")
    ax3.set_title("FFN Receptive Fields (Top: Full Topo | Bottom: Base)", fontsize=10, weight="bold")
    ax3.set_xlabel("Receptive Dimension", fontsize=9)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_path = "results/figures/v386_full_topographic_transformer_curves.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    log(f"Figure saved to {fig_path}")

    log("\n=========================================================================================")
    log(" EXPERIMENT v386 FINISHED SUCCESSFULLY")
    log("=========================================================================================")

if __name__ == "__main__":
    main()
