"""
Prototype v385: Topographic Attention in Transformers (Autoregressive Language Modeling)
Hypothesis:
  H1: Applying 2D Dirichlet Topographic Cellular Regularization (TCR) to the attention projection
      matrices (W_q, W_k, W_v, W_o) breaks permutation symmetry in Transformer attention heads,
      inducing macroscopic spatial continuity (AdjCos > 0.85).
  H2: Spatial continuity in attention heads acts as a benign or regularizing inductive bias for
      autoregressive language modeling, maintaining competitive validation perplexity on Tiny Shakespeare.
  H3: Topographic attention projections exhibit dramatic spectral compressibility in the 2D-DCT domain,
      enabling 2x - 10x parametric pruning of LLM attention heads with significantly lower perplexity
      degradation than standard unstructured attention projections.

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

def prune_dct_2d(W, sparsity, D):
    """
    Orthonormal 2D-DCT magnitude pruning of a square matrix W.
    W_dct = D @ W @ D.T
    Keep top (1 - sparsity) largest magnitude coefficients.
    Reconstruct: D.T @ W_dct_sparse @ D
    """
    if sparsity <= 0.0:
        return W.clone()
    W_dct = D @ W @ D.T
    k_keep = max(1, int(W.numel() * (1.0 - sparsity)))
    threshold = torch.kthvalue(W_dct.abs().flatten(), W.numel() - k_keep + 1).values
    mask = (W_dct.abs() >= threshold).float()
    W_dct_sparse = W_dct * mask
    return D.T @ W_dct_sparse @ D

# ---------------------------------------------------------
# Transformer Architecture (NanoGPT style)
# ---------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4, max_len=128):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Independent projection matrices for clean topographic regularization
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
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, hs)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, H, T, hs)
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


class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, max_len=128, ffn_dim=256):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class NanoLanguageModel(nn.Module):
    def __init__(self, vocab_size=65, d_model=128, n_heads=4, n_layers=2, max_len=128, ffn_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_len, ffn_dim)
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
        total = 0.0
        for block in self.blocks:
            total = total + block.attn.get_dirichlet_energy()
        return total

    def get_mean_attn_adj_cos(self):
        values = [block.attn.get_mean_adj_cos() for block in self.blocks]
        return float(np.mean(values))

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
# Evaluation Helper with Standard Error (SE) across Sequences
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
            # Compute loss per token without reduction: (B, T)
            loss_token = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1), reduction="none").view(x.shape[0], x.shape[1])
            # Average loss per sequence: (B,)
            seq_loss = loss_token.mean(dim=1)
            all_seq_losses.extend(seq_loss.cpu().tolist())

    seq_losses_arr = np.array(all_seq_losses)
    mean_loss = float(np.mean(seq_losses_arr))
    # SE = std(loss_per_sequence) / sqrt(n_sequences)
    n_seq = len(seq_losses_arr)
    se_loss = float(np.std(seq_losses_arr) / math.sqrt(n_seq))
    ppl = float(math.exp(mean_loss))
    return mean_loss, se_loss, ppl, n_seq

def evaluate_dct_pruning_sweep(model, dataset, sparsity_levels, D, num_batches=15, batch_size=32, device="cpu"):
    """
    Evaluates validation loss & PPL under 2D-DCT magnitude pruning on all attention projections.
    Restores original weights after evaluation.
    """
    results = {}
    orig_weights = {}
    for name, param in model.named_parameters():
        if any(proj in name for proj in [".q.weight", ".k.weight", ".v.weight", ".o.weight"]):
            orig_weights[name] = param.data.clone()

    for s in sparsity_levels:
        with torch.no_grad():
            for name, w_orig in orig_weights.items():
                pruned = prune_dct_2d(w_orig, s, D)
                dict(model.named_parameters())[name].data.copy_(pruned)

        val_loss, val_se, val_ppl, _ = evaluate_language_model(model, dataset, num_batches=num_batches, batch_size=batch_size, device=device)
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
def train_condition(cond_cfg, seed, dataset, D_basis, max_steps=600, lr=2e-3, batch_size=32, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = NanoLanguageModel(
        vocab_size=dataset.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_len=128,
        ffn_dim=256
    ).to(device)

    epsilon = cond_cfg["epsilon"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    total_params = sum(p.numel() for p in model.parameters())
    attn_params = sum(p.numel() for n, p in model.named_parameters() if any(proj in n for proj in [".q.weight", ".k.weight", ".v.weight", ".o.weight"]))

    log(f"  Start Training [{cond_cfg['name']}, Seed {seed}]: Params Total={total_params:,}, AttnParams={attn_params:,}, eps={epsilon}")

    t0 = time.time()
    step_history = []

    for step in range(1, max_steps + 1):
        model.train()
        x, y = dataset.get_batch(split="train", batch_size=batch_size, device=device)
        optimizer.zero_grad()
        logits = model(x)
        task_loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

        if epsilon > 0.0:
            reg_energy = model.get_attn_dirichlet_energy()
            total_loss = task_loss + epsilon * reg_energy
        else:
            reg_energy = torch.tensor(0.0)
            total_loss = task_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Fast feedback on first 5 steps of Run 1
        if step <= 5 and seed == 42 and "eps2e-3" in cond_cfg["name"]:
            log(f"  [Fast Feedback Step {step}/5] TaskLoss = {task_loss.item():.4f} | RegEnergy = {reg_energy.item():.2f}")

        # Periodic logging every 150 steps
        if step % 150 == 0 or step == max_steps:
            val_loss, val_se, val_ppl, n_seq = evaluate_language_model(model, dataset, num_batches=15, batch_size=batch_size, device=device)
            adj_cos = model.get_mean_attn_adj_cos()
            st_s = step / (time.time() - t0)
            eta_s = (max_steps - step) / st_s if st_s > 0 else 0
            log(f"  Step {step:3d}/{max_steps} | Val Loss: {val_loss:.4f} (SE: {val_se:.4f}) | Val PPL: {val_ppl:5.2f} | Attn AdjCos: {adj_cos:.4f} | {st_s:.1f} st/s (ETA: {eta_s:.1f}s)")
            step_history.append({
                "step": step,
                "val_loss": val_loss,
                "val_se": val_se,
                "val_ppl": val_ppl,
                "attn_adj_cos": adj_cos
            })

    wall_time = time.time() - t0
    final_adj_cos = model.get_mean_attn_adj_cos()
    final_val_loss, final_val_se, final_val_ppl, _ = evaluate_language_model(model, dataset, num_batches=20, batch_size=batch_size, device=device)

    # 2D-DCT Sparsity sweep evaluation
    sparsity_levels = [0.0, 0.5, 0.75, 0.90]
    log(f"  Evaluating 2D-DCT magnitude pruning on attention weights (0%, 50%, 75%, 90%)...")
    pruning_res = evaluate_dct_pruning_sweep(model, dataset, sparsity_levels, D_basis, num_batches=15, batch_size=batch_size, device=device)

    # Sample attention matrix for visualization (Layer 0, W_q)
    sample_wq = model.blocks[0].attn.q.weight.detach().cpu().numpy()

    return {
        "final_val_loss": final_val_loss,
        "final_val_se": final_val_se,
        "final_val_ppl": final_val_ppl,
        "final_adj_cos": final_adj_cos,
        "wall_time": wall_time,
        "step_history": step_history,
        "pruning_res": pruning_res,
        "sample_wq": sample_wq
    }

# ---------------------------------------------------------
# Main Suite
# ---------------------------------------------------------
def main():
    log("=========================================================================================")
    log(" EXPERIMENT v385: Topographic Attention in Transformers (Autoregressive LM)")
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

    D_basis = get_dct_basis(D_MODEL, device="cpu")

    # -----------------------------------------------------
    # Conditions definition (Regla de Oro: Primero el Candidato)
    # -----------------------------------------------------
    conditions = [
        # Candidate 1: Dirichlet 2D on attention projections (eps = 2e-3)
        {
            "name": "Topographic_Attn_eps2e-3",
            "epsilon": 0.002,
            "desc": "Topographic Attention (Dirichlet 2D, eps=2e-3 on W_q, W_k, W_v, W_o)"
        },
        # Candidate 2: Dirichlet 2D on attention projections (eps = 5e-4)
        {
            "name": "Topographic_Attn_eps5e-4",
            "epsilon": 0.0005,
            "desc": "Topographic Attention (Dirichlet 2D, eps=5e-4 on W_q, W_k, W_v, W_o)"
        },
        # Candidate 3: Dirichlet 2D on attention projections (eps = 1e-2)
        {
            "name": "Topographic_Attn_eps1e-2",
            "epsilon": 0.01,
            "desc": "Topographic Attention (Dirichlet 2D, eps=1e-2 on W_q, W_k, W_v, W_o)"
        },
        # Baseline 1: Standard AdamW (eps = 0.0)
        {
            "name": "Baseline_Standard_AdamW",
            "epsilon": 0.0,
            "desc": "Standard Transformer (eps=0.0, uncoupled attention projections)"
        }
    ]

    log(f"Total Conditions: {len(conditions)} | Seeds per Condition: {len(SEEDS)} | Total Runs: {len(conditions) * len(SEEDS)}")
    log("Architecture Inventory:")
    log("  NanoLanguageModel (2 layers, d_model=128, n_heads=4, head_dim=32, ffn_dim=256, seq_len=128)")
    log("  Attn Projections per layer: W_q (128x128), W_k (128x128), W_v (128x128), W_o (128x128)")
    log("  Total Attn Parameters: 131,072 | Total Model Parameters: ~297k")

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
                D_basis=D_basis,
                max_steps=MAX_STEPS,
                lr=LR,
                batch_size=BATCH_SIZE,
                device="cpu"
            )
            res["seed"] = seed
            res["cond_name"] = cond_name
            seed_runs.append(res)

            if cond_name not in sample_heatmaps:
                sample_heatmaps[cond_name] = res["sample_wq"]

        all_results.append({
            "cond_name": cond_name,
            "cond_cfg": cond,
            "runs": seed_runs
        })

    # -----------------------------------------------------
    # Phase 2: Consolidation & Table
    # -----------------------------------------------------
    log("\n=========================================================================================")
    log(" PHASE 2: CONSOLIDATED RESULTS (Val Perplexity, Attn AdjCos & DCT Pruning)")
    log("=========================================================================================")

    summary_rows = []
    log(f"{'Condition':<28} | {'Val Loss':<14} | {'Val PPL':<12} | {'Attn AdjCos':<12} | {'DCT 50% PPL':<12} | {'DCT 75% PPL':<12} | {'DCT 90% PPL':<12}")
    log("-" * 115)

    for cond_res in all_results:
        cname = cond_res["cond_name"]
        runs = cond_res["runs"]
        val_losses = [r["final_val_loss"] for r in runs]
        val_ppls = [r["final_val_ppl"] for r in runs]
        val_ses = [r["final_val_se"] for r in runs]
        adj_coss = [r["final_adj_cos"] for r in runs]

        m_loss = float(np.mean(val_losses))
        sd_loss = float(np.std(val_losses))
        m_ppl = float(np.mean(val_ppls))
        sd_ppl = float(np.std(val_ppls))
        m_se = float(np.mean(val_ses))
        m_adjcos = float(np.mean(adj_coss))
        sd_adjcos = float(np.std(adj_coss))

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
            "mean_adjcos": m_adjcos,
            "std_adjcos": sd_adjcos,
            "dct_ppl_0": m_ppl,
            "dct_ppl_50": dct_ppl_50,
            "dct_ppl_75": dct_ppl_75,
            "dct_ppl_90": dct_ppl_90
        })

        log(f"{cname:<28} | {m_loss:5.3f} +/- {sd_loss:5.3f} | {m_ppl:5.2f} +/- {sd_ppl:4.2f} | {m_adjcos:5.4f} +/- {sd_adjcos:5.4f} | {dct_ppl_50:11.2f} | {dct_ppl_75:11.2f} | {dct_ppl_90:11.2f}")

    # -----------------------------------------------------
    # Phase 3: Artifact Generation (Figures & JSON)
    # -----------------------------------------------------
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    json_path = "results/raw/v385_topographic_attention.json"
    raw_payload = {
        "experiment_id": "v385",
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
                        "final_adj_cos": r["final_adj_cos"],
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

    # Subplot 1: Perplexity vs DCT Sparsity
    ax1 = fig.add_subplot(1, 3, 1)
    sparsities = [0.0, 50.0, 75.0, 90.0]
    colors = {"Topographic_Attn_eps2e-3": "#2ecc71",
              "Topographic_Attn_eps5e-4": "#3498db",
              "Topographic_Attn_eps1e-2": "#9b59b6",
              "Baseline_Standard_AdamW": "#e74c3c"}

    for s in summary_rows:
        cname = s["cond_name"]
        ppls = [s["dct_ppl_0"], s["dct_ppl_50"], s["dct_ppl_75"], s["dct_ppl_90"]]
        ax1.plot(sparsities, ppls, marker="o", label=cname.replace("_", " "),
                 color=colors.get(cname, "#34495e"), linewidth=2.0)

    ax1.set_title("Attn 2D-DCT Pruning vs Validation Perplexity", fontsize=11, weight="bold")
    ax1.set_xlabel("Attention 2D-DCT Sparsity (%)", fontsize=10)
    ax1.set_ylabel("Validation Perplexity (lower is better)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8)

    # Subplot 2: Adjacent Cosine Similarity in Attention
    ax2 = fig.add_subplot(1, 3, 2)
    cnames = [s["cond_name"].replace("Topographic_Attn_", "Topo_").replace("Baseline_", "") for s in summary_rows]
    adj_means = [s["mean_adjcos"] for s in summary_rows]
    adj_stds = [s["std_adjcos"] for s in summary_rows]
    bar_colors = [colors.get(s["cond_name"], "#34495e") for s in summary_rows]

    ax2.bar(cnames, adj_means, yerr=adj_stds, color=bar_colors, edgecolor="black", alpha=0.85, capsize=4)
    ax2.set_title("Attn Adjacent Cosine Similarity (Cortical Smoothness)", fontsize=11, weight="bold")
    ax2.set_ylabel("Adjacent Cosine Similarity", fontsize=10)
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    for i, m in enumerate(adj_means):
        ax2.text(i, m + 0.03, f"{m:.3f}", ha="center", fontsize=9, weight="bold")

    # Subplot 3: Heatmap Comparison of W_q (Layer 0)
    ax3 = fig.add_subplot(1, 3, 3)
    # Side-by-side or difference between Topo eps2e-3 and Baseline
    topo_wq = sample_heatmaps.get("Topographic_Attn_eps2e-3", np.zeros((128, 128)))
    base_wq = sample_heatmaps.get("Baseline_Standard_AdamW", np.zeros((128, 128)))
    # Composite display: Top half Topo, Bottom half Base
    composite = np.vstack([topo_wq[:64, :64], base_wq[:64, :64]])
    im = ax3.imshow(composite, cmap="magma", aspect="auto")
    ax3.axhline(64, color="white", linewidth=2.0, linestyle="--")
    ax3.set_title("Weight Receptive Fields (Top: Topo eps2e-3 | Bottom: Baseline)", fontsize=10, weight="bold")
    ax3.set_xlabel("Receptive Field Dimension", fontsize=9)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_path = "results/figures/v385_topographic_attention_curves.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    log(f"Figure saved to {fig_path}")

    log("\n=========================================================================================")
    log(" EXPERIMENT v385 FINISHED SUCCESSFULLY")
    log("=========================================================================================")

if __name__ == "__main__":
    main()
