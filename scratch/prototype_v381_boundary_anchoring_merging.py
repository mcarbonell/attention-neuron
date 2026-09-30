#!/usr/bin/env python3
"""
================================================================================
EXPERIMENT v381: BOUNDARY ANCHORING & ZERO-COST MODEL MERGING (H1 RESOLUTION)
================================================================================
Hypothesis & Scientific Objective:
  In v380, unattenuated macroscopic regularization proved that spatial Dirichlet
  coupling forces the weight matrix into a continuous topographic manifold
  (AdjCos = 0.97) and collapses effective rank (Rank 4 retains 90% energy).
  However, cross-seed unpermuted alignment (rho_raw) remained low (~0.02 - 0.08)
  because the spatial prior possesses a continuous translation/reflection gauge
  symmetry: every seed self-organizes a smooth map, but each seed chooses an
  arbitrary phase/offset along the neuron axis.

  In v381, we introduce BOUNDARY CONDITIONS (Dirichlet boundary pinning and
  monotonic coordinate tilt potentials):
    1. Dirichlet Boundary Anchor:
       Pins row 0 to +c and row (d_out-1) to -c, imposing a unique Dirichlet
       boundary value problem (BVP) on the 2D lattice.
    2. Coordinate Tilt Potential:
       Adds a linear potential gradient -lambda * sum_i y_i * mean_j(W_{i, j})
       where y_i in [-1, 1], breaking both translation and reflection symmetries.

  Core Testable Prediction:
    By freezing the global coordinate system, independent training runs (seeds)
    are forced into the SAME canonical permutation basin.
    Therefore, naive direct weight averaging across seeds (Zero-Cost Model Soup):
      W_soup = (W_A + W_B) / 2
    will retain high test accuracy without suffering the catastrophic
    permutation barrier that collapses baseline standard networks (~10% - 30%).

Compliance Checklist (GEMINI.md):
  1. No modification of existing scripts (new file prototype_v381_*.py).
  2. Candidates evaluated FIRST, Baselines LAST.
  3. Strict vectorization (zero python for-loops in tensor operations).
  4. Fast feedback in first 5 batches of Epoch 1.
  5. Timestamps [+HH:MM:SS.ss] with flush=True on all logs.
  6. Multi-seed evaluation (3 independent seeds: 42, 100, 2026).
  7. Exact measurement of Pairwise Model Soup Accuracy and Barrier across seeds.
  8. Full structured JSON output in results/raw/v381_boundary_anchoring_merging.json.
  9. Heatmaps and Soup Barrier plots persisted in results/figures/.
================================================================================
"""

import os
import sys
import time
import json
import math
import copy
import subprocess
import platform
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Matplotlib headless setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- TIMER & LOGGING ---
T0 = time.time()

def log(msg: str):
    elapsed = time.time() - T0
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60
    print(f"[+{h:02d}:{m:02d}:{s:05.2f}] {msg}", flush=True)

def get_git_commit() -> str:
    try:
        res = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return res.decode("utf-8").strip()
    except Exception:
        return "unknown"

# --- ANCHORED TOPOGRAPHIC REGULARIZER MODULE ---
class AnchoredTopographicRegularizer(nn.Module):
    """
    Combines unattenuated 2D Dirichlet spatial coupling with boundary pinning
    or linear coordinate tilt to lock the global coordinate frame.
    """
    def __init__(
        self,
        mode: str = "none",
        epsilon: float = 0.0,
        anchor_type: str = "none",
        lambda_anchor: float = 0.0,
        pin_val: float = 0.05
    ):
        super().__init__()
        self.mode = mode
        self.epsilon = epsilon
        self.anchor_type = anchor_type
        self.lambda_anchor = lambda_anchor
        self.pin_val = pin_val

    def penalty(self, weight: torch.Tensor) -> torch.Tensor:
        total_p = torch.tensor(0.0, device=weight.device)

        # 1. Unattenuated Spatial Dirichlet Energy
        if self.epsilon > 0.0 and self.mode == "dirichlet":
            diff_r = weight[1:, :] - weight[:-1, :]
            diff_c = weight[:, 1:] - weight[:, :-1]
            total_p = total_p + self.epsilon * 0.5 * (diff_r.pow(2).sum() + diff_c.pow(2).sum())

        # 2. Boundary Condition / Coordinate Anchoring
        if self.lambda_anchor > 0.0:
            d_out, d_in = weight.shape

            if self.anchor_type == "dirichlet_boundary":
                # Fix top row (i=0) to +pin_val and bottom row (i=d_out-1) to -pin_val
                top_err = (weight[0, :] - self.pin_val).pow(2).sum()
                bot_err = (weight[-1, :] + self.pin_val).pow(2).sum()
                total_p = total_p + self.lambda_anchor * 0.5 * (top_err + bot_err)

            elif self.anchor_type == "coordinate_tilt":
                # Linear coordinate potential y_i in [-1, 1] along row axis
                # Minimizing - sum(y_i * W_{i, j}) pulls top negative, bottom positive
                y = torch.linspace(-1.0, 1.0, steps=d_out, device=weight.device).unsqueeze(1)
                tilt_energy = - (y * weight).sum()
                total_p = total_p + self.lambda_anchor * tilt_energy

        return total_p

# --- MODEL ARCHITECTURE ---
class TopoMLPv381(nn.Module):
    """
    3-Layer Multi-Layer Perceptron (784 -> 256 -> 128 -> 10)
    Regularization is applied to representation layers W1 and W2.
    """
    def __init__(
        self,
        mode: str = "none",
        epsilon: float = 0.0,
        anchor_type: str = "none",
        lambda_anchor: float = 0.0
    ):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.regularizer = AnchoredTopographicRegularizer(
            mode=mode,
            epsilon=epsilon,
            anchor_type=anchor_type,
            lambda_anchor=lambda_anchor
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        return out

    def compute_reg_loss(self) -> torch.Tensor:
        reg1 = self.regularizer.penalty(self.fc1.weight)
        reg2 = self.regularizer.penalty(self.fc2.weight)
        return reg1 + reg2

# --- EVALUATION HELPERS ---
def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for vx, vy in data_loader:
            vx, vy = vx.to(device), vy.to(device)
            v_logits = model(vx)
            total_loss += criterion(v_logits, vy).item() * len(vy)
            v_preds = v_logits.argmax(dim=-1)
            correct += (v_preds == vy).sum().item()
            total += len(vy)
    return total_loss / total, (correct / total) * 100.0

def create_model_soup(model_a: nn.Module, model_b: nn.Module) -> nn.Module:
    """
    Creates a naive direct weight average of two models: W_soup = (W_A + W_B) / 2
    """
    soup_model = copy.deepcopy(model_a)
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    soup_state = {}
    for k in state_a:
        soup_state[k] = 0.5 * (state_a[k] + state_b[k])
    soup_model.load_state_dict(soup_state)
    return soup_model

# --- SPECTRAL & TOPOGRAPHIC METRICS ---
def compute_svd_metrics(weight: torch.Tensor) -> Dict[str, any]:
    with torch.no_grad():
        w = weight.detach().float()
        U, S, V = torch.linalg.svd(w, full_matrices=False)
        s_sum = S.sum().item()
        s_norm = S / (s_sum + 1e-12)
        s_sq = S.pow(2)
        s_sq_sum = s_sq.sum().item()
        cum_energy = torch.cumsum(s_sq / (s_sq_sum + 1e-12), dim=0)

        p = s_norm.clamp(min=1e-12)
        entropy = -torch.sum(p * torch.log(p)).item()
        effective_rank = math.exp(entropy)

        ranks = [4, 8, 16, 32, 64]
        energy_pct = {}
        for r in ranks:
            if r <= len(cum_energy):
                energy_pct[f"rank_{r}"] = round(cum_energy[r - 1].item() * 100.0, 2)

        return {
            "effective_rank": round(effective_rank, 2),
            "energy_retained_pct": energy_pct,
            "top1_ratio_pct": round(s_norm[0].item() * 100.0, 2),
            "singular_values_top10": [round(v, 4) for v in S[:10].tolist()]
        }

def compute_cortical_adjacency(weight: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        w = weight.detach().float()
        w_norm = F.normalize(w, p=2, dim=1)
        adj_cos = (w_norm[1:] * w_norm[:-1]).sum(dim=1)
        return {
            "mean_adj_cosine": round(adj_cos.mean().item(), 4),
            "std_adj_cosine": round(adj_cos.std().item(), 4)
        }

def compute_pairwise_alignment(weights_list: List[torch.Tensor]) -> Dict[str, float]:
    n = len(weights_list)
    if n < 2:
        return {"rho_raw_mean": 0.0, "rho_raw_std": 0.0}

    rhos = []
    with torch.no_grad():
        vecs = [w.detach().view(-1).float() for w in weights_list]
        for i in range(n):
            for j in range(i + 1, n):
                cos_sim = F.cosine_similarity(vecs[i], vecs[j], dim=0).item()
                rhos.append(cos_sim)

    rhos_t = torch.tensor(rhos)
    return {
        "rho_raw_mean": round(rhos_t.mean().item(), 4),
        "rho_raw_std": round(rhos_t.std().item(), 4)
    }

# --- TRAINING LOOP SINGLE SEED ---
def train_single_seed(
    cond_name: str,
    mode: str,
    epsilon: float,
    anchor_type: str,
    lambda_anchor: float,
    seed: int,
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    suite_progress: Dict[str, any]
) -> Tuple[nn.Module, Dict[str, any]]:
    torch.manual_seed(seed)
    model = TopoMLPv381(
        mode=mode,
        epsilon=epsilon,
        anchor_type=anchor_type,
        lambda_anchor=lambda_anchor
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    t_start = time.time()
    step_count = 0
    t_overhead = 0.0

    log(f"--- Launching: {cond_name} | Seed: {seed} | Mode: {mode} | Anchor: {anchor_type} (lambda={lambda_anchor}) ---")

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_reg = 0.0
        correct = 0
        total = 0

        for b_idx, (bx, by) in enumerate(train_loader):
            step_count += 1
            suite_progress["completed_steps"] += 1
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            logits = model(bx)
            task_loss = criterion(logits, by)

            t_reg0 = time.time()
            reg_loss = model.compute_reg_loss()
            t_overhead += (time.time() - t_reg0)

            total_loss = task_loss + reg_loss
            total_loss.backward()
            optimizer.step()

            ep_loss += task_loss.item() * len(by)
            ep_reg += reg_loss.item() * len(by)
            preds = logits.argmax(dim=-1)
            correct += (preds == by).sum().item()
            total += len(by)

            # Fast feedback in first 5 batches of Epoch 1
            if epoch == 1 and b_idx < 5:
                batch_acc = (preds == by).float().mean().item() * 100.0
                log(f"  [Ep1/B{b_idx+1}] TaskLoss: {task_loss.item():.4f} | RegLoss: {reg_loss.item():.6f} | Acc: {batch_acc:.1f}%")

        tr_loss = ep_loss / total
        tr_acc = (correct / total) * 100.0
        val_loss, val_acc = evaluate_model(model, test_loader, device)

        # ETA calculations
        elapsed_so_far = time.time() - t_start
        steps_per_sec = step_count / max(elapsed_so_far, 0.001)
        rem_steps_model = (epochs - epoch) * len(train_loader)
        eta_model_sec = rem_steps_model / max(steps_per_sec, 0.001)

        total_elapsed = time.time() - T0
        rem_steps_suite = suite_progress["total_steps"] - suite_progress["completed_steps"]
        overall_sps = suite_progress["completed_steps"] / max(total_elapsed, 0.001)
        eta_suite_sec = rem_steps_suite / max(overall_sps, 0.001)

        log(f"  [{cond_name} S{seed}] Ep {epoch:02d}/{epochs:02d} | TrLoss: {tr_loss:.4f} | TrAcc: {tr_acc:.2f}% | "
            f"ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.2f}% | Speed: {steps_per_sec:.1f} st/s | "
            f"ETA_mod: {eta_model_sec:.0f}s | ETA_suite: {eta_suite_sec:.0f}s")

    t_total = time.time() - t_start

    w1 = model.fc1.weight.data.cpu()
    w2 = model.fc2.weight.data.cpu()
    svd_w1 = compute_svd_metrics(w1)
    cortical_w1 = compute_cortical_adjacency(w1)

    diag_result = {
        "cond_name": cond_name,
        "mode": mode,
        "epsilon": epsilon,
        "anchor_type": anchor_type,
        "lambda_anchor": lambda_anchor,
        "seed": seed,
        "params": n_params,
        "final_tr_loss": round(tr_loss, 4),
        "final_tr_acc": round(tr_acc, 2),
        "val_loss": round(val_loss, 4),
        "val_acc": round(val_acc, 2),
        "wall_clock_time": round(t_total, 2),
        "reg_overhead_time": round(t_overhead, 4),
        "svd_w1": svd_w1,
        "cortical_w1": cortical_w1,
        "w1_tensor": w1,
        "w2_tensor": w2
    }

    return model, diag_result

# --- PLOTTING / FIGURE GENERATION ---
def generate_v381_artifacts(conditions_summary: List[Dict[str, any]], figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Side-by-Side Heatmaps: Seed 42 vs Seed 100 for each condition
    n_conds = len(conditions_summary)
    fig, axes = plt.subplots(2, n_conds, figsize=(3.8 * n_conds, 7.5), squeeze=False)

    for idx, cond in enumerate(conditions_summary):
        # Row 0: Seed 42
        ax0 = axes[0, idx]
        w_s42 = cond["w1_seed42"][:64, :64].numpy()
        im0 = ax0.imshow(w_s42, cmap="coolwarm", aspect="auto")
        ax0.set_title(f"{cond['cond_name']}\nSeed 42", fontsize=9)
        if idx == 0:
            ax0.set_ylabel("Seed 42 (Neurons 0..64)")
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        # Row 1: Seed 100
        ax1 = axes[1, idx]
        w_s100 = cond["w1_seed100"][:64, :64].numpy()
        im1 = ax1.imshow(w_s100, cmap="coolwarm", aspect="auto")
        ax1.set_title(f"Seed 100\n(rho_raw={cond['cross_seed_rho_raw_w1']:.4f})", fontsize=9)
        ax1.set_xlabel("Inputs (Patch 0..64)")
        if idx == 0:
            ax1.set_ylabel("Seed 100 (Neurons 0..64)")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    fig.tight_layout()
    heatmap_path = os.path.join(figures_dir, "v381_cross_seed_heatmaps.png")
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    log(f"Cross-seed alignment heatmaps saved to: {heatmap_path}")

    # 2. Model Souping Accuracy & Barrier Bar Plot
    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = [c["cond_name"] for c in conditions_summary]
    indiv_accs = [c["mean_val_acc"] for c in conditions_summary]
    soup_accs = [c["mean_soup_acc"] for c in conditions_summary]
    barriers = [c["mean_soup_barrier"] for c in conditions_summary]

    x = range(len(names))
    width = 0.35

    rects1 = ax.bar([i - width/2 for i in x], indiv_accs, width, label="Individual Seed Acc (%)", color="#4C72B0")
    rects2 = ax.bar([i + width/2 for i in x], soup_accs, width, label="Model Soup Acc (%) [Naive Average]", color="#55A868")

    # Add text labels on soup bars showing barrier
    for i, (s_acc, bar) in enumerate(zip(soup_accs, barriers)):
        ax.annotate(f"Soup: {s_acc:.1f}%\n(Barrier: -{bar:.1f}%)",
                    xy=(i + width/2, s_acc / 2),
                    xytext=(0, 0), textcoords="offset points",
                    ha='center', va='center', fontsize=8, color="white", weight="bold")

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("v381: Zero-Cost Model Merging (Model Souping) Across Independent Seeds", fontsize=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9.5)

    fig.tight_layout()
    soup_path = os.path.join(figures_dir, "v381_model_soup_barrier.png")
    fig.savefig(soup_path, dpi=150)
    plt.close(fig)
    log(f"Model soup barrier plot saved to: {soup_path}")

# --- MAIN SUITE EXECUTION ---
def main():
    log("=" * 80)
    log("EXPERIMENT v381: BOUNDARY ANCHORING & ZERO-COST MODEL MERGING")
    log("=" * 80)

    commit_hash = get_git_commit()
    py_ver = platform.python_version()
    torch_ver = torch.__version__
    device_name = "cpu"
    device = torch.device(device_name)

    log(f"Platform: {platform.platform()} | Python: {py_ver} | PyTorch: {torch_ver} | Commit: {commit_hash}")
    log(f"Device: {device} (Ryzen CPU Vectorized Operations)")

    sample_model = TopoMLPv381()
    total_params = sum(p.numel() for p in sample_model.parameters())
    log("\n--- ARCHITECTURAL INVENTORY (3-Layer MLP) ---")
    log(f"Layer 1 (fc1): Linear(784 -> 256) | W1: [256, 784] ({784*256:,} params) + b1: [256] params")
    log(f"Layer 2 (fc2): Linear(256 -> 128) | W2: [128, 256] ({256*128:,} params) + b2: [128] params")
    log(f"Layer 3 (fc3): Linear(128 -> 10)  | W3: [10, 128]   ({128*10:,} params)  + b3: [10] params")
    log(f"Total Trainable Parameters: {total_params:,}")
    log("Anchor Targets: W1 and W2 boundaries & coordinate tilt\n")

    batch_size = 256
    epochs = 10
    seeds = [42, 100, 2026]

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    log(f"Loading MNIST from: {data_dir}")
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    log(f"Dataset: Train={len(train_dataset):,} samples, Test={len(test_dataset):,} samples\n")

    # TEST SUITE: Candidates FIRST, Baselines LAST
    test_suite = [
        # Candidate 1: Dirichlet 2D + Dirichlet Boundary Pinning (Top row +c, Bottom row -c)
        {
            "cond_name": "Topo_BoundaryAnchor_eps2e-3",
            "mode": "dirichlet",
            "epsilon": 2e-3,
            "anchor_type": "dirichlet_boundary",
            "lambda_anchor": 0.05
        },
        # Candidate 2: Dirichlet 2D + Linear Coordinate Tilt (lambda = 1e-3, balanced)
        {
            "cond_name": "Topo_CoordTilt_eps2e-3_lam1e-3",
            "mode": "dirichlet",
            "epsilon": 2e-3,
            "anchor_type": "coordinate_tilt",
            "lambda_anchor": 1e-3
        },
        # Candidate 3: Dirichlet 2D + Strong Coordinate Tilt (lambda = 2e-3)
        {
            "cond_name": "Topo_CoordTilt_eps2e-3_lam2e-3",
            "mode": "dirichlet",
            "epsilon": 2e-3,
            "anchor_type": "coordinate_tilt",
            "lambda_anchor": 2e-3
        },
        # Reference 1: Dirichlet 2D Free (v380 winner, no anchor)
        {
            "cond_name": "Topo_Free_NoAnchor_eps2e-3",
            "mode": "dirichlet",
            "epsilon": 2e-3,
            "anchor_type": "none",
            "lambda_anchor": 0.0
        },
        # Reference 2: Standard Baseline AdamW (zero topography, zero anchor)
        {
            "cond_name": "Baseline_Standard_AdamW",
            "mode": "none",
            "epsilon": 0.0,
            "anchor_type": "none",
            "lambda_anchor": 0.0
        }
    ]

    total_runs = len(test_suite) * len(seeds)
    total_steps = total_runs * epochs * len(train_loader)
    suite_progress = {
        "total_steps": total_steps,
        "completed_steps": 0
    }

    log(f"Test Suite: {len(test_suite)} conditions x {len(seeds)} seeds = {total_runs} training runs total.")
    log(f"Step Budget: {total_steps:,} steps across all runs.\n")

    all_run_results = []
    conditions_summary = []

    for cond in test_suite:
        cond_name = cond["cond_name"]
        mode = cond["mode"]
        eps = cond["epsilon"]
        anchor_type = cond["anchor_type"]
        lambda_anchor = cond["lambda_anchor"]

        trained_models = []
        seed_runs = []
        w1_seeds = []
        w2_seeds = []

        for s in seeds:
            m, res = train_single_seed(
                cond_name=cond_name,
                mode=mode,
                epsilon=eps,
                anchor_type=anchor_type,
                lambda_anchor=lambda_anchor,
                seed=s,
                epochs=epochs,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                suite_progress=suite_progress
            )
            trained_models.append(m)
            w1_seeds.append(res.pop("w1_tensor"))
            w2_seeds.append(res.pop("w2_tensor"))
            seed_runs.append(res)
            all_run_results.append(res)

        # Cross-Seed Alignment Metric (H1)
        align_w1 = compute_pairwise_alignment(w1_seeds)
        align_w2 = compute_pairwise_alignment(w2_seeds)

        # MODEL SOUPING: Naive pairwise weight averaging across independent seeds
        soup_accs = []
        soup_losses = []
        n_m = len(trained_models)
        for i in range(n_m):
            for j in range(i + 1, n_m):
                m_soup = create_model_soup(trained_models[i], trained_models[j])
                s_loss, s_acc = evaluate_model(m_soup, test_loader, device)
                soup_accs.append(s_acc)
                soup_losses.append(s_loss)
                log(f"  [SOUP: {cond_name}] Seed {seeds[i]} + Seed {seeds[j]} -> Soup Acc: {s_acc:.2f}% | Loss: {s_loss:.4f}")

        mean_soup_acc = round(sum(soup_accs) / len(soup_accs), 2)
        std_soup_acc = round(float(torch.tensor(soup_accs).std().item()), 2)

        val_accs = [r["val_acc"] for r in seed_runs]
        val_losses = [r["val_loss"] for r in seed_runs]
        eff_ranks = [r["svd_w1"]["effective_rank"] for r in seed_runs]
        adj_cosines = [r["cortical_w1"]["mean_adj_cosine"] for r in seed_runs]
        wall_times = [r["wall_clock_time"] for r in seed_runs]
        mean_indiv_acc = round(sum(val_accs) / len(val_accs), 2)

        soup_barrier = round(mean_indiv_acc - mean_soup_acc, 2)

        summary_entry = {
            "cond_name": cond_name,
            "mode": mode,
            "epsilon": eps,
            "anchor_type": anchor_type,
            "lambda_anchor": lambda_anchor,
            "mean_val_acc": mean_indiv_acc,
            "std_val_acc": round(float(torch.tensor(val_accs).std().item()), 2),
            "mean_soup_acc": mean_soup_acc,
            "std_soup_acc": std_soup_acc,
            "mean_soup_barrier": soup_barrier,
            "mean_eff_rank_w1": round(sum(eff_ranks) / len(eff_ranks), 2),
            "mean_adj_cosine_w1": round(sum(adj_cosines) / len(adj_cosines), 4),
            "cross_seed_rho_raw_w1": align_w1["rho_raw_mean"],
            "cross_seed_rho_raw_w2": align_w2["rho_raw_mean"],
            "mean_wall_time_s": round(sum(wall_times) / len(wall_times), 2),
            "w1_seed42": w1_seeds[0],
            "w1_seed100": w1_seeds[1]
        }
        conditions_summary.append(summary_entry)

    # --- REPORT TABLE ---
    log("\n" + "=" * 125)
    log(f"{'Condition':<32} | {'Indiv Acc (%)':<14} | {'Soup Acc (%)':<14} | {'Barrier (dAcc)':<14} | {'rho_raw W1':<11} | {'rho_raw W2':<11} | {'EffRank W1':<10}")
    log("-" * 125)
    for c in conditions_summary:
        log(f"{c['cond_name']:<32} | "
            f"{c['mean_val_acc']:>5.2f} +/- {c['std_val_acc']:<4.2f} | "
            f"{c['mean_soup_acc']:>5.2f} +/- {c['std_soup_acc']:<4.2f} | "
            f"{c['mean_soup_barrier']:>14.2f} | "
            f"{c['cross_seed_rho_raw_w1']:>11.4f} | "
            f"{c['cross_seed_rho_raw_w2']:>11.4f} | "
            f"{c['mean_eff_rank_w1']:>10.2f}")
    log("=" * 125 + "\n")

    # Generate figures and persist
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
    generate_v381_artifacts(conditions_summary, figures_dir)

    for c in conditions_summary:
        c.pop("w1_seed42", None)
        c.pop("w1_seed100", None)

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "v381_boundary_anchoring_merging.json")

    payload = {
        "experiment_id": "v381",
        "date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "commit_hash": commit_hash,
        "platform": platform.platform(),
        "python_version": py_ver,
        "pytorch_version": torch_ver,
        "device": device_name,
        "hyperparameters": {
            "batch_size": batch_size,
            "epochs": epochs,
            "seeds": seeds,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "dataset": "MNIST (60k train / 10k test)",
            "model": "MLP (784 -> 256 -> 128 -> 10)"
        },
        "conditions_summary": conditions_summary,
        "individual_runs": all_run_results
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    log(f"Structured experimental record saved to: {json_path}")
    log("Execution of Experiment v381 completed successfully.")

if __name__ == "__main__":
    main()
