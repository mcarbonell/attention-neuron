#!/usr/bin/env python3
"""
================================================================================
EXPERIMENT v383: 2D-DCT SPECTRAL COMPRESSION & HARMONIC PRUNING
================================================================================
Hypothesis & Scientific Objective:
  In v380, Topographic Cellular Regularization (TCR) forced neural network weight
  matrices into spatially smooth, low-rank manifolds (AdjCos = 0.97, SVD Rank 4
  captures 90% energy in Turing DoG).

  In v383, we test the primary downstream application of this smooth manifold:
  PARAMETRIC COMPRESSION VIA 2D DISCRETE COSINE TRANSFORM (2D-DCT).

  Theoretical Prediction:
    1. Standard Baseline (AdamW):
       Weights are high-frequency white noise. In the 2D-DCT domain, energy is
       dispersed broadly across all (u, v) frequency bins. Truncating 75% - 95%
       of high-frequency DCT coefficients will destroy critical features,
       causing catastrophic accuracy collapse.
    2. TCR Models (Dirichlet 2D & Turing DoG):
       Weights are low-frequency continuous fields. In the 2D-DCT domain,
       spectral energy is concentrated in the lowest spatial harmonics.
       Therefore, zeroing out 50%, 75%, 90%, or even 95% of DCT coefficients
       (achieving 2x, 4x, 10x, 20x parameter compression) will preserve the
       reconstructed weight matrix and sustain task accuracy (>95% - 97%).

Compliance Checklist (GEMINI.md):
  1. No modification of existing scripts (new file prototype_v383_*.py).
  2. Candidates evaluated FIRST (Turing DoG, Dirichlet 2D), Baseline LAST.
  3. Strict vectorization (zero python for-loops in tensor operations;
     orthonormal DCT2D via matrix product D_H @ W @ D_W.T).
  4. Fast feedback in first 5 batches of Epoch 1.
  5. Timestamps [+HH:MM:SS.ss] with flush=True on all logs.
  6. Multi-seed evaluation (3 independent seeds: 42, 100, 2026).
  7. Exact measurement of Accuracy vs Sparsity (0%, 50%, 75%, 90%, 95%, 98%).
  8. Full structured JSON output in results/raw/v383_spectral_dct_compression.json.
  9. Compression curves and 2D spectral energy heatmaps persisted in results/figures/.
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

# --- EXACT ORTHONORMAL 2D-DCT MODULE ---
def create_dct_matrix(n: int, device: torch.device) -> torch.Tensor:
    """
    Constructs an orthonormal Type-II DCT matrix of size n x n.
    Satisfies D @ D.T == D.T @ D == Eye(n).
    """
    coords_n = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(0) # [1, n]
    coords_k = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1) # [n, 1]
    
    alpha = torch.ones(n, 1, dtype=torch.float32, device=device)
    alpha[0, 0] = 1.0 / math.sqrt(2.0)
    
    D = math.sqrt(2.0 / n) * alpha * torch.cos(math.pi * coords_k * (2.0 * coords_n + 1.0) / (2.0 * n))
    return D

class OrthonormalDCT2D:
    def __init__(self, h: int, w: int, device: torch.device):
        self.D_H = create_dct_matrix(h, device)
        self.D_W = create_dct_matrix(w, device)

    def forward(self, matrix: torch.Tensor) -> torch.Tensor:
        # C = D_H @ matrix @ D_W.T
        return self.D_H @ matrix @ self.D_W.t()

    def inverse(self, spectral_coeffs: torch.Tensor) -> torch.Tensor:
        # W = D_H.T @ spectral_coeffs @ D_W
        return self.D_H.t() @ spectral_coeffs @ self.D_W

# --- TOPOGRAPHIC REGULARIZER MODULE ---
class UnattenuatedTopographicRegularizer(nn.Module):
    def __init__(self, mode: str = "none", epsilon: float = 0.0):
        super().__init__()
        self.mode = mode
        self.epsilon = epsilon
        
        if mode == "turing":
            dog = torch.tensor([
                [-0.05, -0.10, -0.10, -0.10, -0.05],
                [-0.10,  0.20,  0.50,  0.20, -0.10],
                [-0.10,  0.50,  1.00,  0.50, -0.10],
                [-0.10,  0.20,  0.50,  0.20, -0.10],
                [-0.05, -0.10, -0.10, -0.10, -0.05]
            ], dtype=torch.float32)
            dog = dog - dog.mean()
            self.register_buffer("kernel", dog.unsqueeze(0).unsqueeze(0))
        else:
            self.kernel = None

    def penalty(self, weight: torch.Tensor) -> torch.Tensor:
        if self.epsilon <= 0.0 or self.mode == "none":
            return torch.tensor(0.0, device=weight.device)

        if self.mode == "dirichlet":
            diff_r = weight[1:, :] - weight[:-1, :]
            diff_c = weight[:, 1:] - weight[:, :-1]
            return self.epsilon * 0.5 * (diff_r.pow(2).sum() + diff_c.pow(2).sum())

        elif self.mode == "turing":
            w_img = weight.unsqueeze(0).unsqueeze(0)
            pad = self.kernel.shape[-1] // 2
            filtered = F.conv2d(w_img, self.kernel, padding=pad).squeeze(0).squeeze(0)
            norm_f = filtered.norm(p="fro").clamp(min=1e-6)
            alignment = - (weight * filtered).sum() / norm_f
            return self.epsilon * alignment

        return torch.tensor(0.0, device=weight.device)

# --- MODEL ARCHITECTURE ---
class TopoMLPv383(nn.Module):
    def __init__(self, mode: str = "none", epsilon: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.regularizer = UnattenuatedTopographicRegularizer(mode=mode, epsilon=epsilon)

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

# --- SPECTRAL DCT PRUNING FUNCTION ---
def evaluate_spectral_dct_pruning(
    base_model: nn.Module,
    sparsity_levels: List[float],
    test_loader: DataLoader,
    device: torch.device,
    dct1: OrthonormalDCT2D,
    dct2: OrthonormalDCT2D
) -> Dict[str, any]:
    """
    Evaluates model test accuracy under 2D-DCT magnitude thresholding.
    Sparsity = fraction of smallest DCT coefficients set to zero.
    Compression Ratio = 1 / (1 - Sparsity).
    """
    base_model.eval()
    w1_orig = base_model.fc1.weight.data.clone().to(device)
    w2_orig = base_model.fc2.weight.data.clone().to(device)

    # Compute 2D-DCT
    c1 = dct1.forward(w1_orig)
    c2 = dct2.forward(w2_orig)
    
    total_energy_w1 = c1.pow(2).sum().item()
    total_energy_w2 = c2.pow(2).sum().item()

    pruning_results = []

    for sp in sparsity_levels:
        comp_ratio = round(1.0 / max(1.0 - sp, 1e-4), 2)

        if sp <= 0.0:
            c1_pruned = c1.clone()
            c2_pruned = c2.clone()
            energy_ret1 = 100.0
            energy_ret2 = 100.0
        else:
            # Thresholding on C1
            k1 = int(round(sp * c1.numel()))
            if k1 > 0:
                thresh1 = torch.kthvalue(c1.abs().view(-1), k1).values.item()
                mask1 = (c1.abs() > thresh1).float()
                c1_pruned = c1 * mask1
            else:
                c1_pruned = c1.clone()
            energy_ret1 = (c1_pruned.pow(2).sum().item() / max(total_energy_w1, 1e-12)) * 100.0

            # Thresholding on C2
            k2 = int(round(sp * c2.numel()))
            if k2 > 0:
                thresh2 = torch.kthvalue(c2.abs().view(-1), k2).values.item()
                mask2 = (c2.abs() > thresh2).float()
                c2_pruned = c2 * mask2
            else:
                c2_pruned = c2.clone()
            energy_ret2 = (c2_pruned.pow(2).sum().item() / max(total_energy_w2, 1e-12)) * 100.0

        # IDCT reconstruction
        w1_rec = dct1.inverse(c1_pruned)
        w2_rec = dct2.inverse(c2_pruned)

        # Inject into pruned model copy and evaluate
        pruned_model = copy.deepcopy(base_model)
        pruned_model.fc1.weight.data.copy_(w1_rec)
        pruned_model.fc2.weight.data.copy_(w2_rec)

        p_loss, p_acc = evaluate_model(pruned_model, test_loader, device)

        pruning_results.append({
            "sparsity": sp,
            "compression_ratio": f"{comp_ratio}x",
            "test_acc": round(p_acc, 2),
            "test_loss": round(p_loss, 4),
            "energy_retained_w1_pct": round(energy_ret1, 2),
            "energy_retained_w2_pct": round(energy_ret2, 2)
        })

    return {
        "pruning_results": pruning_results,
        "c1_spectrum_sample": c1[:64, :64].abs().cpu()
    }

# --- TRAINING LOOP SINGLE SEED ---
def train_single_seed(
    cond_name: str,
    mode: str,
    epsilon: float,
    seed: int,
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    suite_progress: Dict[str, any]
) -> Tuple[nn.Module, Dict[str, any]]:
    torch.manual_seed(seed)
    model = TopoMLPv383(mode=mode, epsilon=epsilon).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    t_start = time.time()
    step_count = 0
    t_overhead = 0.0

    log(f"--- Training: {cond_name} | Seed: {seed} | Mode: {mode} | Eps: {epsilon} ---")

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
    w_norm = F.normalize(w1, p=2, dim=1)
    adj_cos = (w_norm[1:] * w_norm[:-1]).sum(dim=1).mean().item()

    diag_result = {
        "cond_name": cond_name,
        "mode": mode,
        "epsilon": epsilon,
        "seed": seed,
        "params": n_params,
        "final_tr_loss": round(tr_loss, 4),
        "final_tr_acc": round(tr_acc, 2),
        "val_loss": round(val_loss, 4),
        "val_acc": round(val_acc, 2),
        "adj_cosine_w1": round(adj_cos, 4),
        "wall_clock_time": round(t_total, 2),
        "reg_overhead_time": round(t_overhead, 4)
    }

    return model, diag_result

# --- PLOTTING / FIGURE GENERATION ---
def generate_v383_artifacts(conditions_summary: List[Dict[str, any]], figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)

    # 1. 2D-DCT Energy Spectra Heatmaps
    n_conds = len(conditions_summary)
    fig, axes = plt.subplots(1, n_conds, figsize=(4 * n_conds, 4.2), squeeze=False)

    for idx, cond in enumerate(conditions_summary):
        ax = axes[0, idx]
        spec = cond["c1_spectrum_sample"].numpy() # [64, 64]
        # Log-scale magnitude
        log_spec = torch.log1p(torch.tensor(spec)).numpy()
        im = ax.imshow(log_spec, cmap="inferno", aspect="auto")
        ax.set_title(f"{cond['cond_name']}\n(Top 64x64 DCT Freqs)", fontsize=9)
        ax.set_xlabel("Horizontal Frequency (u)")
        if idx == 0:
            ax.set_ylabel("Vertical Frequency (v)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    spec_path = os.path.join(figures_dir, "v383_dct_energy_spectra.png")
    fig.savefig(spec_path, dpi=150)
    plt.close(fig)
    log(f"DCT 2D energy spectra heatmaps saved to: {spec_path}")

    # 2. Accuracy vs Compression Ratio Curve
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sparsity_labels = ["0% (1x)", "50% (2x)", "75% (4x)", "90% (10x)", "95% (20x)", "98% (50x)"]
    x = range(len(sparsity_labels))

    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a"]
    markers = ["o", "s", "^", "D"]

    for idx, cond in enumerate(conditions_summary):
        accs = cond["mean_pruning_accs"]
        ax.plot(x, accs, marker=markers[idx % len(markers)], color=colors[idx % len(colors)],
                linewidth=2.2, markersize=7, label=f"{cond['cond_name']} (Unpruned: {accs[0]:.2f}%)")

    ax.set_xlabel("DCT Sparsity Level (Compression Ratio)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("v383: 2D-DCT Spectral Compression & Pruning Resistance", fontsize=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(sparsity_labels, fontsize=9.5)
    ax.set_ylim(10, 101)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9.5)

    fig.tight_layout()
    comp_path = os.path.join(figures_dir, "v383_dct_compression_curve.png")
    fig.savefig(comp_path, dpi=150)
    plt.close(fig)
    log(f"DCT compression curve plot saved to: {comp_path}")

# --- MAIN SUITE EXECUTION ---
def main():
    log("=" * 80)
    log("EXPERIMENT v383: 2D-DCT SPECTRAL COMPRESSION & HARMONIC PRUNING")
    log("=" * 80)

    commit_hash = get_git_commit()
    py_ver = platform.python_version()
    torch_ver = torch.__version__
    device_name = "cpu"
    device = torch.device(device_name)

    log(f"Platform: {platform.platform()} | Python: {py_ver} | PyTorch: {torch_ver} | Commit: {commit_hash}")
    log(f"Device: {device} (Ryzen CPU Vectorized Operations)")

    sample_model = TopoMLPv383()
    total_params = sum(p.numel() for p in sample_model.parameters())
    log("\n--- ARCHITECTURAL INVENTORY (3-Layer MLP) ---")
    log(f"Layer 1 (fc1): Linear(784 -> 256) | W1: [256, 784] ({784*256:,} params) + b1: [256] params")
    log(f"Layer 2 (fc2): Linear(256 -> 128) | W2: [128, 256] ({256*128:,} params) + b2: [128] params")
    log(f"Layer 3 (fc3): Linear(128 -> 10)  | W3: [10, 128]   ({128*10:,} params)  + b3: [10] params")
    log(f"Total Trainable Parameters: {total_params:,}")
    log("Compression Targets: W1 (256x784) and W2 (128x256) 2D-DCT coefficients\n")

    # Construct Orthonormal 2D-DCT transforms
    dct1 = OrthonormalDCT2D(256, 784, device)
    dct2 = OrthonormalDCT2D(128, 256, device)

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
        # Candidate 1: Turing Morphogenesis DoG (the low-rank star of v380, 90% energy in Rank 4)
        {
            "cond_name": "Turing_DoG_eps2e-3",
            "mode": "turing",
            "epsilon": 2e-3
        },
        # Candidate 2: Dirichlet 2D Laplacian (balanced smooth model, AdjCos = 0.91)
        {
            "cond_name": "Dirichlet_2D_eps2e-3",
            "mode": "dirichlet",
            "epsilon": 2e-3
        },
        # Candidate 3: Dirichlet 2D Laplacian (extreme smooth model, AdjCos = 0.97)
        {
            "cond_name": "Dirichlet_2D_eps1e-2",
            "mode": "dirichlet",
            "epsilon": 1e-2
        },
        # Reference Control: Standard AdamW (zero topography)
        {
            "cond_name": "Baseline_Standard_AdamW",
            "mode": "none",
            "epsilon": 0.0
        }
    ]

    # Sparsity sweep: 0%, 50% (2x), 75% (4x), 90% (10x), 95% (20x), 98% (50x)
    sparsity_levels = [0.0, 0.50, 0.75, 0.90, 0.95, 0.98]

    total_runs = len(test_suite) * len(seeds)
    total_steps = total_runs * epochs * len(train_loader)
    suite_progress = {
        "total_steps": total_steps,
        "completed_steps": 0
    }

    log(f"Test Suite: {len(test_suite)} conditions x {len(seeds)} seeds = {total_runs} training runs total.")
    log(f"Sparsity Sweep Levels: {[f'{int(s*100)}%' for s in sparsity_levels]}")
    log(f"Compression Factors: {[f'{round(1.0/max(1.0-s, 1e-4), 1)}x' for s in sparsity_levels]}\n")

    all_run_results = []
    conditions_summary = []

    for cond in test_suite:
        cond_name = cond["cond_name"]
        mode = cond["mode"]
        eps = cond["epsilon"]

        seed_pruning_runs = []
        c1_spectra = []
        indiv_accs = []
        adj_cosines = []

        for s in seeds:
            m, res = train_single_seed(
                cond_name=cond_name,
                mode=mode,
                epsilon=eps,
                seed=s,
                epochs=epochs,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                suite_progress=suite_progress
            )
            indiv_accs.append(res["val_acc"])
            adj_cosines.append(res["adj_cosine_w1"])

            # Evaluate 2D-DCT pruning
            p_res = evaluate_spectral_dct_pruning(
                base_model=m,
                sparsity_levels=sparsity_levels,
                test_loader=test_loader,
                device=device,
                dct1=dct1,
                dct2=dct2
            )
            seed_pruning_runs.append(p_res["pruning_results"])
            c1_spectra.append(p_res["c1_spectrum_sample"])
            res["dct_pruning"] = p_res["pruning_results"]
            all_run_results.append(res)

        # Average pruning accuracy across seeds for each sparsity level
        mean_accs_per_sparsity = []
        mean_energy_w1_per_sparsity = []
        for sp_idx in range(len(sparsity_levels)):
            accs_at_sp = [seed_pruning_runs[seed_idx][sp_idx]["test_acc"] for seed_idx in range(len(seeds))]
            e1_at_sp = [seed_pruning_runs[seed_idx][sp_idx]["energy_retained_w1_pct"] for seed_idx in range(len(seeds))]
            mean_accs_per_sparsity.append(round(sum(accs_at_sp) / len(accs_at_sp), 2))
            mean_energy_w1_per_sparsity.append(round(sum(e1_at_sp) / len(e1_at_sp), 2))

        summary_entry = {
            "cond_name": cond_name,
            "mode": mode,
            "epsilon": eps,
            "unpruned_val_acc": round(sum(indiv_accs) / len(indiv_accs), 2),
            "adj_cosine_w1": round(sum(adj_cosines) / len(adj_cosines), 4),
            "mean_pruning_accs": mean_accs_per_sparsity,
            "mean_energy_w1": mean_energy_w1_per_sparsity,
            "c1_spectrum_sample": c1_spectra[0] # Representative spectrum from Seed 42
        }
        conditions_summary.append(summary_entry)

    # --- REPORT TABLE ---
    log("\n" + "=" * 135)
    log(f"{'Condition':<26} | {'0% (1x)':<9} | {'50% (2x)':<9} | {'75% (4x)':<9} | {'90% (10x)':<10} | {'95% (20x)':<10} | {'98% (50x)':<10} | {'AdjCos':<8}")
    log("-" * 135)
    for c in conditions_summary:
        accs = c["mean_pruning_accs"]
        log(f"{c['cond_name']:<26} | "
            f"{accs[0]:>7.2f}% | "
            f"{accs[1]:>7.2f}% | "
            f"{accs[2]:>7.2f}% | "
            f"{accs[3]:>8.2f}% | "
            f"{accs[4]:>8.2f}% | "
            f"{accs[5]:>8.2f}% | "
            f"{c['adj_cosine_w1']:>8.4f}")
    log("=" * 135 + "\n")

    # Generate figures and persist
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
    generate_v383_artifacts(conditions_summary, figures_dir)

    for c in conditions_summary:
        c.pop("c1_spectrum_sample", None)

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "v383_spectral_dct_compression.json")

    payload = {
        "experiment_id": "v383",
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
            "model": "MLP (784 -> 256 -> 128 -> 10)",
            "sparsity_levels": sparsity_levels
        },
        "conditions_summary": conditions_summary,
        "individual_runs": all_run_results
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    log(f"Structured experimental record saved to: {json_path}")
    log("Execution of Experiment v383 completed successfully.")

if __name__ == "__main__":
    main()
