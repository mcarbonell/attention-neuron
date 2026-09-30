#!/usr/bin/env python3
"""
================================================================================
EXPERIMENT v380: UNATTENUATED MACROSCOPIC TOPOGRAPHIC CELLULAR REGULARIZATION
================================================================================
Hypothesis & Mathematical Reconcilation:
  In v379, the spatial regularizer was normalized with .mean(), which introduced
  an inadvertent 1/N attenuation factor (1 / 200,704), reducing regularizer
  gradients to ~10^-9 (a factor of 10^6 weaker than the task gradient ~10^-3).

  In v380, we test the UNATTENUATED formulation directly derived from Eq. 16
  of docs/proposal_cellular_weight_regularization.md:
    R_dirichlet(W) = (epsilon / 2) * [ sum((Delta_r W)^2) + sum((Delta_c W)^2) ]
  yielding:
    -grad_W R = epsilon * Delta W (Laplacian heat diffusion on every element).

  With this formulation:
    - epsilon = 5e-4: grad_R ~ 1e-4 (~10% of task gradient)
    - epsilon = 2e-3: grad_R ~ 4e-4 (~30% of task gradient)
    - epsilon = 1e-2: grad_R ~ 2e-3 (~150% of task gradient, dominant regime)

  We test whether this unattenuated coupling crosses the phase transition,
  forcing:
    1. Low-rank spectral collapse (SVD effective rank drop on W1).
    2. Cortical topography (adjacent neuron cosine similarity AdjCos >> 0).
    3. Permutation gauge symmetry breaking across independent seeds (rho_raw >> 0).
    4. Generalization / accuracy trade-off Pareto boundary.

Compliance Checklist (GEMINI.md):
  1. No modification of existing scripts (new file prototype_v380_*.py).
  2. Candidates evaluated FIRST, Baseline LAST.
  3. Strict vectorization (zero python for-loops in tensor forward/penalty).
  4. Fast feedback in first 5 batches of Epoch 1.
  5. Timestamps [+HH:MM:SS.ss] with flush=True on all logs.
  6. Multi-seed evaluation (3 independent seeds: 42, 100, 2026).
  7. Full config, architecture inventory, ETA monitoring, and structured JSON output
     in results/raw/v380_macroscopic_cellular_weights.json.
  8. Heatmaps and SVD curves persisted in results/figures/.
================================================================================
"""

import os
import sys
import time
import json
import math
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

# --- UNATTENUATED TOPOGRAPHIC REGULARIZER MODULE ---
class UnattenuatedTopographicRegularizer(nn.Module):
    """
    Unattenuated spatial regularizer where penalty is summed (not divided by matrix size N),
    ensuring that the local spatial gradient on each weight has magnitude ~ epsilon * (Delta W).
    """
    def __init__(self, mode: str = "none", epsilon: float = 0.0):
        super().__init__()
        self.mode = mode
        self.epsilon = epsilon
        
        if mode == "turing":
            # 5x5 Mexican hat / Difference of Gaussians (DoG)
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
            # Unattenuated Dirichlet energy: 0.5 * epsilon * sum(||grad W||^2)
            # Yields -grad_W R = epsilon * Delta W on every coordinate
            diff_r = weight[1:, :] - weight[:-1, :]
            diff_c = weight[:, 1:] - weight[:, :-1]
            return self.epsilon * 0.5 * (diff_r.pow(2).sum() + diff_c.pow(2).sum())

        elif self.mode == "cortical_1d":
            # Smoothness strictly along neuron axis (rows)
            diff_r = weight[1:, :] - weight[:-1, :]
            return self.epsilon * 0.5 * diff_r.pow(2).sum()

        elif self.mode == "turing":
            # Bounded alignment with Mexican Hat DoG filter
            # To ensure stability while exerting macro force, we align the normalized inner product
            # scaled by total Frobenius norm: - epsilon * sum(W * (K * W)) / (||K * W|| + 1e-6)
            w_img = weight.unsqueeze(0).unsqueeze(0)
            pad = self.kernel.shape[-1] // 2
            filtered = F.conv2d(w_img, self.kernel, padding=pad).squeeze(0).squeeze(0)
            norm_f = filtered.norm(p="fro").clamp(min=1e-6)
            # Alignment energy yielding directional gradient towards Turing pattern
            alignment = - (weight * filtered).sum() / norm_f
            return self.epsilon * alignment

        return torch.tensor(0.0, device=weight.device)

# --- MODEL ARCHITECTURE ---
class TopoMLPv380(nn.Module):
    """
    3-Layer Multi-Layer Perceptron (784 -> 256 -> 128 -> 10)
    Regularization is applied to representation layers W1 and W2.
    """
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

# --- DIAGNOSTIC & SPECTRAL METRICS ---
def compute_svd_metrics(weight: torch.Tensor) -> Dict[str, any]:
    with torch.no_grad():
        w = weight.detach().float()
        U, S, V = torch.linalg.svd(w, full_matrices=False)
        s_sum = S.sum().item()
        s_norm = S / (s_sum + 1e-12)
        s_sq = S.pow(2)
        s_sq_sum = s_sq.sum().item()
        cum_energy = torch.cumsum(s_sq / (s_sq_sum + 1e-12), dim=0)

        # Shannon entropy based effective rank
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

def compute_cross_seed_alignment(weights_list: List[torch.Tensor]) -> Dict[str, float]:
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

# --- TRAINING & EVALUATION LOOP ---
def train_and_eval_single_seed(
    cond_name: str,
    mode: str,
    epsilon: float,
    seed: int,
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    suite_progress: Dict[str, any]
) -> Dict[str, any]:
    torch.manual_seed(seed)
    model = TopoMLPv380(mode=mode, epsilon=epsilon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    t_start = time.time()
    step_count = 0
    t_overhead = 0.0

    log(f"--- Launching: {cond_name} | Seed: {seed} | Mode: {mode} | Eps: {epsilon} ---")

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

        # Evaluate on validation/test set
        model.eval()
        va_loss = 0.0
        va_correct = 0
        va_total = 0
        with torch.no_grad():
            for vx, vy in test_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_logits = model(vx)
                va_loss += criterion(v_logits, vy).item() * len(vy)
                v_preds = v_logits.argmax(dim=-1)
                va_correct += (v_preds == vy).sum().item()
                va_total += len(vy)

        val_loss = va_loss / va_total
        val_acc = (va_correct / va_total) * 100.0

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

    # Final diagnostics
    w1 = model.fc1.weight.data.cpu()
    w2 = model.fc2.weight.data.cpu()
    svd_w1 = compute_svd_metrics(w1)
    cortical_w1 = compute_cortical_adjacency(w1)

    return {
        "cond_name": cond_name,
        "mode": mode,
        "epsilon": epsilon,
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

# --- PLOTTING / FIGURE GENERATION ---
def generate_artifacts_and_plots(conditions_summary: List[Dict[str, any]], figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Weight Heatmaps Figure
    fig, axes = plt.subplots(1, len(conditions_summary), figsize=(3.8 * len(conditions_summary), 4), squeeze=False)
    for idx, cond in enumerate(conditions_summary):
        ax = axes[0, idx]
        w_sample = cond["representative_w1"][:64, :64].numpy()
        im = ax.imshow(w_sample, cmap="coolwarm", aspect="auto")
        ax.set_title(f"{cond['cond_name']}\n(eps={cond['epsilon']})", fontsize=9)
        ax.set_xlabel("Inputs (Patch 0..64)")
        if idx == 0:
            ax.set_ylabel("Neurons (Patch 0..64)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    heatmap_path = os.path.join(figures_dir, "v380_weight_heatmaps.png")
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    log(f"Weight heatmaps saved to: {heatmap_path}")

    # 2. SVD Singular Decay Curves Figure
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for cond in conditions_summary:
        sing_vals = cond["mean_svd_w1"]["singular_values_top10"]
        top_v = sing_vals[0] if sing_vals[0] > 0 else 1.0
        decay = [v / top_v for v in sing_vals]
        ax.plot(range(1, len(decay) + 1), decay, marker="o", label=f"{cond['cond_name']} (EffRank: {cond['mean_svd_w1']['effective_rank']:.1f})")

    ax.set_xlabel("Singular Value Index (k)")
    ax.set_ylabel("Normalized Singular Value (sigma_k / sigma_1)")
    ax.set_title("v380: Unattenuated SVD Spectral Decay on W1", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    svd_path = os.path.join(figures_dir, "v380_svd_spectrum.png")
    fig.savefig(svd_path, dpi=150)
    plt.close(fig)
    log(f"SVD decay plot saved to: {svd_path}")

# --- MAIN SUITE EXECUTION ---
def main():
    log("=" * 80)
    log("EXPERIMENT v380: UNATTENUATED MACROSCOPIC CELLULAR WEIGHT REGULARIZATION")
    log("=" * 80)
    
    commit_hash = get_git_commit()
    py_ver = platform.python_version()
    torch_ver = torch.__version__
    device_name = "cpu"
    device = torch.device(device_name)
    
    log(f"Platform: {platform.platform()} | Python: {py_ver} | PyTorch: {torch_ver} | Commit: {commit_hash}")
    log(f"Device: {device} (Ryzen CPU Vectorized Operations)")

    # Architecture Inventory
    sample_model = TopoMLPv380()
    total_params = sum(p.numel() for p in sample_model.parameters())
    log("\n--- ARCHITECTURAL INVENTORY (3-Layer MLP) ---")
    log(f"Layer 1 (fc1): Linear(784 -> 256) | W1: [256, 784] ({784*256:,} params) + b1: [256] params")
    log(f"Layer 2 (fc2): Linear(256 -> 128) | W2: [128, 256] ({256*128:,} params) + b2: [128] params")
    log(f"Layer 3 (fc3): Linear(128 -> 10)  | W3: [10, 128]   ({128*10:,} params)  + b3: [10] params")
    log(f"Total Trainable Parameters: {total_params:,}")
    log("Regularization Targets: W1 and W2 (Unattenuated Sum Formulation)\n")

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
        # Candidate 1: Dirichlet 2D Laplacian (Sweep across unattenuated force)
        {"cond_name": "Dirichlet_2D_eps1e-2", "mode": "dirichlet", "epsilon": 1e-2},
        {"cond_name": "Dirichlet_2D_eps2e-3", "mode": "dirichlet", "epsilon": 2e-3},
        {"cond_name": "Dirichlet_2D_eps5e-4", "mode": "dirichlet", "epsilon": 5e-4},
        
        # Candidate 2: Cortical 1D Neuron (Neuron axis coupling)
        {"cond_name": "Cortical_1D_eps2e-3",  "mode": "cortical_1d", "epsilon": 2e-3},
        
        # Candidate 3: Turing Morphogenesis DoG
        {"cond_name": "Turing_DoG_eps2e-3",   "mode": "turing", "epsilon": 2e-3},
        
        # Reference Control: Standard AdamW (epsilon = 0.0)
        {"cond_name": "Baseline_Standard_AdamW", "mode": "none", "epsilon": 0.0}
    ]

    total_runs = len(test_suite) * len(seeds)
    total_steps = total_runs * epochs * len(train_loader)
    suite_progress = {
        "total_steps": total_steps,
        "completed_steps": 0
    }

    log(f"Test Suite: {len(test_suite)} conditions x {len(seeds)} seeds = {total_runs} training runs total.")
    log(f"Total Step Budget: {total_steps:,} steps across all runs.\n")

    all_run_results = []
    conditions_summary = []

    for cond in test_suite:
        cond_name = cond["cond_name"]
        mode = cond["mode"]
        eps = cond["epsilon"]
        
        seed_runs = []
        w1_seeds = []
        w2_seeds = []

        for s in seeds:
            res = train_and_eval_single_seed(
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
            w1_seeds.append(res.pop("w1_tensor"))
            w2_seeds.append(res.pop("w2_tensor"))
            seed_runs.append(res)
            all_run_results.append(res)

        align_w1 = compute_cross_seed_alignment(w1_seeds)
        align_w2 = compute_cross_seed_alignment(w2_seeds)

        val_accs = [r["val_acc"] for r in seed_runs]
        val_losses = [r["val_loss"] for r in seed_runs]
        eff_ranks = [r["svd_w1"]["effective_rank"] for r in seed_runs]
        adj_cosines = [r["cortical_w1"]["mean_adj_cosine"] for r in seed_runs]
        wall_times = [r["wall_clock_time"] for r in seed_runs]
        reg_overheads = [r["reg_overhead_time"] for r in seed_runs]

        rep_w1 = w1_seeds[0]

        summary_entry = {
            "cond_name": cond_name,
            "mode": mode,
            "epsilon": eps,
            "mean_val_acc": round(sum(val_accs) / len(val_accs), 2),
            "std_val_acc": round(float(torch.tensor(val_accs).std().item()), 2),
            "mean_val_loss": round(sum(val_losses) / len(val_losses), 4),
            "mean_eff_rank_w1": round(sum(eff_ranks) / len(eff_ranks), 2),
            "mean_adj_cosine_w1": round(sum(adj_cosines) / len(adj_cosines), 4),
            "cross_seed_rho_raw_w1": align_w1["rho_raw_mean"],
            "cross_seed_rho_raw_w2": align_w2["rho_raw_mean"],
            "mean_wall_time_s": round(sum(wall_times) / len(wall_times), 2),
            "mean_overhead_s": round(sum(reg_overheads) / len(reg_overheads), 4),
            "representative_w1": rep_w1,
            "mean_svd_w1": seed_runs[0]["svd_w1"]
        }
        conditions_summary.append(summary_entry)

    # --- REPORT TABLE ---
    log("\n" + "=" * 115)
    log(f"{'Condition':<30} | {'Mode':<10} | {'Eps':<7} | {'ValAcc (%)':<12} | {'EffRank W1':<10} | {'AdjCos (H2)':<11} | {'rho_raw (H1)':<12} | {'Time (s)':<8}")
    log("-" * 115)
    for c in conditions_summary:
        log(f"{c['cond_name']:<30} | {c['mode']:<10} | {c['epsilon']:<7.0e} | "
            f"{c['mean_val_acc']:>5.2f} +/- {c['std_val_acc']:<4.2f} | "
            f"{c['mean_eff_rank_w1']:>10.2f} | {c['mean_adj_cosine_w1']:>11.4f} | "
            f"{c['cross_seed_rho_raw_w1']:>12.4f} | {c['mean_wall_time_s']:>8.2f}")
    log("=" * 115 + "\n")

    # Generate figures and persist
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
    generate_artifacts_and_plots(conditions_summary, figures_dir)

    for c in conditions_summary:
        c.pop("representative_w1", None)

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "v380_macroscopic_cellular_weights.json")

    payload = {
        "experiment_id": "v380",
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
    log("Execution of Experiment v380 completed successfully.")

if __name__ == "__main__":
    main()
