"""
Prototype v384: Spectral-LoRA (2D-DCT Low-Frequency Adapters) vs Low-Rank Adapters
Hypothesis:
  H1: Parameter adaptation is naturally concentrated in low spatial frequencies.
      Truncated 2D-DCT adapters (dW = D_out^T @ C @ D_in with k x k trainable params)
      achieve competitive transfer accuracy with 10x - 40x fewer adapter parameters than LoRA.
  H2: A base model pre-trained with Dirichlet Topographic Cellular Regularization (TCR)
      provides an aligned, smooth spectral foundation that transfers more effectively under
      spectral modulation than an unstructured (white-noise) standard baseline.
  H3: Truncating high-frequency weight updates acts as an implicit regularizer,
      mitigating overfitting during fine-tuning.

Rigour Level: Level 1 (Exploratory multi-seed transfer study, 3 seeds per condition)
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
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
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
    """
    Computes the orthonormal DCT-II matrix D of shape (N, N).
    D @ D.T = I_N
    Row k of D is the k-th DCT basis vector.
    """
    n = torch.arange(N, device=device).unsqueeze(1)
    k = torch.arange(N, device=device).unsqueeze(0)
    D = torch.cos(math.pi * (n + 0.5) * k / N)
    D[:, 0] *= 1.0 / math.sqrt(N)
    D[:, 1:] *= math.sqrt(2.0 / N)
    return D.T  # Shape: (N, N) where D[k, :] is k-th basis vector

# ---------------------------------------------------------
# Regularization: Dirichlet 2D
# ---------------------------------------------------------
def dirichlet_energy_2d(W):
    diff_i = W[1:, :] - W[:-1, :]
    diff_j = W[:, 1:] - W[:, :-1]
    return torch.sum(diff_i ** 2) + torch.sum(diff_j ** 2)

# ---------------------------------------------------------
# Base Model (Pretraining on Fashion-MNIST)
# ---------------------------------------------------------
class BaseMLP(nn.Module):
    def __init__(self, d_in=784, d_h1=256, d_h2=128, d_out=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h1)
        self.fc2 = nn.Linear(d_h1, d_h2)
        self.fc3 = nn.Linear(d_h2, d_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.fc3(h2)
        return out

# ---------------------------------------------------------
# Adapters for Fine-Tuning
# ---------------------------------------------------------
class SpectralAdapter(nn.Module):
    """
    Spectral 2D-DCT Adapter:
    dW = D_out[:k_out, :]^T @ C @ D_in[:k_in, :]
    delta_y = ((x @ U_in) @ C.T) @ U_out * alpha
    Parameters: exactly k_out * k_in floats!
    """
    def __init__(self, in_features, out_features, k_in=16, k_out=16, alpha=1.0, device="cpu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k_in = k_in
        self.k_out = k_out
        self.alpha = alpha

        Din = get_dct_basis(in_features, device=device)
        Dout = get_dct_basis(out_features, device=device)

        # U_in shape: (in_features, k_in)
        self.register_buffer("U_in", Din[:k_in, :].T.contiguous())
        # U_out shape: (k_out, out_features)
        self.register_buffer("U_out", Dout[:k_out, :].contiguous())

        # Trainable coupling tensor initialized to 0
        self.C = nn.Parameter(torch.zeros(k_out, k_in, device=device))

    def forward(self, x):
        # x: (B, in_features)
        # x @ U_in: (B, k_in)
        # @ C.T: (B, k_out)
        # @ U_out: (B, out_features)
        return ((x @ self.U_in) @ self.C.T) @ self.U_out * self.alpha

    def num_adapter_params(self):
        return self.k_in * self.k_out


class LoRAAdapter(nn.Module):
    """
    Standard Low-Rank Adapter (LoRA):
    dW = B @ A * (alpha / r)
    Parameters: r * (in_features + out_features)
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, device="cpu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.A = nn.Parameter(torch.randn(rank, in_features, device=device) / math.sqrt(rank))
        self.B = nn.Parameter(torch.zeros(out_features, rank, device=device))

    def forward(self, x):
        # x: (B, in_features)
        # x @ A.T: (B, rank)
        # @ B.T: (B, out_features)
        return (x @ self.A.T) @ self.B.T * self.scaling

    def num_adapter_params(self):
        return self.rank * (self.in_features + self.out_features)


class AdaptedMLP(nn.Module):
    """
    Fine-tuning network:
    Backbone weights (w1, b1, w2, b2) are FROZEN.
    Optional adapter applied to fc1 and fc2.
    Classification head (fc3) is re-initialized and trainable.
    """
    def __init__(self, base_w1, base_b1, base_w2, base_b2, adapter_type="none", **adapter_kwargs):
        super().__init__()
        device = base_w1.device
        self.w1 = nn.Parameter(base_w1.clone().detach(), requires_grad=False)
        self.b1 = nn.Parameter(base_b1.clone().detach(), requires_grad=False)
        self.w2 = nn.Parameter(base_w2.clone().detach(), requires_grad=False)
        self.b2 = nn.Parameter(base_b2.clone().detach(), requires_grad=False)

        self.adapter_type = adapter_type
        if adapter_type == "spectral":
            k = adapter_kwargs.get("k", 16)
            self.adapt1 = SpectralAdapter(784, 256, k_in=k, k_out=k, device=device)
            self.adapt2 = SpectralAdapter(256, 128, k_in=k, k_out=k, device=device)
        elif adapter_type == "lora":
            rank = adapter_kwargs.get("rank", 4)
            self.adapt1 = LoRAAdapter(784, 256, rank=rank, device=device)
            self.adapt2 = LoRAAdapter(256, 128, rank=rank, device=device)
        else:
            self.adapt1 = None
            self.adapt2 = None

        # Trainable readout head for target task
        self.head = nn.Linear(128, 10, device=device)

    def forward(self, x):
        h1 = x @ self.w1.T + self.b1
        if self.adapt1 is not None:
            h1 = h1 + self.adapt1(x)
        h1 = F.relu(h1)

        h2 = h1 @ self.w2.T + self.b2
        if self.adapt2 is not None:
            h2 = h2 + self.adapt2(h1)
        h2 = F.relu(h2)

        out = self.head(h2)
        return out

    def get_adapter_params_count(self):
        count = 0
        if self.adapt1 is not None:
            count += self.adapt1.num_adapter_params()
        if self.adapt2 is not None:
            count += self.adapt2.num_adapter_params()
        return count

    def get_trainable_params_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ---------------------------------------------------------
# Training and Evaluation Helpers
# ---------------------------------------------------------
def evaluate(model, loader, device="cpu"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.view(x.size(0), -1).to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / total, 100.0 * correct / total

def pretrain_base_model(seed, mode="dirichlet", epsilon=0.01, epochs=5, lr=1e-3, batch_size=256, device="cpu"):
    """
    Pretrains BaseMLP on Fashion-MNIST (60k) for given seed.
    Returns (w1, b1, w2, b2) detached.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.FashionMNIST("./data", train=True, download=False, transform=transform)
    test_ds = torchvision.datasets.FashionMNIST("./data", train=False, download=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = BaseMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    log(f"--- Pretraining Base Model on Fashion-MNIST [Seed {seed}, Mode: {mode}, eps: {epsilon}] ---")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for b_idx, (x, y) in enumerate(train_loader):
            x, y = x.view(x.size(0), -1).to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            task_loss = criterion(logits, y)

            if mode == "dirichlet" and epsilon > 0.0:
                reg = dirichlet_energy_2d(model.fc1.weight) + dirichlet_energy_2d(model.fc2.weight)
                loss = task_loss + epsilon * reg
            else:
                loss = task_loss

            loss.backward()
            optimizer.step()

            # Fast feedback on first 5 batches of Epoch 1
            if ep == 1 and b_idx < 5:
                log(f"[Fast Feedback Pretrain] Ep 1, Batch {b_idx + 1}/5: TaskLoss = {task_loss.item():.4f}")

        val_loss, val_acc = evaluate(model, test_loader, device=device)
        log(f"Pretrain Ep {ep}/{epochs} - Test Loss: {val_loss:.4f} | Test Acc: {val_acc:.2f}%")

    elapsed = time.time() - t0
    log(f"Pretraining Finished in {elapsed:.1f}s. Final Fashion-MNIST Test Acc: {val_acc:.2f}%")

    return (model.fc1.weight.detach().clone(),
            model.fc1.bias.detach().clone(),
            model.fc2.weight.detach().clone(),
            model.fc2.bias.detach().clone(),
            val_acc)

def finetune_target(base_weights, cond_cfg, seed, epochs=5, lr=2e-3, batch_size=256, device="cpu"):
    """
    Fine-tunes AdaptedMLP on MNIST (60k) with frozen base backbone.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST("./data", train=True, download=False, transform=transform)
    test_ds = torchvision.datasets.MNIST("./data", train=False, download=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    w1, b1, w2, b2 = base_weights
    adapter_type = cond_cfg["adapter_type"]
    adapter_kwargs = cond_cfg.get("kwargs", {})

    model = AdaptedMLP(w1, b1, w2, b2, adapter_type=adapter_type, **adapter_kwargs).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    adapter_params = model.get_adapter_params_count()
    total_trainable = model.get_trainable_params_count()

    log(f"  Finetune Start [{cond_cfg['name']}, Seed {seed}]: AdapterParams = {adapter_params}, Trainable = {total_trainable}")

    epoch_history = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for b_idx, (x, y) in enumerate(train_loader):
            x, y = x.view(x.size(0), -1).to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            # Fast feedback on first 5 batches of Epoch 1
            if ep == 1 and b_idx < 5:
                log(f"  [Fast Feedback Finetune] Ep 1, Batch {b_idx + 1}/5: Loss = {loss.item():.4f}")

        val_loss, val_acc = evaluate(model, test_loader, device=device)
        epoch_history.append({"epoch": ep, "test_loss": val_loss, "test_acc": val_acc})
        log(f"  Finetune Ep {ep}/{epochs} - Test Loss: {val_loss:.4f} | Test Acc: {val_acc:.2f}%")

    wall_time = time.time() - t0
    final_acc = epoch_history[-1]["test_acc"]
    final_loss = epoch_history[-1]["test_loss"]

    return {
        "final_test_acc": final_acc,
        "final_test_loss": final_loss,
        "adapter_params": adapter_params,
        "total_trainable_params": total_trainable,
        "wall_time": wall_time,
        "epoch_history": epoch_history
    }

# ---------------------------------------------------------
# Main Benchmark Suite
# ---------------------------------------------------------
def main():
    log("=========================================================================================")
    log(" EXPERIMENT v384: Spectral-LoRA (2D-DCT Low-Frequency Adapters) vs Low-Rank Adapters")
    log("=========================================================================================")
    log(f"Commit: {get_git_commit()} | Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    log(f"Platform: {platform.platform()} | Device: CPU")

    SEEDS = [42, 100, 2026]
    PRETRAIN_EPOCHS = 5
    FINETUNE_EPOCHS = 5
    BATCH_SIZE = 256
    LR_PRETRAIN = 1e-3
    LR_FINETUNE = 2e-3
    DIRICHLET_EPS = 0.01

    # -----------------------------------------------------
    # Conditions definition (Regla de Oro: Primero el Candidato)
    # -----------------------------------------------------
    conditions = [
        # Candidate 1: Spectral Adapter k=16 on Dirichlet Base (512 adapter params)
        {
            "name": "Spectral_k16_DirichletBase",
            "base_type": "dirichlet",
            "adapter_type": "spectral",
            "kwargs": {"k": 16},
            "desc": "Spectral-LoRA (k=16, 512 params) on Dirichlet-regularized base"
        },
        # Candidate 2: Spectral Adapter k=8 on Dirichlet Base (128 adapter params)
        {
            "name": "Spectral_k8_DirichletBase",
            "base_type": "dirichlet",
            "adapter_type": "spectral",
            "kwargs": {"k": 8},
            "desc": "Spectral-LoRA (k=8, 128 params) on Dirichlet-regularized base"
        },
        # Candidate 3: Spectral Adapter k=24 on Dirichlet Base (1152 adapter params)
        {
            "name": "Spectral_k24_DirichletBase",
            "base_type": "dirichlet",
            "adapter_type": "spectral",
            "kwargs": {"k": 24},
            "desc": "Spectral-LoRA (k=24, 1152 params) on Dirichlet-regularized base"
        },
        # Candidate 4: Spectral Adapter k=16 on Standard Base (to test base dependence)
        {
            "name": "Spectral_k16_StandardBase",
            "base_type": "standard",
            "adapter_type": "spectral",
            "kwargs": {"k": 16},
            "desc": "Spectral-LoRA (k=16, 512 params) on Standard AdamW base"
        },
        # Baseline 1: Standard LoRA rank 4 on Dirichlet Base (5696 adapter params)
        {
            "name": "LoRA_r4_DirichletBase",
            "base_type": "dirichlet",
            "adapter_type": "lora",
            "kwargs": {"rank": 4},
            "desc": "LoRA (r=4, 5696 params) on Dirichlet-regularized base"
        },
        # Baseline 2: Standard LoRA rank 4 on Standard Base (5696 adapter params)
        {
            "name": "LoRA_r4_StandardBase",
            "base_type": "standard",
            "adapter_type": "lora",
            "kwargs": {"rank": 4},
            "desc": "LoRA (r=4, 5696 params) on Standard AdamW base"
        },
        # Baseline 3: Linear Probe only on Dirichlet Base (0 adapter params)
        {
            "name": "LinearProbe_DirichletBase",
            "base_type": "dirichlet",
            "adapter_type": "none",
            "kwargs": {},
            "desc": "Linear Probe (0 adapter params, frozen backbone) on Dirichlet base"
        },
        # Baseline 4: Linear Probe only on Standard Base (0 adapter params)
        {
            "name": "LinearProbe_StandardBase",
            "base_type": "standard",
            "adapter_type": "none",
            "kwargs": {},
            "desc": "Linear Probe (0 adapter params, frozen backbone) on Standard base"
        }
    ]

    log(f"Total Conditions: {len(conditions)} | Seeds per Condition: {len(SEEDS)}")
    log("Architecture Inventory:")
    log("  Backbone: 784 -> 256 (fc1) -> 128 (fc2) -> 10 (fc3)")
    log("  Pretraining Task: Fashion-MNIST (60k train / 10k test)")
    log("  Fine-tuning Task: MNIST (60k train / 10k test) with FROZEN backbone")

    # -----------------------------------------------------
    # Phase 1: Pre-training Base Models (Cached per seed)
    # -----------------------------------------------------
    log("\n=========================================================================================")
    log(" PHASE 1: PRE-TRAINING BASE MODELS (Fashion-MNIST)")
    log("=========================================================================================")

    pretrained_bases = {"dirichlet": {}, "standard": {}}

    for seed in SEEDS:
        # Dirichlet Base
        w1_d, b1_d, w2_d, b2_d, acc_d = pretrain_base_model(
            seed=seed, mode="dirichlet", epsilon=DIRICHLET_EPS,
            epochs=PRETRAIN_EPOCHS, lr=LR_PRETRAIN, batch_size=BATCH_SIZE
        )
        pretrained_bases["dirichlet"][seed] = (w1_d, b1_d, w2_d, b2_d)

        # Standard Base
        w1_s, b1_s, w2_s, b2_s, acc_s = pretrain_base_model(
            seed=seed, mode="standard", epsilon=0.0,
            epochs=PRETRAIN_EPOCHS, lr=LR_PRETRAIN, batch_size=BATCH_SIZE
        )
        pretrained_bases["standard"][seed] = (w1_s, b1_s, w2_s, b2_s)

    # -----------------------------------------------------
    # Phase 2: Fine-Tuning Suite (MNIST)
    # -----------------------------------------------------
    log("\n=========================================================================================")
    log(" PHASE 2: ADAPTATION BENCHMARK (MNIST Transfer with Frozen Backbone)")
    log("=========================================================================================")

    all_results = []
    total_runs = len(conditions) * len(SEEDS)
    current_run = 0

    for cond in conditions:
        cond_name = cond["name"]
        base_type = cond["base_type"]
        log(f"\n>>> Running Condition: {cond_name} ({cond['desc']}) <<<")

        seed_runs = []
        for seed in SEEDS:
            current_run += 1
            progress_pct = (current_run / total_runs) * 100.0
            log(f"[Progress: {progress_pct:.1f}% | Run {current_run}/{total_runs}] Condition: {cond_name} | Seed: {seed}")

            base_weights = pretrained_bases[base_type][seed]
            res = finetune_target(
                base_weights=base_weights,
                cond_cfg=cond,
                seed=seed,
                epochs=FINETUNE_EPOCHS,
                lr=LR_FINETUNE,
                batch_size=BATCH_SIZE
            )
            res["seed"] = seed
            res["cond_name"] = cond_name
            res["base_type"] = base_type
            res["adapter_type"] = cond["adapter_type"]
            seed_runs.append(res)

        all_results.append({
            "cond_name": cond_name,
            "cond_cfg": cond,
            "runs": seed_runs
        })

    # -----------------------------------------------------
    # Phase 3: Analysis and Aggregation
    # -----------------------------------------------------
    log("\n=========================================================================================")
    log(" PHASE 3: CONSOLIDATED RESULTS & PARAMETRIC EFFICIENCY INDEX")
    log("=========================================================================================")

    summary_rows = []
    log(f"{'Condition':<30} | {'Adapter Params':<14} | {'Test Acc (%)':<16} | {'Final Loss':<12} | {'Time (s)':<8}")
    log("-" * 90)

    for cond_res in all_results:
        cname = cond_res["cond_name"]
        runs = cond_res["runs"]
        accs = [r["final_test_acc"] for r in runs]
        losses = [r["final_test_loss"] for r in runs]
        times = [r["wall_time"] for r in runs]
        adapt_p = runs[0]["adapter_params"]

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        mean_loss = float(np.mean(losses))
        mean_time = float(np.mean(times))

        summary_rows.append({
            "cond_name": cname,
            "adapter_params": adapt_p,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "mean_loss": mean_loss,
            "mean_time": mean_time,
            "all_accs": accs
        })

        log(f"{cname:<30} | {adapt_p:<14d} | {mean_acc:6.2f} +/- {std_acc:4.2f} | {mean_loss:10.4f} | {mean_time:6.2f}s")

    # -----------------------------------------------------
    # Phase 4: Artifact Generation (Figures & JSON)
    # -----------------------------------------------------
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    json_path = "results/raw/v384_spectral_finetuning.json"
    raw_payload = {
        "experiment_id": "v384",
        "date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "commit_hash": get_git_commit(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": "cpu",
        "hyperparameters": {
            "pretrain_epochs": PRETRAIN_EPOCHS,
            "finetune_epochs": FINETUNE_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr_pretrain": LR_PRETRAIN,
            "lr_finetune": LR_FINETUNE,
            "dirichlet_eps": DIRICHLET_EPS,
            "seeds": SEEDS
        },
        "summary": summary_rows,
        "detailed_results": all_results
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_payload, f, indent=2)
    log(f"\nRaw results saved to {json_path}")

    # Plot Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Adapter Params vs Accuracy (Pareto efficiency)
    names = [s["cond_name"] for s in summary_rows]
    params = [max(1, s["adapter_params"]) for s in summary_rows]
    mean_accs = [s["mean_acc"] for s in summary_rows]
    std_accs = [s["std_acc"] for s in summary_rows]

    colors = []
    for s in summary_rows:
        if "Spectral" in s["cond_name"]:
            colors.append("#2ecc71" if "Dirichlet" in s["cond_name"] else "#1abc9c")
        elif "LoRA" in s["cond_name"]:
            colors.append("#e74c3c" if "Dirichlet" in s["cond_name"] else "#e67e22")
        else:
            colors.append("#95a5a6")

    ax1.errorbar(params, mean_accs, yerr=std_accs, fmt="o", capsize=4, color="#34495e", zorder=2)
    for p, acc, name, c in zip(params, mean_accs, names, colors):
        ax1.scatter([p], [acc], color=c, s=120, zorder=3, edgecolors="black", linewidth=1.2)
        offset_y = 0.2 if "Dirichlet" in name else -0.3
        ax1.annotate(name.replace("_", " "), (p, acc), textcoords="offset points",
                     xytext=(0, 8 if offset_y > 0 else -14), ha="center", fontsize=8, weight="bold")

    ax1.set_xscale("log")
    ax1.set_title("Pareto Efficiency: Accuracy vs Adapter Parameters", fontsize=12, weight="bold")
    ax1.set_xlabel("Adapter Parameters (log scale)", fontsize=10)
    ax1.set_ylabel("MNIST Test Accuracy (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Bar Comparison with Error Bars
    y_pos = np.arange(len(names))
    ax2.barh(y_pos, mean_accs, xerr=std_accs, color=colors, edgecolor="black", alpha=0.85, capsize=4)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([n.replace("_", " ") for n in names], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlim(min(mean_accs) - 2.0, 100.0)
    ax2.set_xlabel("MNIST Test Accuracy (%)", fontsize=10)
    ax2.set_title("Adapter Comparison (Mean +/- Std across 3 seeds)", fontsize=12, weight="bold")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.5)

    for i, (m, sd, p) in enumerate(zip(mean_accs, std_accs, params)):
        ax2.text(m + 0.3, i, f"{m:.2f}% (p={p})", va="center", fontsize=8, weight="bold")

    plt.tight_layout()
    fig_path = "results/figures/v384_spectral_finetuning_curves.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    log(f"Figure saved to {fig_path}")

    log("\n=========================================================================================")
    log(" EXPERIMENT v384 FINISHED SUCCESSFULLY")
    log("=========================================================================================")

if __name__ == "__main__":
    main()
