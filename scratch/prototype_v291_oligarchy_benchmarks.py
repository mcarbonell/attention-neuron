import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import argparse
import time

# Attempt to load torch-directml for AMD GPU support as per user memory
try:
    import torch_directml
    dml_device = torch_directml.device()
    print(f"[Device Selection] Found AMD DirectML device: {dml_device}")
except ImportError:
    dml_device = None

def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if dml_device is not None:
        return dml_device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

# ══════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════

class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features, init_val=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Trainable gate parameters
        self.gate = nn.Parameter(torch.full((out_features,), init_val))
        # Keep the projection weights frozen (the backbone)
        for param in self.linear.parameters():
            param.requires_grad = False
                
    def forward(self, x):
        return self.linear(x) * self.gate

class GatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, init_val=1.0, activation='silu'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.SiLU() if activation == 'silu' else nn.ReLU()
        
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(GatedLinear(prev_dim, h_dim, init_val))
            prev_dim = h_dim
        
        # Last layer maps to outputs (no gating/always 1.0 gating)
        self.layers.append(GatedLinear(prev_dim, output_dim, 1.0))
        
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

def calculate_pr(gate_tensor):
    """
    Participation Ratio (PR) to measure the Effective Number of Gates:
    N_eff = (sum(|g|))^2 / sum(g^2)
    """
    abs_g = torch.abs(gate_tensor)
    sum_abs = torch.sum(abs_g).item()
    sum_sq = torch.sum(gate_tensor**2).item()
    if sum_sq == 0: 
        return 0
    return (sum_abs ** 2) / sum_sq

# ══════════════════════════════════════════════════════════════════════
# TRAINING AND EVALUATION HARNESS
# ══════════════════════════════════════════════════════════════════════

def evaluate(model, loader, device, input_flat_dim):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.view(-1, input_flat_dim).to(device)
            target = target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return 100. * correct / total

def train_and_evaluate(model, train_loader, test_loader, device, input_flat_dim, epochs, max_lr):
    optimizer = optim.Adam(model.parameters(), lr=max_lr / 10)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr, 
        steps_per_epoch=len(train_loader), 
        epochs=epochs
    )
    criterion = nn.CrossEntropyLoss()
    
    history = {"acc": [], "pr_layers": []}
    
    # Fast feedback counter
    printed_fast_feedback = False
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, input_flat_dim).to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Fast Feedback Rule: print training progress in first 5 batches of Epoch 1
            if epoch == 0 and batch_idx < 5:
                print(f"    [Fast Feedback] Epoch 1, Batch {batch_idx+1}/5 - Loss: {loss.item():.4f}")
                printed_fast_feedback = True
                
        # Calculate evaluation stats
        acc = evaluate(model, test_loader, device, input_flat_dim)
        
        # Calculate PR for all gated layers (except output layer)
        epoch_prs = []
        for i in range(len(model.layers) - 1):
            pr = calculate_pr(model.layers[i].gate)
            epoch_prs.append(pr)
            
        history["acc"].append(acc)
        history["pr_layers"].append(epoch_prs)
        
        # Format PR printout
        pr_str = ", ".join([f"L{i+1}: {pr:.1f}" for i, pr in enumerate(epoch_prs)])
        print(f"  Epoch {epoch+1}/{epochs} | Acc: {acc:.2f}% | Effective N [{pr_str}]")
        
    return history

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNERS
# ══════════════════════════════════════════════════════════════════════

def run_experiment_1_fashion_mnist(device, epochs, batch_size):
    print("\n" + "="*70)
    print("EXPERIMENT 1: Fashion-MNIST Gating (SiLU, init=0.0 vs init=1.0)")
    print("="*70)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train_loader = DataLoader(datasets.FashionMNIST('data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.FashionMNIST('data', train=False, transform=transform), batch_size=batch_size, shuffle=False)
    
    # 1. Run CANDIDATE first (init=0.0) as per Golden Rule (Primero el Candidato)
    set_seed(42)
    print("\n--- Running Candidate: Gate Init = 0.0 (Discovery Mode) ---")
    model_cand = GatedMLP(784, [4096], 10, init_val=0.0, activation='silu').to(device)
    hist_cand = train_and_evaluate(model_cand, train_loader, test_loader, device, 784, epochs, max_lr=0.05)
    
    # 2. Run BASELINE second
    set_seed(42)
    print("\n--- Running Baseline: Gate Init = 1.0 (Baseline Mode) ---")
    model_base = GatedMLP(784, [4096], 10, init_val=1.0, activation='silu').to(device)
    hist_base = train_and_evaluate(model_base, train_loader, test_loader, device, 784, epochs, max_lr=0.05)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), hist_cand["acc"], label="Discovery (Init=0.0)", marker='o', color='teal')
    plt.plot(range(1, epochs+1), hist_base["acc"], label="Baseline (Init=1.0)", marker='s', color='orange')
    plt.title("Fashion-MNIST Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), [pr[0] for pr in hist_cand["pr_layers"]], label="Discovery (Init=0.0)", marker='o', color='teal')
    plt.plot(range(1, epochs+1), [pr[0] for pr in hist_base["pr_layers"]], label="Baseline (Init=1.0)", marker='s', color='orange')
    plt.title("Fashion-MNIST Effective Gates (PR)")
    plt.xlabel("Epoch")
    plt.ylabel("Effective N (max 4096)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/exp1_fashion_mnist_gating.png")
    print(f"\nSaved figure to: results/figures/exp1_fashion_mnist_gating.png")
    
    return hist_cand["acc"][-1], hist_base["acc"][-1]

def run_experiment_2_three_layers(device, epochs, batch_size):
    print("\n" + "="*70)
    print("EXPERIMENT 2: 3-Layer GatedMLP on MNIST (784 -> 4096 -> 4096 -> 10)")
    print("="*70)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=batch_size, shuffle=False)
    
    # 1. Run CANDIDATE (init=0.0) first
    set_seed(42)
    print("\n--- Running Candidate (3 layers): Gate Init = 0.0 ---")
    model_cand = GatedMLP(784, [4096, 4096], 10, init_val=0.0, activation='silu').to(device)
    hist_cand = train_and_evaluate(model_cand, train_loader, test_loader, device, 784, epochs, max_lr=0.05)
    
    # 2. Run BASELINE (init=1.0) second
    set_seed(42)
    print("\n--- Running Baseline (3 layers): Gate Init = 1.0 ---")
    model_base = GatedMLP(784, [4096, 4096], 10, init_val=1.0, activation='silu').to(device)
    hist_base = train_and_evaluate(model_base, train_loader, test_loader, device, 784, epochs, max_lr=0.05)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), hist_cand["acc"], label="Discovery (Init=0.0)", marker='o', color='teal')
    plt.plot(range(1, epochs+1), hist_base["acc"], label="Baseline (Init=1.0)", marker='s', color='orange')
    plt.title("3-Layer MNIST Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), [pr[0] for pr in hist_cand["pr_layers"]], label="L1 (Init=0.0)", linestyle='-', color='teal', marker='o')
    plt.plot(range(1, epochs+1), [pr[1] for pr in hist_cand["pr_layers"]], label="L2 (Init=0.0)", linestyle='--', color='teal', marker='^')
    plt.plot(range(1, epochs+1), [pr[0] for pr in hist_base["pr_layers"]], label="L1 (Init=1.0)", linestyle='-', color='orange', marker='s')
    plt.plot(range(1, epochs+1), [pr[1] for pr in hist_base["pr_layers"]], label="L2 (Init=1.0)", linestyle='--', color='orange', marker='d')
    plt.title("3-Layer Effective Gates (PR) by Layer")
    plt.xlabel("Epoch")
    plt.ylabel("Effective N")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/figures/exp2_3_layer_mnist.png")
    print(f"\nSaved figure to: results/figures/exp2_3_layer_mnist.png")
    
    return hist_cand["acc"][-1], hist_base["acc"][-1]

def run_experiment_3_scaling_sweep(device, epochs, batch_size):
    print("\n" + "="*70)
    print("EXPERIMENT 3: Effective N vs D Scaling Sweep on MNIST")
    print("="*70)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=batch_size, shuffle=False)
    
    D_list = [512, 1024, 2048, 4096, 8192]
    results = {}
    
    # Run from smallest to largest to trace behavior
    for D in D_list:
        print(f"\n--- Training GatedMLP with D = {D} (SiLU, init=0.0) ---")
        set_seed(42)
        model = GatedMLP(784, [D], 10, init_val=0.0, activation='silu').to(device)
        hist = train_and_evaluate(model, train_loader, test_loader, device, 784, epochs, max_lr=0.05)
        
        final_acc = hist["acc"][-1]
        final_pr = hist["pr_layers"][-1][0]
        ratio = final_pr / D
        
        results[D] = {
            "acc": final_acc,
            "N_eff": final_pr,
            "ratio": ratio
        }
        print(f"-> D = {D} Finished | Acc: {final_acc:.2f}% | Final N_eff: {final_pr:.1f} | Ratio (N_eff/D): {ratio:.4f}")
        
    # Plotting scaling curves
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Hidden Dimension D')
    ax1.set_ylabel('Effective N (PR)', color=color)
    ax1.plot(D_list, [results[d]["N_eff"] for d in D_list], marker='o', color=color, label="Effective N")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Ratio (N_eff / D)', color=color)
    ax2.plot(D_list, [results[d]["ratio"] for d in D_list], marker='s', color=color, linestyle='--', label="Ratio")
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Scaling Behavior: Effective N and Activation Ratio vs Backbone Dimension D")
    fig.tight_layout()
    plt.savefig("results/figures/exp3_scaling_sweep.png")
    print(f"\nSaved figure to: results/figures/exp3_scaling_sweep.png")
    
    # Print tabular summary
    print("\nSummary Table for Scaling Sweep:")
    print("D\t\tAcc\t\tN_eff\t\tRatio")
    print("-" * 50)
    for d in D_list:
        print(f"{d}\t\t{results[d]['acc']:.2f}%\t\t{results[d]['N_eff']:.1f}\t\t{results[d]['ratio']:.4f}")
        
    return results

def run_experiment_4_cifar10(device, epochs, batch_size):
    print("\n" + "="*70)
    print("EXPERIMENT 4: CIFAR-10 Gating (SiLU, init=0.0 vs init=1.0)")
    print("="*70)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_loader = DataLoader(datasets.CIFAR10('data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.CIFAR10('data', train=False, transform=transform), batch_size=batch_size, shuffle=False)
    
    # Input dim for flattened CIFAR-10 is 3 * 32 * 32 = 3072
    input_dim = 3072
    
    # 1. Run CANDIDATE (init=0.0) first
    set_seed(42)
    print("\n--- Running Candidate (CIFAR-10): Gate Init = 0.0 ---")
    model_cand = GatedMLP(input_dim, [4096], 10, init_val=0.0, activation='silu').to(device)
    hist_cand = train_and_evaluate(model_cand, train_loader, test_loader, device, input_dim, epochs, max_lr=0.05)
    
    # 2. Run BASELINE (init=1.0) second
    set_seed(42)
    print("\n--- Running Baseline (CIFAR-10): Gate Init = 1.0 ---")
    model_base = GatedMLP(input_dim, [4096], 10, init_val=1.0, activation='silu').to(device)
    hist_base = train_and_evaluate(model_base, train_loader, test_loader, device, input_dim, epochs, max_lr=0.05)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), hist_cand["acc"], label="Discovery (Init=0.0)", marker='o', color='teal')
    plt.plot(range(1, epochs+1), hist_base["acc"], label="Baseline (Init=1.0)", marker='s', color='orange')
    plt.title("CIFAR-10 Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), [pr[0] for pr in hist_cand["pr_layers"]], label="Discovery (Init=0.0)", marker='o', color='teal')
    plt.plot(range(1, epochs+1), [pr[0] for pr in hist_base["pr_layers"]], label="Baseline (Init=1.0)", marker='s', color='orange')
    plt.title("CIFAR-10 Effective Gates (PR)")
    plt.xlabel("Epoch")
    plt.ylabel("Effective N (max 4096)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/figures/exp4_cifar10_gating.png")
    print(f"\nSaved figure to: results/figures/exp4_cifar10_gating.png")
    
    return hist_cand["acc"][-1], hist_base["acc"][-1]

# ══════════════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Oligarchy Hypothesis Validation Experiments")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4], default=None, 
                        help="Run a specific experiment (1-4). If not specified, runs all experiments.")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of epochs for training (default: 10).")
    parser.add_argument("--batch-size", type=int, default=128, 
                        help="Batch size (default: 128).")
    parser.add_argument("--cpu", action="store_true", 
                        help="Force execution on CPU.")
    parser.add_argument("--quick", action="store_true", 
                        help="Run in super-fast debug mode (1 epoch, restricted data) to verify script sanity.")
    args = parser.parse_args()
    
    device = get_device(force_cpu=args.cpu)
    print(f"Using Device: {device}")
    
    epochs = args.epochs
    batch_size = args.batch_size
    
    if args.quick:
        print("\n[Quick Mode] Running in QUICK DEBUG mode (1 epoch, batch_size=128)")
        epochs = 1
        
    start_time = time.time()
    
    if args.exp is None:
        print("\n[Run] Running ALL validation experiments...")
        run_experiment_1_fashion_mnist(device, epochs, batch_size)
        run_experiment_2_three_layers(device, epochs, batch_size)
        run_experiment_3_scaling_sweep(device, epochs, batch_size)
        run_experiment_4_cifar10(device, epochs, batch_size)
    else:
        print(f"\n[Run] Running Selected Experiment {args.exp}...")
        if args.exp == 1:
            run_experiment_1_fashion_mnist(device, epochs, batch_size)
        elif args.exp == 2:
            run_experiment_2_three_layers(device, epochs, batch_size)
        elif args.exp == 3:
            run_experiment_3_scaling_sweep(device, epochs, batch_size)
        elif args.exp == 4:
            run_experiment_4_cifar10(device, epochs, batch_size)
            
    total_time = time.time() - start_time
    print(f"\nDone! All selected experiments finished in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
