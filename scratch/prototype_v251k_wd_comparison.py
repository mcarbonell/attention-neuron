import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# Configuration
CONFIG = {
    "batch_size": 128,
    "epochs": 10,
    "max_lr": 0.05,
    "device": "cpu",
    "seed": 42,
    "hidden_dim": 4096
}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gate = nn.Parameter(torch.ones(out_features))
        for param in self.linear.parameters():
            param.requires_grad = False
                
    def forward(self, x):
        return self.linear(x) * self.gate

class GatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = GatedLinear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = GatedLinear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.view(-1, 28*28).to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(loader.dataset)

def run_trial(wd, train_loader, test_loader, config):
    set_seed(config["seed"])
    model = GatedMLP(784, config["hidden_dim"], 10).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["max_lr"]/10, weight_decay=wd)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["max_lr"], steps_per_epoch=len(train_loader), epochs=config["epochs"])
    criterion = nn.CrossEntropyLoss()
    
    epoch_accs = []
    print(f"\n--- Running Trial with WD={wd} ---")
    for epoch in range(config["epochs"]):
        model.train()
        for data, target in train_loader:
            data, target = data.view(-1, 28*28).to(config["device"]), target.to(config["device"])
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        acc = evaluate(model, test_loader, config["device"])
        epoch_accs.append(acc)
        print(f"Epoch {epoch+1}/{config['epochs']} - Test Acc: {acc:.2f}%")
        
    gates = model.layer1.gate.detach().cpu().numpy()
    return epoch_accs, gates

def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=CONFIG["batch_size"], shuffle=False)
    
    # Run trials
    accs_no_wd, gates_no_wd = run_trial(0.0, train_loader, test_loader, CONFIG)
    accs_wd, gates_wd = run_trial(1e-3, train_loader, test_loader, CONFIG)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # 1. Accuracy Curve
    plt.subplot(1, 3, 1)
    plt.plot(range(1, 11), accs_no_wd, label="WD=0", marker='o')
    plt.plot(range(1, 11), accs_wd, label="WD=1e-3", marker='s')
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Test Acc (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Gate Distribution (No WD)
    plt.subplot(1, 3, 2)
    plt.hist(gates_no_wd, bins=50, color='skyblue', alpha=0.7, label="WD=0")
    plt.title("Distribution (WD=0)")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")

    # 3. Gate Distribution (With WD)
    plt.subplot(1, 3, 3)
    plt.hist(gates_wd, bins=50, color='salmon', alpha=0.7, label="WD=1e-3")
    plt.title("Distribution (WD=1e-3)")
    plt.xlabel("Gate Value")
    
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/v251k_wd_comparison.png")
    print("\nComparison plot saved to: results/figures/v251k_wd_comparison.png")
    
    # Save raw data
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v251k_wd_comparison.json", "w") as f:
        json.dump({
            "no_wd": {"acc": accs_no_wd, "mean_gate": float(np.mean(gates_no_wd))},
            "with_wd": {"acc": accs_wd, "mean_gate": float(np.mean(gates_wd))}
        }, f, indent=4)

    # Detailed Statistics Output
    def print_stats(name, data):
        print(f"\n--- Statistics for {name} ---")
        print(f"Mean:     {np.mean(data):.6f}")
        print(f"Std Dev:  {np.std(data):.6f}")
        print(f"Min:      {np.min(data):.6f}")
        print(f"Max:      {np.max(data):.6f}")
        print(f"Range:    {np.max(data) - np.min(data):.6f}")
        print(f"Percentiles:")
        for p in [1, 5, 25, 50, 75, 95, 99]:
            print(f"  {p}%: {np.percentile(data, p):.6f}")
        
        sparsity = np.mean(np.abs(data) < 1e-3) * 100
        print(f"Sparsity (|x| < 1e-3): {sparsity:.2f}%")

    print_stats("WD = 0.0", gates_no_wd)
    print_stats("WD = 1e-3", gates_wd)

if __name__ == "__main__":
    main()
