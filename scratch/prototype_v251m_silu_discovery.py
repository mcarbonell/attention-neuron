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
    def __init__(self, in_features, out_features, init_val=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gate = nn.Parameter(torch.full((out_features,), init_val))
        for param in self.linear.parameters():
            param.requires_grad = False
                
    def forward(self, x):
        return self.linear(x) * self.gate

class GatedMLP_SiLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, init_val=1.0):
        super().__init__()
        self.layer1 = GatedLinear(input_dim, hidden_dim, init_val)
        self.silu = nn.SiLU() # Smooth activation with non-zero gradient at 0
        self.layer2 = GatedLinear(hidden_dim, output_dim, 1.0)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.silu(x)
        x = self.layer2(x)
        return x

def calculate_pr(gate_tensor):
    abs_g = torch.abs(gate_tensor)
    sum_abs = torch.sum(abs_g).item()
    sum_sq = torch.sum(gate_tensor**2).item()
    if sum_sq == 0: return 0
    return (sum_abs ** 2) / sum_sq

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

def run_trial(init_val, train_loader, test_loader, config):
    print(f"\n--- Running Trial (SiLU) with Gate Init = {init_val} ---")
    set_seed(config["seed"])
    model = GatedMLP_SiLU(784, config["hidden_dim"], 10, init_val).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["max_lr"]/10)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["max_lr"], steps_per_epoch=len(train_loader), epochs=config["epochs"])
    criterion = nn.CrossEntropyLoss()
    
    history = {"acc": [], "pr": []}
    
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
        pr = calculate_pr(model.layer1.gate)
        history["acc"].append(acc)
        history["pr"].append(pr)
        print(f"Epoch {epoch+1}/{config['epochs']} - Acc: {acc:.2f}% | Effective N: {pr:.2f}/{config['hidden_dim']}")
        
    return history

def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=CONFIG["batch_size"], shuffle=False)
    
    # Run trials
    h_one = run_trial(1.0, train_loader, test_loader, CONFIG)
    h_zero = run_trial(0.0, train_loader, test_loader, CONFIG)
    
    # Visualization
    plt.figure(figsize=(15, 6))
    
    # 1. Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 11), h_one["acc"], label="SiLU Init=1.0", marker='o')
    plt.plot(range(1, 11), h_zero["acc"], label="SiLU Init=0.0", marker='s')
    plt.title("Convergence Comparison (SiLU)")
    plt.xlabel("Epoch")
    plt.ylabel("Test Acc (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. PR
    plt.subplot(1, 2, 2)
    plt.plot(range(1, 11), h_one["pr"], label="SiLU Init=1.0", marker='o')
    plt.plot(range(1, 11), h_zero["pr"], label="SiLU Init=0.0", marker='s')
    plt.title("Effective N (Participation Ratio)")
    plt.xlabel("Epoch")
    plt.ylabel("Effective N")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/v251m_silu_discovery.png")
    print("\nSiLU discovery plot saved to: results/figures/v251m_silu_discovery.png")

if __name__ == "__main__":
    main()
