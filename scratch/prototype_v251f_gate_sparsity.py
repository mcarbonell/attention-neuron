import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os
import math
import numpy as np

# Configuration
CONFIG = {
    "batch_size": 128,
    "epochs": 10,
    "max_lr": 0.05,
    "weight_decay": 1e-3, # Regularization on gates
    "device": "cpu",
    "seed": 42,
    "hidden_dim": 4096
}

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features, frozen=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gate = nn.Parameter(torch.ones(out_features))
        
        if frozen:
            for param in self.linear.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        x = self.linear(x)
        return x * self.gate

class GatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = GatedLinear(input_dim, hidden_dim, frozen=True)
        self.relu = nn.ReLU()
        self.layer2 = GatedLinear(hidden_dim, output_dim, frozen=True)
        
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

def analyze_gates(model):
    # Analyze hidden layer gates
    gates = model.layer1.gate.detach().cpu().numpy()
    abs_gates = np.abs(gates)
    
    thresholds = [1e-1, 1e-2, 1e-3, 1e-4]
    stats = {
        "mean": float(np.mean(gates)),
        "std": float(np.std(gates)),
        "max": float(np.max(gates)),
        "min": float(np.min(gates)),
        "sparsity": {}
    }
    
    for t in thresholds:
        count = np.sum(abs_gates < t)
        stats["sparsity"][f"below_{t}"] = float(count / len(gates) * 100)
        
    return stats, gates

def main():
    set_seed(CONFIG["seed"])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    model = GatedMLP(784, CONFIG["hidden_dim"], 10).to(CONFIG["device"])
    
    # Weight decay only on gates
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["max_lr"]/10, weight_decay=CONFIG["weight_decay"])
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=CONFIG["max_lr"], 
        steps_per_epoch=len(train_loader), 
        epochs=CONFIG["epochs"]
    )
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n--- Training Sparsity Analysis | Dim: {CONFIG['hidden_dim']} | WD: {CONFIG['weight_decay']} ---")
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        for data, target in train_loader:
            data, target = data.view(-1, 28*28).to(CONFIG["device"]), target.to(CONFIG["device"])
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        test_acc = evaluate(model, test_loader, CONFIG["device"])
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Test Acc: {test_acc:.2f}%")
        
    # Final Analysis
    stats, gates = analyze_gates(model)
    
    print("\n--- Gate Statistics (Hidden Layer) ---")
    print(f"Mean: {stats['mean']:.6f} | Std: {stats['std']:.6f}")
    print(f"Max:  {stats['max']:.6f} | Min: {stats['min']:.6f}")
    print("\n--- Sparsity (%) ---")
    for k, v in stats["sparsity"].items():
        print(f"{k:<15}: {v:>6.2f}%")
        
    # Save results
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v251f_sparsity_analysis.json", "w") as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()
