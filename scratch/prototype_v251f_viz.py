import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration (same as v251f)
CONFIG = {
    "batch_size": 128,
    "epochs": 10,
    "max_lr": 0.05,
    "weight_decay": 1e-3,
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

def main():
    set_seed(CONFIG["seed"])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=CONFIG["batch_size"], shuffle=True)
    
    model = GatedMLP(784, CONFIG["hidden_dim"], 10).to(CONFIG["device"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["max_lr"]/10, weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG["max_lr"], steps_per_epoch=len(train_loader), epochs=CONFIG["epochs"])
    criterion = nn.CrossEntropyLoss()
    
    print("Training for distribution analysis...")
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
        print(f"Epoch {epoch+1} complete.")

    # Extract gates
    gates = model.layer1.gate.detach().cpu().numpy()
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(gates, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"Gate Value Distribution (D={CONFIG['hidden_dim']})")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Absolute value distribution (log scale)
    plt.subplot(1, 2, 2)
    plt.hist(np.abs(gates), bins=100, color='salmon', edgecolor='black', alpha=0.7)
    plt.yscale('log')
    plt.title("Absolute Gate Values (Log Scale)")
    plt.xlabel("|Gate Value|")
    plt.ylabel("Frequency (Log)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    # Save the plot
    os.makedirs("results/figures", exist_ok=True)
    plot_path = "results/figures/v251f_gate_distribution.png"
    plt.savefig(plot_path)
    print(f"\nDistribution plot saved to: {plot_path}")
    
    # Basic analysis
    mean = np.mean(gates)
    std = np.std(gates)
    print(f"\n--- Statistical Summary ---")
    print(f"Mean: {mean:.6f}")
    print(f"Std Dev: {std:.6f}")
    print(f"Symmetry: {'Symmetric' if abs(mean) < 0.05 else 'Skewed'}")
    
if __name__ == "__main__":
    main()
