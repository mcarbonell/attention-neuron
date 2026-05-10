import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os
import math

# --- Configuration ---
CONFIG = {
    "batch_size": 128,
    "epochs": 10,
    "gate_lr": 1e-3,
    "hidden_dim": 2048, # Increased as requested
    "device": "cpu",
    "seed": 42
}

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# --- Architecture Components ---
class TernaryLinearGated(nn.Module):
    def __init__(self, in_features, out_features, init_val=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. Initialize Weights as Random Ternary {-1, 0, 1}
        weights = torch.randint(-1, 2, (out_features, in_features)).float()
        
        # 2. Register as Buffer (Frozen)
        self.register_buffer("weight", weights)
        
        # 3. Float Gating (Learnable) - Initialized to init_val
        self.gate = nn.Parameter(torch.full((out_features,), float(init_val)))
        
        # No bias for extreme efficiency
        
    def forward(self, x):
        # x: [Batch, In], W: [Out, In]
        # Standard matmul with frozen ternary weights
        x = torch.matmul(x, self.weight.t())
        
        # Apply Float Gating
        return x * self.gate

    def __repr__(self):
        return f"TernaryLinearGated(in={self.in_features}, out={self.out_features}, weights=TERNARY_FROZEN, gate=FLOAT_LEARNABLE)"

class GatedTernaryMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = TernaryLinearGated(input_dim, hidden_dim, init_val=0.0)
        self.silu1 = nn.SiLU()
        self.layer2 = TernaryLinearGated(hidden_dim, hidden_dim, init_val=1.0)
        self.silu2 = nn.SiLU()
        self.layer3 = TernaryLinearGated(hidden_dim, output_dim, init_val=1.0)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.silu1(x)
        x = self.layer2(x)
        x = self.silu2(x)
        x = self.layer3(x)
        return x

    def print_architecture(self):
        print("\n--- Network Architecture ---")
        print(f"Input: 784")
        print(f"Layer 1: {self.layer1}")
        print(f"Layer 2: {self.layer2}")
        print(f"Layer 3: {self.layer3}")
        
        total_params = sum(p.numel() for p in self.parameters())
        learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.buffers())
        
        print(f"Total Parameters:     {total_params:,}")
        print(f"Learnable (Gates):    {learnable_params:,}")
        print(f"Frozen (Ternary W):   {frozen_params:,}")
        print("---------------------------\n")

    def print_gate_stats(self):
        print("\n--- Gate Statistics ---")
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3]):
            g = layer.gate.data
            print(f"Layer {i+1}: Mean={g.mean():.6f}, Std={g.std():.6f}, Max={g.max():.6f}, Min={g.min():.6f}, Norm={g.norm():.4f}")
        print("-----------------------\n")

# --- Training / Eval ---
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

def train_model(model, train_loader, test_loader, config):
    model.print_architecture()
    optimizer = optim.Adam(model.parameters(), lr=config["gate_lr"])
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    results = []
    total_eval_time = 0
    
    for epoch in range(config["epochs"]):
        model.train()
        epoch_start = time.time()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.view(-1, 28*28).to(config["device"]), target.to(config["device"])
            
            f_start = time.time()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_eval_time += (time.time() - f_start)
            
            if i < 5 and epoch == 0:
                print(f"  Batch {i}: Loss {loss.item():.4f}")
        
        acc = evaluate(model, test_loader, config["device"])
        results.append(acc)
        print(f"Epoch {epoch+1}/{config['epochs']} | Acc: {acc:.2f}% | Time: {time.time()-epoch_start:.2f}s")
        model.print_gate_stats()
        
    wall_clock = time.time() - start_time
    
    metrics = {
        "final_objective": acc,
        "wall_clock_time": wall_clock,
        "function_evaluation_time": total_eval_time,
        "internal_overhead_time": wall_clock - total_eval_time,
        "PEI": acc / math.log10(sum(p.numel() for p in model.parameters()) + 1),
        "total_params": sum(p.numel() for p in model.parameters())
    }
    
    return results, metrics

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
    
    model = GatedTernaryMLP(784, CONFIG["hidden_dim"], 10).to(CONFIG["device"])
    res, metrics = train_model(model, train_loader, test_loader, CONFIG)
    
    # Save results
    os.makedirs("results/raw", exist_ok=True)
    report = {
        "config": CONFIG,
        "metrics": metrics,
        "history": res
    }
    with open("results/raw/v253_ternary_weights.json", "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nResults saved to results/raw/v253_ternary_weights.json")

if __name__ == "__main__":
    main()
