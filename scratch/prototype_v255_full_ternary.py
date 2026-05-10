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
    "epochs": 15, # Extra epochs for discrete convergence
    "gate_lr": 1e-3,
    "hidden_dim": 2048,
    "ternary_threshold": 0.05,
    "device": "cpu",
    "seed": 42
}

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# --- Ternary Logic (STE) ---
class TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        # Quantize to {-1, 0, 1}
        x_q = torch.where(x > threshold, 1.0, 
              torch.where(x < -threshold, -1.0, 0.0))
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def ternary_gate_op(x, threshold):
    return TernarySTE.apply(x, threshold)

# --- Architecture Components ---
class FullTernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, init_val=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. Weights: Frozen Random Ternary {-1, 0, 1}
        weights = torch.randint(-1, 2, (out_features, in_features)).float()
        self.register_buffer("weight", weights)
        
        # 2. Gates: Learnable (Continuous Latent, Discrete Forward)
        self.gate_latent = nn.Parameter(torch.full((out_features,), float(init_val)))
        self.threshold = CONFIG["ternary_threshold"]
        
    def forward(self, x):
        # Weights are ternary, Gates are ternary
        x = torch.matmul(x, self.weight.t())
        g_q = ternary_gate_op(self.gate_latent, self.threshold)
        return x * g_q

    def __repr__(self):
        return f"FullTernaryLinear(in={self.in_features}, out={self.out_features}, weights=TERNARY_FROZEN, gates=TERNARY_STE)"

class FullTernaryMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = FullTernaryLinear(input_dim, hidden_dim, init_val=0.0)
        self.silu1 = nn.SiLU()
        self.layer2 = FullTernaryLinear(hidden_dim, hidden_dim, init_val=1.0)
        self.silu2 = nn.SiLU()
        self.layer3 = FullTernaryLinear(hidden_dim, output_dim, init_val=1.0)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.silu1(x)
        x = self.layer2(x)
        x = self.silu2(x)
        x = self.layer3(x)
        return x

    def print_architecture(self):
        print("\n--- Network Architecture (FULL TERNARY {-1,0,1}) ---")
        print(f"Layer 1: {self.layer1}")
        print(f"Layer 2: {self.layer2}")
        print(f"Layer 3: {self.layer3}")
        
        total_params = sum(p.numel() for p in self.parameters()) + sum(p.numel() for p in self.buffers())
        print(f"Total Parameters: {total_params:,}")
        print("---------------------------\n")

    def print_gate_stats(self):
        print("\n--- Gate Statistics (Latent) ---")
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3]):
            g = layer.gate_latent.data
            g_q = torch.where(g > CONFIG["ternary_threshold"], 1.0, 
                  torch.where(g < -CONFIG["ternary_threshold"], -1.0, 0.0))
            sparsity = (g_q == 0).float().mean().item() * 100
            print(f"Layer {i+1}: LatentMean={g.mean():.4f}, LatentStd={g.std():.4f}, Sparsity={sparsity:.1f}%")
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
    
    for epoch in range(config["epochs"]):
        model.train()
        epoch_start = time.time()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.view(-1, 28*28).to(config["device"]), target.to(config["device"])
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if i < 5 and epoch == 0:
                print(f"  Batch {i}: Loss {loss.item():.4f}")
        
        acc = evaluate(model, test_loader, config["device"])
        results.append(acc)
        print(f"Epoch {epoch+1}/{config['epochs']} | Acc: {acc:.2f}% | Time: {time.time()-epoch_start:.2f}s")
        model.print_gate_stats()
        
    return results, time.time() - start_time

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
    
    model = FullTernaryMLP(784, CONFIG["hidden_dim"], 10).to(CONFIG["device"])
    res, duration = train_model(model, train_loader, test_loader, CONFIG)
    
    # Save results
    os.makedirs("results/raw", exist_ok=True)
    report = {
        "config": CONFIG,
        "history": res,
        "duration": duration
    }
    with open("results/raw/v255_full_ternary.json", "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nResults saved to results/raw/v255_full_ternary.json")

if __name__ == "__main__":
    main()
