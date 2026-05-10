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
    "gate_lr": 1e-2,
    "hidden_dim": 512,
    "ternary_threshold": 0.05,
    "device": "cpu", # DirectML usage: "private_use_c:0" or "cpu"
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
        # Identity gradient (Straight-Through Estimator)
        return grad_output, None

def ternary_gate_op(x, threshold):
    return TernarySTE.apply(x, threshold)

# --- Architecture ---
class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features, ternary=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Frozen weights from the start
        for p in self.linear.parameters():
            p.requires_grad = False
            
        self.gate = nn.Parameter(torch.randn(out_features) * 0.1)
        self.ternary = ternary
        self.threshold = CONFIG["ternary_threshold"]
                
    def forward(self, x):
        x = self.linear(x)
        if self.ternary:
            g_q = ternary_gate_op(self.gate, self.threshold)
            return x * g_q
        else:
            return x * torch.sigmoid(self.gate) # Baseline uses sigmoid for stability

class TernaryGatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, ternary=False):
        super().__init__()
        self.layer1 = GatedLinear(input_dim, hidden_dim, ternary=ternary)
        self.relu1 = nn.ReLU()
        self.layer2 = GatedLinear(hidden_dim, hidden_dim, ternary=ternary)
        self.relu2 = nn.ReLU()
        self.layer3 = GatedLinear(hidden_dim, output_dim, ternary=ternary)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x

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

def train_model(name, model, train_loader, test_loader, config):
    print(f"\n--- Training {name} ---")
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
            
            # Regla de Oro: Fast Feedback
            if i < 5 and epoch == 0:
                print(f"  Batch {i}: Loss {loss.item():.4f}")
        
        acc = evaluate(model, test_loader, config["device"])
        results.append(acc)
        print(f"Epoch {epoch+1}/{config['epochs']} | Acc: {acc:.2f}% | Time: {time.time()-epoch_start:.2f}s")
        
    wall_clock = time.time() - start_time
    
    # Calculate Metrics as per GEMINI.md
    total_params = sum(p.numel() for p in model.parameters())
    pei = acc / math.log10(total_params + 1)
    
    # Calculate gate sparsity (only for ternary)
    sparsity = 0
    if hasattr(model.layer1, "ternary") and model.layer1.ternary:
        with torch.no_grad():
            all_gates = torch.cat([model.layer1.gate, model.layer2.gate, model.layer3.gate])
            q_gates = torch.where(all_gates > config["ternary_threshold"], 1.0, 
                      torch.where(all_gates < -config["ternary_threshold"], -1.0, 0.0))
            sparsity = (q_gates == 0).float().mean().item() * 100

    metrics = {
        "final_objective": acc,
        "wall_clock_time": wall_clock,
        "function_evaluation_time": total_eval_time,
        "internal_overhead_time": wall_clock - total_eval_time,
        "PEI": pei,
        "gate_sparsity": sparsity,
        "total_params": total_params
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
    
    # 1. Continuous Gating (Baseline)
    baseline_model = TernaryGatedMLP(784, CONFIG["hidden_dim"], 10, ternary=False).to(CONFIG["device"])
    b_res, b_metrics = train_model("Continuous Baseline", baseline_model, train_loader, test_loader, CONFIG)
    
    # 2. Ternary Gating
    set_seed(CONFIG["seed"]) # Reset seed for fair weight initialization
    ternary_model = TernaryGatedMLP(784, CONFIG["hidden_dim"], 10, ternary=True).to(CONFIG["device"])
    t_res, t_metrics = train_model("Ternary Gating (STE)", ternary_model, train_loader, test_loader, CONFIG)
    
    print("\n--- Final Results ---")
    print(f"Baseline Final Acc: {b_metrics['final_objective']:.2f}%")
    print(f"Ternary Final Acc:  {t_metrics['final_objective']:.2f}% (Sparsity: {t_metrics['gate_sparsity']:.1f}%)")
    print(f"PEI (Ternary):      {t_metrics['PEI']:.4f}")
    print(f"Overhead (Ternary): {t_metrics['internal_overhead_time']:.2f}s")

    # Save
    os.makedirs("results/raw", exist_ok=True)
    report = {
        "config": CONFIG,
        "baseline": b_metrics,
        "ternary": t_metrics,
        "history_baseline": b_res,
        "history_ternary": t_res
    }
    with open("results/raw/v252_ternary_gating.json", "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nResults saved to results/raw/v252_ternary_gating.json")

if __name__ == "__main__":
    main()
