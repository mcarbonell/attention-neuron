import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os
import math

# Configuration
CONFIG = {
    "batch_size": 64,
    "epochs": 10,
    "lr": 1e-3,
    "hidden_dim": 512,
    "device": "cpu", # Faster for small networks as per GEMINI.md
    "seed": 42
}

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features, frozen=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gate = nn.Parameter(torch.ones(out_features))
        
        if frozen:
            for param in self.linear.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        x = self.linear(x)
        return x * self.gate

class StandardMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)

class GatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, frozen=False):
        super().__init__()
        self.layer1 = GatedLinear(input_dim, hidden_dim, frozen=frozen)
        self.relu = nn.ReLU()
        self.layer2 = GatedLinear(hidden_dim, output_dim, frozen=frozen)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def train_and_eval(name, model, train_loader, test_loader, config):
    print(f"\n--- Training {name} ---")
    model.to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    start_wall = time.time()
    total_eval_time = 0
    
    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.view(-1, 28*28).to(config["device"]), target.to(config["device"])
            
            # Forward + measure time
            fwd_start = time.time()
            output = model(data)
            loss = criterion(output, target)
            total_eval_time += (time.time() - fwd_start)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if epoch == 0 and batch_idx < 5:
                print(f"Epoch 1, Batch {batch_idx+1}: Loss = {loss.item():.4f}")
                
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {epoch_loss/len(train_loader):.4f}")
        
    end_wall = time.time()
    
    # Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.view(-1, 28*28).to(config["device"]), target.to(config["device"])
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    accuracy = 100. * correct / len(test_loader.dataset)
    wall_clock = end_wall - start_wall
    overhead = wall_clock - total_eval_time
    pei = accuracy / math.log10(trainable_params + 1)
    
    results = {
        "name": name,
        "final_objective": accuracy,
        "total_evaluations": config["epochs"] * len(train_loader.dataset),
        "wall_clock_time": wall_clock,
        "function_evaluation_time": total_eval_time,
        "internal_overhead_time": overhead,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "PEI": pei
    }
    
    print(f"Final Accuracy: {accuracy:.2f}%")
    return results

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
    
    variants = [
        ("Standard MLP", StandardMLP(784, CONFIG["hidden_dim"], 10)),
        ("Gated MLP (Full)", GatedMLP(784, CONFIG["hidden_dim"], 10, frozen=False)),
        ("Gated MLP (Frozen Weights)", GatedMLP(784, CONFIG["hidden_dim"], 10, frozen=True))
    ]
    
    all_results = []
    for name, model in variants:
        res = train_and_eval(name, model, train_loader, test_loader, CONFIG)
        all_results.append(res)
        
    # Save results
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v251_gating_comparison.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    print("\n--- Summary ---")
    print(f"{'Model':<30} | {'Acc (%)':<10} | {'Trainable Params':<20} | {'PEI':<10}")
    print("-" * 80)
    for res in all_results:
        print(f"{res['name']:<30} | {res['final_objective']:<10.2f} | {res['trainable_params']:<20,} | {res['PEI']:<10.2f}")

if __name__ == "__main__":
    main()
