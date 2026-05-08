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
    "batch_size": 128, # Larger batch for speed in sweep
    "epochs": 10,
    "lr": 1e-3,
    "device": "cpu",
    "seed": 42,
    "hidden_dims": [32, 64, 128, 256, 512, 1024, 2048, 4096]
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

def train_and_eval(hidden_dim, train_loader, test_loader, config):
    name = f"FrozenGated_H{hidden_dim}"
    print(f"\n--- Testing Dimension: {hidden_dim} ---")
    
    model = GatedMLP(784, hidden_dim, 10).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    
    start_time = time.time()
    epoch_results = []
    
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.view(-1, 28*28).to(config["device"]), target.to(config["device"])
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        test_acc = evaluate(model, test_loader, config["device"])
        epoch_results.append(test_acc)
        print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {total_loss/len(train_loader):.4f} - Test Acc: {test_acc:.2f}%")
        
    end_time = time.time()
    wall_clock = end_time - start_time
    
    return {
        "hidden_dim": hidden_dim,
        "trainable_params": trainable_params,
        "final_accuracy": epoch_results[-1],
        "epoch_accuracies": epoch_results,
        "wall_clock_time": wall_clock,
        "pei": epoch_results[-1] / math.log10(trainable_params + 1)
    }

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
    
    results = []
    for h_dim in CONFIG["hidden_dims"]:
        res = train_and_eval(h_dim, train_loader, test_loader, CONFIG)
        results.append(res)
        
    # Save results
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v251b_gating_sweep.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n--- Sweep Summary ---")
    print(f"{'Hidden Dim':<10} | {'Params':<10} | {'Final Acc (%)':<15} | {'PEI':<10}")
    print("-" * 55)
    for r in results:
        print(f"{r['hidden_dim']:<10} | {r['trainable_params']:<10} | {r['final_accuracy']:<15.2f} | {r['pei']:<10.2f}")

if __name__ == "__main__":
    main()
