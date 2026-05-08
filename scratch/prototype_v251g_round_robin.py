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
    "batch_size": 128,
    "epochs": 10,
    "lr": 1e-3,
    "device": "cpu",
    "seed": 42,
    "hidden_dim": 512
}

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gate = nn.Parameter(torch.ones(out_features))
                
    def forward(self, x):
        x = self.linear(x)
        return x * self.gate

class GatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = GatedLinear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.layer2 = GatedLinear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.layer3 = GatedLinear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
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

def train_baseline(train_loader, test_loader, config):
    print("\n--- Training Baseline (Full Network Every Batch) ---")
    model = GatedMLP(784, config["hidden_dim"], 10).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    
    results = []
    for epoch in range(config["epochs"]):
        model.train()
        for data, target in train_loader:
            data, target = data.view(-1, 28*28).to(config["device"]), target.to(config["device"])
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        acc = evaluate(model, test_loader, config["device"])
        results.append(acc)
        print(f"Epoch {epoch+1}/{config['epochs']} - Test Acc: {acc:.2f}%")
    return results

def train_round_robin(train_loader, test_loader, config):
    print("\n--- Training Round-Robin (Global Gating + Rotating Weights) ---")
    model = GatedMLP(784, config["hidden_dim"], 10).to(config["device"])
    
    # Define parameter groups
    weight_layers = [
        list(model.layer1.linear.parameters()),
        list(model.layer2.linear.parameters()),
        list(model.layer3.linear.parameters())
    ]
    gate_params = [model.layer1.gate, model.layer2.gate, model.layer3.gate]
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    
    results = []
    batch_count = 0
    num_layers = len(weight_layers)
    
    for epoch in range(config["epochs"]):
        model.train()
        for data, target in train_loader:
            data, target = data.view(-1, 28*28).to(config["device"]), target.to(config["device"])
            
            # 1. Choose which weights to train this batch
            current_layer_idx = batch_count % num_layers
            
            # 2. Set requires_grad
            # Gating is ALWAYS trainable
            for g in gate_params:
                g.requires_grad = True
            
            # Weights only for the current layer
            for i, layer_params in enumerate(weight_layers):
                is_active = (i == current_layer_idx)
                for p in layer_params:
                    p.requires_grad = is_active
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 3. Step
            # Optimization: PyTorch's optimizer.step() will only update params with gradients
            optimizer.step()
            
            batch_count += 1
        
        acc = evaluate(model, test_loader, config["device"])
        results.append(acc)
        print(f"Epoch {epoch+1}/{config['epochs']} - Layer {current_layer_idx+1} active - Test Acc: {acc:.2f}%")
        
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
    
    baseline_res = train_baseline(train_loader, test_loader, CONFIG)
    rr_res = train_round_robin(train_loader, test_loader, CONFIG)
    
    print("\n--- Final Comparison ---")
    print(f"{'Epoch':<10} | {'Baseline Acc (%)':<20} | {'Round-Robin Acc (%)':<20}")
    print("-" * 60)
    for i in range(CONFIG["epochs"]):
        print(f"{i+1:<10} | {baseline_res[i]:<20.2f} | {rr_res[i]:<20.2f}")
    
    # Save results
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v251g_round_robin.json", "w") as f:
        json.dump({"baseline": baseline_res, "round_robin": rr_res}, f, indent=4)

if __name__ == "__main__":
    main()
