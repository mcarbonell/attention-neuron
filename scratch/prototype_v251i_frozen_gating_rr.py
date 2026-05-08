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
    "warmup_epochs": 2,
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

def train_frozen_gating_rr(train_loader, test_loader, config):
    print(f"\n--- Phased Training: Warmup (Gates) -> Frozen Gates + Epoch-RR Weights ---")
    model = GatedMLP(784, config["hidden_dim"], 10).to(config["device"])
    
    # Groups
    weight_layers = [
        list(model.layer1.linear.parameters()),
        list(model.layer2.linear.parameters()),
        list(model.layer3.linear.parameters())
    ]
    gate_params = [model.layer1.gate, model.layer2.gate, model.layer3.gate]
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    
    results = []
    num_layers = len(weight_layers)
    
    for epoch in range(config["epochs"]):
        model.train()
        is_warmup = (epoch < config["warmup_epochs"])
        
        # Determine training mode for the epoch
        if is_warmup:
            mode = "WARMUP (Gates Only, Weights Frozen)"
            # Enable Gates, Disable Weights
            for g in gate_params: g.requires_grad = True
            for layer in weight_layers:
                for p in layer: p.requires_grad = False
        else:
            # Phase 2: Frozen Gates + Round-Robin Weights per Epoch
            current_layer_idx = (epoch - config["warmup_epochs"]) % num_layers
            mode = f"EPOCH-RR (Layer {current_layer_idx+1} Weights, Gates Frozen)"
            
            # Disable ALL Gates
            for g in gate_params: g.requires_grad = False
            
            # Enable only ONE Weight layer
            for i, layer_params in enumerate(weight_layers):
                is_active = (i == current_layer_idx)
                for p in layer_params:
                    p.requires_grad = is_active
        
        print(f"Epoch {epoch+1}/{config['epochs']} | {mode}")
        
        for data, target in train_loader:
            data, target = data.view(-1, 28*28).to(config["device"]), target.to(config["device"])
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        acc = evaluate(model, test_loader, config["device"])
        results.append(acc)
        print(f"  -> Test Acc: {acc:.2f}%")
        
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
    
    phased_res = train_frozen_gating_rr(train_loader, test_loader, CONFIG)
    
    print("\n--- Summary ---")
    for i, acc in enumerate(phased_res):
        print(f"Epoch {i+1}: {acc:.2f}%")
        
    # Save results
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v251i_frozen_gating_rr.json", "w") as f:
        json.dump(phased_res, f, indent=4)

if __name__ == "__main__":
    main()
