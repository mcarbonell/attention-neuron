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
    "max_lr": 5e-2,
    "device": "cpu",
    "seed": 42,
    "hidden_dims": [1024, 2048, 4096]
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
    print(f"\n--- Testing Dim: {hidden_dim} | Scheduler: OneCycleLR (MaxLR: {config['max_lr']}) ---")
    
    model = GatedMLP(784, hidden_dim, 10).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=config["max_lr"]/10) # Initial LR for Adam
    
    # Setup OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config["max_lr"], 
        steps_per_epoch=len(train_loader), 
        epochs=config["epochs"]
    )
    
    criterion = nn.CrossEntropyLoss()
    
    epoch_results = []
    for epoch in range(config["epochs"]):
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        for data, target in train_loader:
            data, target = data.view(-1, 28*28).to(config["device"]), target.to(config["device"])
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        test_acc = evaluate(model, test_loader, config["device"])
        epoch_results.append(test_acc)
        print(f"Epoch {epoch+1}/{config['epochs']} - LR: {current_lr:.4f} - Test Acc: {test_acc:.2f}%")
        
    return {
        "hidden_dim": hidden_dim,
        "final_accuracy": epoch_results[-1],
        "epoch_accuracies": epoch_results,
        "max_lr": config["max_lr"]
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
    with open("results/raw/v251e_scheduled_gating.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n--- Scheduled Gating Summary ---")
    print(f"{'Hidden Dim':<10} | {'Final Acc (%)':<15}")
    print("-" * 30)
    for r in results:
        print(f"{r['hidden_dim']:<10} | {r['final_accuracy']:<15.2f}")

if __name__ == "__main__":
    main()
