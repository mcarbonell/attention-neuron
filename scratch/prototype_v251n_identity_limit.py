import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
CONFIG = {
    "batch_size": 128,
    "epochs": 10,
    "max_lr": 0.05,
    "device": "cpu",
    "seed": 42,
    "hidden_dim": 4096
}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features, init_val=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gate = nn.Parameter(torch.full((out_features,), init_val))
        for param in self.linear.parameters():
            param.requires_grad = False
                
    def forward(self, x):
        return self.linear(x) * self.gate

class GatedMLP_Identity(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, init_val=0.0):
        super().__init__()
        self.layer1 = GatedLinear(input_dim, hidden_dim, init_val)
        self.identity = nn.Identity() # NO ACTIVATION
        self.layer2 = GatedLinear(hidden_dim, output_dim, 1.0)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.identity(x)
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

def main():
    set_seed(CONFIG["seed"])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=CONFIG["batch_size"], shuffle=False)
    
    model = GatedMLP_Identity(784, CONFIG["hidden_dim"], 10, 0.0).to(CONFIG["device"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["max_lr"]/10)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG["max_lr"], steps_per_epoch=len(train_loader), epochs=CONFIG["epochs"])
    criterion = nn.CrossEntropyLoss()
    
    print("\n--- Running Trial (Identity Activation) with Gate Init = 0.0 ---")
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
        
        acc = evaluate(model, test_loader, CONFIG["device"])
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Test Acc: {acc:.2f}%")

if __name__ == "__main__":
    main()
