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
    "conv1_filters": 256,
    "conv2_filters": 512,
    "device": "cpu",
    "seed": 42
}

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# --- Architecture Components ---
class TernaryConv2dGated(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, init_val=0.0):
        super().__init__()
        shape = (out_channels, in_channels, kernel_size, kernel_size)
        weights = torch.randint(-1, 2, shape).float()
        self.register_buffer("weight", weights)
        self.gate = nn.Parameter(torch.full((out_channels,), float(init_val)))
        
    def forward(self, x):
        x = torch.nn.functional.conv2d(x, self.weight, padding=self.weight.shape[-1]//2)
        return x * self.gate.view(1, -1, 1, 1)

class TernaryLinearGated(nn.Module):
    def __init__(self, in_features, out_features, init_val=1.0):
        super().__init__()
        weights = torch.randint(-1, 2, (out_features, in_features)).float()
        self.register_buffer("weight", weights)
        self.gate = nn.Parameter(torch.full((out_features,), float(init_val)))
        
    def forward(self, x):
        x = torch.matmul(x, self.weight.t())
        return x * self.gate

class GatedTernaryCNN_GAP(nn.Module):
    def __init__(self, conv1_filters, conv2_filters):
        super().__init__()
        self.conv1 = TernaryConv2dGated(1, conv1_filters, kernel_size=5, init_val=0.0)
        self.silu1 = nn.SiLU()
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = TernaryConv2dGated(conv1_filters, conv2_filters, kernel_size=5, init_val=1.0)
        self.silu2 = nn.SiLU()
        
        # --- NEW: Global Average Pooling ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # FC is now much smaller: only conv2_filters inputs
        self.fc = TernaryLinearGated(conv2_filters, 10, init_val=1.0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.silu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.silu2(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def print_architecture(self):
        print("\n--- Network Architecture (Gated Ternary CNN + GAP) ---")
        total_frozen = sum(p.numel() for p in self.buffers())
        total_learnable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Conv1: {CONFIG['conv1_filters']} filters")
        print(f"Conv2: {CONFIG['conv2_filters']} filters")
        print(f"GAP Layer Enabled")
        print(f"FC:    {10} outputs")
        print(f"Frozen Params:    {total_frozen:,}")
        print(f"Learnable Gates:  {total_learnable:,}")
        print("---------------------------\n")

# --- Training / Eval ---
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
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
            data, target = data.to(config["device"]), target.to(config["device"])
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
    
    model = GatedTernaryCNN_GAP(CONFIG["conv1_filters"], CONFIG["conv2_filters"]).to(CONFIG["device"])
    res, duration = train_model(model, train_loader, test_loader, CONFIG)
    
    # Save
    os.makedirs("results/raw", exist_ok=True)
    report = {"config": CONFIG, "history": res, "duration": duration}
    with open("results/raw/v257_ternary_cnn_gap.json", "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nResults saved to results/raw/v257_ternary_cnn_gap.json")

if __name__ == "__main__":
    main()
