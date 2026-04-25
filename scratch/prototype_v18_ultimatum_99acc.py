import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import json
import os

class AttentionNeuronLayerV18(nn.Module):
    def __init__(self, in_features, out_features, rank=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Fixed base weights (Kaiming Normal)
        std = math.sqrt(2.0 / in_features)
        self.register_buffer('w_init', torch.randn(out_features, in_features) * std)
        
        # Dual modulation (rank-r)
        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.delta_in_a = nn.Parameter(torch.zeros(out_features, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_features))
        
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        
        w_evolved = self.w_init * (1.0 + w_m) + w_a
        
        y = torch.matmul(x, w_evolved.t()) + self.bias
        return self.bn(y)

class UltimatumNetV18(nn.Module):
    def __init__(self):
        super().__init__()
        # Rank 128 for the first layer (Feature extraction powerhouse)
        self.layer1 = AttentionNeuronLayerV18(784, 2048, rank=128)
        self.layer2 = AttentionNeuronLayerV18(2048, 1024, rank=64)
        self.layer3 = AttentionNeuronLayerV18(1024, 10, rank=64)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784)
        # Optional: Add small input noise to improve robustness
        if self.training:
            x = x + torch.randn_like(x) * 0.02
            
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V18 'THE ULTIMATUM' (Goal: 99%+) on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 60 # More time to reach the summit
    MAX_LR = 0.003 # Slightly lower for more stability
    
    train_transform = transforms.Compose([
        transforms.RandomRotation(12), # Slightly more rotation
        transforms.RandomAffine(degrees=0, translate=(0.12, 0.12), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    model = UltimatumNetV18().to(device)
    params = count_parameters(model)
    print(f"Trainable Parameters: {params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR / 10, weight_decay=0.05)
    
    # Calculate total steps explicitly for robustness
    total_steps = len(train_loader) * EPOCHS
    
    # 40% of the time cooling down for precise convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=total_steps,
        pct_start=0.3, anneal_strategy='cos'
    )
    
    # Label Smoothing to help generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0
    t_start = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = correct / 10000
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "results/v18_ultimatum_best.pt")
            
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f}")
        
    t_total = time.time() - t_start
    print(f"\nULTIMATUM finished in {t_total:.1f}s")
    print(f"Final Best Test Accuracy: {best_acc:.4f}")
    
    results = {
        "model": "V18_Ultimatum",
        "rank_l1": 128,
        "trainable_params": params,
        "best_acc": best_acc,
        "epochs": EPOCHS,
        "wall_clock_time": t_total,
        "dataset": "MNIST",
        "augmentation": "Aggressive",
        "label_smoothing": 0.1
    }
    
    with open("results/raw/v18_ultimatum_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
