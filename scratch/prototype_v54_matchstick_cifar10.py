import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import math
import json
import os

class RGBMatchstickLayer(nn.Module):
    def __init__(self, num_neurons=512):
        super().__init__()
        self.num_neurons = num_neurons
        self.size = 32
        
        # Geometry: (N, 2, 2) -> p0, p1
        self.points = nn.Parameter(torch.rand(num_neurons, 2, 2) * 31.0)
        
        # Color Sensitivity: (N, 3) -> Weights for R, G, B
        # Initialized to random unit vectors to start with diverse color focus
        self.color_weights = nn.Parameter(torch.randn(num_neurons, 3))
        
        # Stroke Widths
        self.log_sigma_pos = nn.Parameter(torch.full((num_neurons,), math.log(1.5)))
        self.log_sigma_neg = nn.Parameter(torch.full((num_neurons,), math.log(3.5)))
        
        y, x = torch.meshgrid(torch.linspace(0, 31, 32), torch.linspace(0, 31, 32), indexing='ij')
        self.register_buffer("grid", torch.stack([x, y], dim=-1).view(1, 1024, 2))

    def get_masks(self):
        # Sample 10 points along the segment
        t = torch.linspace(0, 1, 10, device=self.points.device).view(1, 10, 1)
        p0, p1 = self.points[:, 0:1, :], self.points[:, 1:2, :]
        line_points = (1-t) * p0 + t * p1 # (N, 10, 2)
        
        diff = self.grid.unsqueeze(0) - line_points.unsqueeze(2) # (N, 10, 1024, 2)
        dist_sq = torch.sum(diff**2, dim=-1) # (N, 10, 1024)
        min_dist_sq, _ = torch.min(dist_sq, dim=1) # (N, 1024)
        
        sigma_pos = torch.exp(self.log_sigma_pos).view(-1, 1)
        sigma_neg = torch.exp(self.log_sigma_neg).view(-1, 1)
        
        stroke = torch.exp(-min_dist_sq / (2 * sigma_pos**2))
        surround = torch.exp(-min_dist_sq / (2 * sigma_neg**2))
        
        return stroke - 0.4 * surround

    def forward(self, x):
        B = x.size(0)
        masks = self.get_masks() # (N, 1024)
        
        x_flat = x.view(B, 3, 1024) # CIFAR: (B, 3, 32, 32)
        
        # 1. Project each channel onto the matchstick masks
        # (B, 3, 1024) @ (1024, N) -> (B, 3, N)
        channel_proj = torch.matmul(x_flat, masks.t())
        
        # 2. Apply learnable color sensitivity
        # color_weights: (N, 3) -> (1, 3, N)
        w = self.color_weights.t().unsqueeze(0)
        h = torch.sum(channel_proj * w, dim=1) # (B, N)
        
        return h

class CIFARMatchNet(nn.Module):
    def __init__(self, num_neurons=512):
        super().__init__()
        self.match_layer = RGBMatchstickLayer(num_neurons=num_neurons)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        h = self.match_layer(x)
        return self.classifier(h)

def train_cifar():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V54: RGB MATCHSTICKS ON CIFAR-10 ---")
    print(f"Device: {device}")
    
    NUM_NEURONS = 512
    EPOCHS = 10
    LR = 0.002
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_loader = DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transform_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.CIFAR10('./data', train=False, transform=transform_test), batch_size=512)

    model = CIFARMatchNet(num_neurons=NUM_NEURONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    start_wall = time.time()
    
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                model.match_layer.points.clamp_(0, 31.9)
            
            epoch_loss += loss.item()
            if batch_idx == 0:
                print(f"  Epoch {epoch} | Initial Loss: {loss.item():.4f}")

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = correct / 10000
        history.append(acc)
        print(f"Epoch {epoch:2d} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Time: {time.time()-epoch_start:.2f}s")

    total_time = time.time() - start_wall
    
    metrics = {
        "experiment": "v54_matchstick_cifar10",
        "best_objective": max(history),
        "wall_clock_time": total_time,
        "num_neurons": NUM_NEURONS,
        "history_acc": history
    }
    
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v54_matchstick_cifar10.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Save weights
    torch.save(model.state_dict(), "v54_matchsticks_cifar.pth")
    print("✅ Model saved as v54_matchsticks_cifar.pth")
    
    print(f"\n✅ Sondeo rápido completado. Best Acc: {max(history):.4f}")

if __name__ == "__main__":
    train_cifar()
