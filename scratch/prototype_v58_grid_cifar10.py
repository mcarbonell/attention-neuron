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
        
        # GRID INITIALIZATION
        # 512 neurons -> 16 rows x 32 columns
        cols = 32
        rows = 16
        x_coords = torch.linspace(2, 30, cols)
        y_coords = torch.linspace(2, 30, rows)
        
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        centers = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2) # (512, 2)
        
        # Initialize as small vertical segments
        init_points = torch.zeros(num_neurons, 2, 2)
        init_points[:, 0, :] = centers - torch.tensor([0.0, 1.5])
        init_points[:, 1, :] = centers + torch.tensor([0.0, 1.5])
        self.points = nn.Parameter(init_points)
        
        # Random color weights to break color symmetry
        self.color_weights = nn.Parameter(torch.randn(num_neurons, 3))
        
        self.log_sigma_pos = nn.Parameter(torch.full((num_neurons,), math.log(1.5)))
        self.log_sigma_neg = nn.Parameter(torch.full((num_neurons,), math.log(3.5)))
        
        y, x = torch.meshgrid(torch.linspace(0, 31, 32), torch.linspace(0, 31, 32), indexing='ij')
        self.register_buffer("grid", torch.stack([x, y], dim=-1).view(1, 1024, 2))

    def get_masks(self):
        t = torch.linspace(0, 1, 10, device=self.points.device).view(1, 10, 1)
        p0, p1 = self.points[:, 0:1, :], self.points[:, 1:2, :]
        line_points = (1-t) * p0 + t * p1
        
        diff = self.grid.unsqueeze(0) - line_points.unsqueeze(2)
        dist_sq = torch.sum(diff**2, dim=-1)
        min_dist_sq, _ = torch.min(dist_sq, dim=1)
        
        sigma_pos = torch.exp(self.log_sigma_pos).view(-1, 1)
        sigma_neg = torch.exp(self.log_sigma_neg).view(-1, 1)
        
        stroke = torch.exp(-min_dist_sq / (2 * sigma_pos**2))
        surround = torch.exp(-min_dist_sq / (2 * sigma_neg**2))
        
        return stroke - 0.4 * surround

    def forward(self, x):
        B = x.size(0)
        masks = self.get_masks()
        x_flat = x.view(B, 3, 1024)
        channel_proj = torch.matmul(x_flat, masks.t())
        w = self.color_weights.t().unsqueeze(0)
        h = torch.sum(channel_proj * w, dim=1)
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

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V58: GRID INITIALIZATION ON CIFAR-10 ---")
    
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

    model = CIFARMatchNet(num_neurons=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, 11):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            with torch.no_grad(): model.match_layer.points.clamp_(0, 31.9)

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = correct / 10000
        history.append(acc)
        print(f"Epoch {epoch:2d} | Acc: {acc:.4f}")

    os.makedirs("results/raw", exist_ok=True)
    metrics = {"experiment": "v58_grid_cifar", "best_objective": max(history), "history_acc": history}
    with open("results/raw/v58_grid_cifar.json", "w") as f: json.dump(metrics, f, indent=4)
    torch.save(model.state_dict(), "v58_grid_cifar.pth")
    print("✅ Model saved as v58_grid_cifar.pth")

if __name__ == "__main__":
    train()
