import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import math
import os
import json
import numpy as np

# --- 1D Reordering Utilities ---

def get_spiral_indices(size=28):
    indices = []
    r, c = size // 2, size // 2
    indices.append(r * size + c)
    step = 1
    while len(indices) < size * size:
        for _ in range(step):
            c += 1
            if 0 <= r < size and 0 <= c < size: indices.append(r * size + c)
        for _ in range(step):
            r += 1
            if 0 <= r < size and 0 <= c < size: indices.append(r * size + c)
        step += 1
        for _ in range(step):
            c -= 1
            if 0 <= r < size and 0 <= c < size: indices.append(r * size + c)
        for _ in range(step):
            r -= 1
            if 0 <= r < size and 0 <= c < size: indices.append(r * size + c)
        step += 1
    return torch.tensor(indices[:size*size])

def get_column_indices(size=28):
    idx = torch.arange(size * size).view(size, size)
    return idx.t().contiguous().view(-1)

def get_rotated_indices(size=28, angle=45):
    y, x = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
    grid = torch.stack([x, y], dim=-1).view(1, size, size, 2)
    theta = math.radians(angle)
    rot_mat = torch.tensor([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0]], dtype=torch.float32).view(1, 2, 3)
    rot_grid = F.affine_grid(rot_mat, [1, 1, size, size], align_corners=True)
    coord_map = torch.arange(size * size).view(1, 1, size, size).float()
    rotated_map = F.grid_sample(coord_map, rot_grid, mode='nearest', align_corners=True)
    return rotated_map.view(-1).long()

# --- Stable Triangular Neuron ---

class TriangularNeuronLayer(nn.Module):
    def __init__(self, in_features, out_features, device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.raw_center = nn.Parameter(torch.rand(out_features, 1))
        self.raw_width = nn.Parameter(torch.rand(out_features, 1) * 0.1 + 0.02)
        self.register_buffer("indices", torch.linspace(0, 1, in_features).view(1, -1))

    def get_masks(self):
        # CONSTRAINT: Width >= 2 pixels to maintain gradients
        min_w = 2.0 / self.in_features
        width = F.softplus(self.raw_width) + min_w
        
        dist = torch.abs(self.indices - self.raw_center)
        masks = F.relu(1.0 - dist / width)
        mask_sums = masks.sum(dim=1, keepdim=True) + 1e-8
        return masks / mask_sums

    def forward(self, x):
        masks = self.get_masks()
        return x @ masks.t()

# --- Omni-View Stable Model ---

class StableNeedleNet(nn.Module):
    def __init__(self, hidden1=1024, hidden2=512, device='cpu'):
        super().__init__()
        self.register_buffer("idx_rows", torch.arange(784))
        self.register_buffer("idx_cols", get_column_indices(28))
        self.register_buffer("idx_d45p", get_rotated_indices(28, 45))
        self.register_buffer("idx_d45m", get_rotated_indices(28, -45))
        self.register_buffer("idx_spir", get_spiral_indices(28))
        
        self.layer1 = TriangularNeuronLayer(3920, hidden1, device=device)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.layer2 = TriangularNeuronLayer(hidden1, hidden2, device=device)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.classifier = nn.Linear(hidden2, 10)
        
    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)
        x_omni = torch.cat([x[:, self.idx_rows], x[:, self.idx_cols], x[:, self.idx_d45p], x[:, self.idx_d45m], x[:, self.idx_spir]], dim=1)
        x = F.relu(self.bn1(self.layer1(x_omni)))
        x = F.relu(self.bn2(self.layer2(x)))
        return self.classifier(x)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V100: STABLE NEEDLE (Width Constraint) ---")
    print(f"Constraint: width >= 2 pixels | Epochs: 5 | Device: {device}")
    
    LR = 0.005
    EPOCHS = 5
    BATCH_SIZE = 128
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1024)
    
    model = StableNeedleNet(device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = correct / 10000
        print(f"Epoch {epoch:2d} | Loss: {loss.item():.4f} | Acc: {acc:.4f} | Time: {time.time()-t0:.1f}s")

    print(f"\nFinal Stable Acc: {acc:.4f} | Total Time: {time.time()-t_start:.1f}s")

if __name__ == "__main__":
    main()
