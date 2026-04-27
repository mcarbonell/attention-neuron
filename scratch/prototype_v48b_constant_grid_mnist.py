import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os
import math
import random

class ConstantGridFrozenMLP(nn.Module):
    """
    V48b: Constant Grid Frozen Projection.
    - Image: 28x28 padded to 32x32.
    - Grid: 8x8 cells of size 4x4 (Total 64 cells).
    - Neurons: 32 projections per cell, ALL WEIGHTS = 1.0.
    - Effectively, this is a 4x4 Sum-Pooling feature extractor.
    """
    def __init__(self, projection_per_patch=32, hidden_size=512, output_size=10, device='cpu'):
        super().__init__()
        self.num_patches = 64 
        self.projection_size = self.num_patches * projection_per_patch # 2048
        
        self.frozen_layer = nn.Linear(1024, self.projection_size)
        
        print(f"Generating Constant Grid (All 1s): 64 patches x {projection_per_patch} neurons.")
        weights = torch.zeros(self.projection_size, 1024)
        
        for p in range(self.num_patches):
            row = (p // 8) * 4
            col = (p % 8) * 4
            
            patch_indices = []
            for r in range(row, row + 4):
                for c in range(col, col + 4):
                    patch_indices.append(r * 32 + c)
            
            start_idx = p * projection_per_patch
            end_idx = start_idx + projection_per_patch
            
            # ALL WEIGHTS = 1.0 for these neurons in their respective patch
            for p_idx in patch_indices:
                weights[start_idx:end_idx, p_idx] = 1.0

        # We won't normalize to Kaiming here because the user specifically asked for "1s"
        # But we might need a small scale to prevent huge activations (16 * max_pixel_val)
        # Let's keep it as 1.0 but divide by 4 (sqrt of patch area) to keep variance sane
        weights = weights / 4.0 
        
        self.frozen_layer.weight.data = weights.to(device)
        nn.init.zeros_(self.frozen_layer.bias)
        self.frozen_layer.weight.requires_grad = False
        self.frozen_layer.bias.requires_grad = False
        
        # Deep Trainable layers
        self.hidden = nn.Linear(self.projection_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (2, 2, 2, 2))
        x = x.view(x.size(0), -1)
        with torch.no_grad():
            x = torch.relu(self.frozen_layer(x))
        x = torch.relu(self.hidden(x))
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V48b: CONSTANT GRID (SUM-POOLING) ---")
    print(f"Device: {device}")

    BATCH_SIZE = 256
    EPOCHS = 20
    LR = 0.001
    SEED = 42
    
    torch.manual_seed(SEED)
    random.seed(SEED)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1024)

    model = ConstantGridFrozenMLP(device=device).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        acc = correct / 10000
        print(f"Epoch {epoch:2d} | Acc: {acc:.4f} | Time: {time.time()-t0:.1f}s")

    print(f"\n🚀 Final Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
