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

class GridPatchFrozenMLP(nn.Module):
    """
    V48: Grid Patch Frozen Projection.
    - Image: 28x28 padded to 32x32.
    - Grid: 8x8 cells of size 4x4 (Total 64 cells).
    - Neurons: 32 projections per cell (Total 2048 neurons).
    - Architecture: (32x32) -> 2048 (FROZEN GRID) -> 512 (TRAINABLE) -> 10 (TRAINABLE)
    """
    def __init__(self, projection_per_patch=32, hidden_size=512, output_size=10, device='cpu'):
        super().__init__()
        self.num_patches = 64 # 8x8 grid of 4x4 patches
        self.projection_size = self.num_patches * projection_per_patch # 2048
        
        # We'll implement the frozen layer as a sparse weight matrix for efficiency
        # Input size is 1024 (32x32)
        self.frozen_layer = nn.Linear(1024, self.projection_size)
        
        print(f"Generating Grid Projection: 64 patches x {projection_per_patch} neurons = {self.projection_size} total.")
        weights = torch.zeros(self.projection_size, 1024)
        
        for p in range(self.num_patches):
            # Calculate patch coordinates in 32x32 grid
            row = (p // 8) * 4
            col = (p % 8) * 4
            
            # Indices of the 16 pixels in this 4x4 patch
            patch_indices = []
            for r in range(row, row + 4):
                for c in range(col, col + 4):
                    patch_indices.append(r * 32 + c)
            
            # Assign 32 neurons to this specific patch
            start_idx = p * projection_per_patch
            end_idx = start_idx + projection_per_patch
            
            # Random weights ONLY for the patch pixels
            patch_weights = torch.randn(projection_per_patch, 16)
            
            # Fill the global weight matrix
            for i, p_idx in enumerate(patch_indices):
                weights[start_idx:end_idx, p_idx] = patch_weights[:, i]

        # Normalization
        target_std = math.sqrt(2.0 / 16) # Each neuron only has 16 inputs
        current_std = weights[weights != 0].std()
        weights = weights * (target_std / (current_std + 1e-8))
        
        self.frozen_layer.weight.data = weights.to(device)
        nn.init.zeros_(self.frozen_layer.bias)
        self.frozen_layer.weight.requires_grad = False
        self.frozen_layer.bias.requires_grad = False
        
        # Deep Trainable layers
        self.hidden = nn.Linear(self.projection_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 1. Pad 28x28 to 32x32
        x = torch.nn.functional.pad(x, (2, 2, 2, 2))
        x = x.view(x.size(0), -1)
        
        # 2. Frozen Grid Projection
        with torch.no_grad():
            x = torch.relu(self.frozen_layer(x))
            
        # 3. Trainable Readout
        x = torch.relu(self.hidden(x))
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V48: GRID PATCH FROZEN PROJECTION ---")
    print(f"Device: {device}")

    # Hyperparams
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

    model = GridPatchFrozenMLP(device=device).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    metrics = {"history": []}
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        acc = correct / 10000
        print(f"Epoch {epoch:2d} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Time: {time.time()-t0:.1f}s")
        metrics["history"].append({"epoch": epoch, "acc": acc})

    print(f"\n🚀 Final Accuracy: {acc:.4f} | Total Time: {time.time() - t_start:.1f}s")
    
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v48_grid_frozen.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
