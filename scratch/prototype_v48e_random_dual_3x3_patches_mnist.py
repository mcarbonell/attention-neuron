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

class Dual3x3PatchFrozenMLP(nn.Module):
    """
    V48e: Dual Random 3x3 Patches (Constant Weights).
    Each of the 2048 neurons sums the pixels of TWO random 3x3 patches.
    Weights are fixed at 1.0 (frozen).
    """
    def __init__(self, projection_size=2048, hidden_size=512, output_size=10, device='cpu'):
        super().__init__()
        self.projection_size = projection_size
        
        # Input 32x32 = 1024
        self.frozen_layer = nn.Linear(1024, projection_size)
        
        print(f"Generating Dual Random 3x3 Patches (2x3x3) for {projection_size} neurons...")
        weights = torch.zeros(projection_size, 1024)
        
        for i in range(projection_size):
            # Pick two random 3x3 patches
            # 32x32 grid, max top/left is 29 (32-3)
            for _ in range(2):
                top = random.randint(0, 29)
                left = random.randint(0, 29)
                
                for r in range(top, top + 3):
                    for c in range(left, left + 3):
                        idx = r * 32 + c
                        weights[i, idx] = 1.0

        # Scale weights: each neuron sums 18 pixels
        # sqrt(18) approx 4.24
        weights = weights / 4.25
        
        self.frozen_layer.weight.data = weights.to(device)
        nn.init.zeros_(self.frozen_layer.bias)
        self.frozen_layer.weight.requires_grad = False
        self.frozen_layer.bias.requires_grad = False
        
        # Deep Trainable layers
        self.hidden = nn.Linear(projection_size, hidden_size)
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
    print(f"--- V48e: DUAL RANDOM 3x3 PATCHES (SUM) ---")
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

    model = Dual3x3PatchFrozenMLP(device=device).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

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
