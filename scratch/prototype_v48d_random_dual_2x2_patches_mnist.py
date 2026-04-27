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

class DualMicroPatchFrozenMLP(nn.Module):
    """
    V48d: Dual Random 2x2 Patches (Constant Weights).
    Each of the 2048 neurons sums the pixels of TWO random 2x2 patches.
    Very sparse feature extraction.
    """
    def __init__(self, projection_size=2048, hidden_size=512, output_size=10, device='cpu'):
        super().__init__()
        self.projection_size = projection_size
        
        # Input 32x32 = 1024
        self.frozen_layer = nn.Linear(1024, projection_size)
        
        print(f"Generating Dual Random Micro-Patches (2x2x2) for {projection_size} neurons...")
        weights = torch.zeros(projection_size, 1024)
        
        for i in range(projection_size):
            # Pick two random 2x2 patches
            # 32x32 grid, max top/left is 30
            for _ in range(2):
                top = random.randint(0, 30)
                left = random.randint(0, 30)
                
                for r in range(top, top + 2):
                    for c in range(left, left + 2):
                        idx = r * 32 + c
                        weights[i, idx] = 1.0

        # Scale weights: each neuron sums 8 pixels
        # sqrt(8) approx 2.82
        weights = weights / 3.0
        
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
    print(f"--- V48d: DUAL RANDOM MICRO-PATCHES (2x2) ---")
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

    model = DualMicroPatchFrozenMLP(device=device).to(device)
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
