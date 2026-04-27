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

class OnOffCenterFrozenMLP(nn.Module):
    """
    V48j: On-Center / Off-Surround Frozen Projection.
    Each of the 2048 neurons is a local contrast detector:
    - 3x3 Center: +1.0
    - 5x5 Surround: -0.5
    """
    def __init__(self, projection_size=2048, hidden_size=512, output_size=10, device='cpu'):
        super().__init__()
        self.projection_size = projection_size
        
        # Input 32x32 = 1024
        self.frozen_layer = nn.Linear(1024, projection_size)
        
        print(f"Generating On-Center/Off-Surround Filters for {projection_size} neurons...")
        weights = torch.zeros(projection_size, 1024)
        
        for i in range(projection_size):
            # Pick a random center for the 5x5 block
            # 32x32 grid, 5x5 block needs margin of 2
            top = random.randint(0, 27)
            left = random.randint(0, 27)
            
            # Fill 5x5 Surround with -0.5
            for r in range(top, top + 5):
                for c in range(left, left + 5):
                    weights[i, r * 32 + c] = -0.5
            
            # Fill 3x3 Center with +1.0 (overwriting the center of the 5x5)
            # Center of top:top+5 is top+1:top+4
            for r in range(top + 1, top + 4):
                for c in range(left + 1, left + 4):
                    weights[i, r * 32 + c] = 1.0

        # Scale weights for stability
        # Each neuron has 9*(1.0) and 16*(-0.5) -> sum is 1.0. 
        # Standard deviation is roughly sqrt(9*1 + 16*0.25)/sqrt(1024)
        weights = weights / 3.6
        
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
    print(f"--- V48j: ON-CENTER / OFF-SURROUND ---")
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

    model = OnOffCenterFrozenMLP(device=device).to(device)
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
