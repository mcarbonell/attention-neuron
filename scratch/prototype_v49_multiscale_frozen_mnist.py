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

class MultiscaleFrozenMLP(nn.Module):
    """
    V49: Multiscale Local Projections.
    Architecture: 784 -> 2048 (FROZEN MULTISCALE) -> 10 (TRAINABLE)
    The frozen layer is divided into groups with different patch sizes.
    """
    def __init__(self, input_size=784, projection_size=2048, output_size=10, device='cpu'):
        super().__init__()
        self.frozen_layer = nn.Linear(input_size, projection_size)
        
        print(f"Generating Multiscale Projection ({projection_size} neurons)...")
        weights = torch.zeros(projection_size, input_size)
        
        # We split the 2048 neurons into 4 groups of 512
        chunk = projection_size // 4
        
        scales = [
            (4, 4),   # Micro
            (8, 8),   # Small
            (16, 16), # Medium
            (28, 28)  # Global
        ]
        
        for i, (p_h, p_w) in enumerate(scales):
            start_idx = i * chunk
            end_idx = (i + 1) * chunk
            
            for j in range(start_idx, end_idx):
                # Random position
                top = random.randint(0, 28 - p_h)
                left = random.randint(0, 28 - p_w)
                
                # Mask
                mask = torch.zeros(28, 28)
                mask[top:top+p_h, left:left+p_w] = 1.0
                
                # Random weights
                neuron_weights = torch.randn(784)
                weights[j] = neuron_weights * mask.view(-1)

        # Normalization: Healthy variance for ReLU
        target_std = math.sqrt(2.0 / input_size)
        current_std = weights[weights != 0].std()
        weights = weights * (target_std / (current_std + 1e-8))
        
        self.frozen_layer.weight.data = weights.to(device)
        nn.init.zeros_(self.frozen_layer.bias)
        self.frozen_layer.weight.requires_grad = False
        self.frozen_layer.bias.requires_grad = False
        
        # Linear Readout (Stay with low parameters)
        self.classifier = nn.Linear(projection_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        with torch.no_grad():
            x = torch.relu(self.frozen_layer(x))
        x = self.classifier(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V49: MULTISCALE FROZEN PROJECTION ---")
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

    model = MultiscaleFrozenMLP(device=device).to(device)
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
    with open("results/raw/v49_multiscale_frozen.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
