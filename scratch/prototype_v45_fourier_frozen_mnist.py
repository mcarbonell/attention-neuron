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

# =============================================================================
# FOURIER (SINE/COSINE) KERNEL GENERATOR
# =============================================================================

def get_fourier_kernel(size, u, v, phase, device='cpu'):
    """Generates a 2D Sine/Cosine wave kernel of (size, size)."""
    d0 = torch.linspace(0, 1, size, device=device)
    y, x = torch.meshgrid(d0, d0, indexing='ij')
    
    # 2D Wave formula: cos(2*pi*(u*x + v*y) + phase)
    wave = torch.cos(2 * math.pi * (u * x + v * y) + phase)
    
    return wave

# =============================================================================
# MODEL
# =============================================================================

class FourierFrozenMLP(nn.Module):
    """
    V45: Fourier Basis Frozen Projection.
    Each of the 2048 neurons is a random 2D sine/cosine wave.
    This acts as a fixed frequency feature extractor.
    """
    def __init__(self, input_size=784, hidden_size=2048, output_size=10, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.frozen_layer = nn.Linear(input_size, hidden_size)
        
        print(f"Generating {hidden_size} Fourier kernels...")
        fourier_weights = torch.zeros(hidden_size, input_size)
        
        for i in range(hidden_size):
            # Random frequencies (u, v) and phase
            # Low frequencies are usually more important for MNIST
            u = random.uniform(-5.0, 5.0) 
            v = random.uniform(-5.0, 5.0)
            phase = random.uniform(0, 2 * math.pi)
            
            kernel = get_fourier_kernel(28, u, v, phase, device='cpu')
            fourier_weights[i] = kernel.reshape(-1)
            
        # Normalize weights to have Kaiming-like variance
        target_std = math.sqrt(2.0 / input_size)
        current_std = fourier_weights.std()
        fourier_weights = fourier_weights * (target_std / (current_std + 1e-8))
        
        self.frozen_layer.weight.data = fourier_weights.to(device)
        nn.init.zeros_(self.frozen_layer.bias)
        
        # Freeze
        self.frozen_layer.weight.requires_grad = False
        self.frozen_layer.bias.requires_grad = False
        
        # Trainable readout
        self.trainable_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        with torch.no_grad():
            x = self.frozen_layer(x)
            x = torch.relu(x)
        x = self.trainable_layer(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V45: FOURIER BASIS FROZEN PROJECTION ---")
    print(f"Device: {device}")

    # Hyperparameters
    HIDDEN_SIZE = 2048
    BATCH_SIZE = 256
    EPOCHS = 20
    LR = 0.001
    SEED = 42
    
    torch.manual_seed(SEED)
    random.seed(SEED)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1024)

    # Model
    model = FourierFrozenMLP(hidden_size=HIDDEN_SIZE, device=device).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    metrics = {"history": []}
    t_start = time.time()

    print("Starting training...")
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
            
            if epoch == 1 and batch_idx < 5:
                print(f"  > Batch {batch_idx} | Loss: {loss.item():.4f}")

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

    t_end = time.time()
    print(f"\n🚀 Final Accuracy: {acc:.4f} | Total Time: {t_end - t_start:.1f}s")
    
    # Save results
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v45_fourier_frozen.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
