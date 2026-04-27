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
# GABOR KERNEL GENERATOR
# =============================================================================

def get_gabor_kernel(size, sigma, theta, lambd, gamma, psi, device='cpu'):
    """Generates a Gabor kernel of (size, size)."""
    d0 = torch.linspace(-(size // 2), size // 2, size, device=device)
    y, x = torch.meshgrid(d0, d0, indexing='ij')
    
    # Rotation
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = -x * math.sin(theta) + y * math.cos(theta)
    
    # Gabor formula
    # exp(- (x'^2 + gamma^2 * y'^2) / (2 * sigma^2)) * cos(2*pi * x' / lambd + psi)
    gb = torch.exp(-0.5 * (x_theta**2 + gamma**2 * y_theta**2) / (sigma**2)) * \
         torch.cos(2 * math.pi * x_theta / lambd + psi)
    
    return gb

# =============================================================================
# MODEL
# =============================================================================

class GaborFrozenMLP(nn.Module):
    """
    V43: Gabor Structured Frozen Projection.
    Each of the 2048 neurons in the first layer is initialized with 
    a Gabor filter (edge/texture detector) with random orientation and frequency.
    """
    def __init__(self, input_size=784, hidden_size=2048, output_size=10, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.frozen_layer = nn.Linear(input_size, hidden_size)
        
        print(f"Generating {hidden_size} Gabor kernels...")
        gabor_weights = torch.zeros(hidden_size, input_size)
        
        for i in range(hidden_size):
            # Randomize Gabor parameters for diversity
            theta = random.uniform(0, math.pi)          # Orientation
            sigma = random.uniform(1.0, 4.0)           # Spread
            lambd = random.uniform(3.0, 10.0)          # Wavelength (thickness)
            gamma = random.uniform(0.5, 1.0)           # Aspect ratio
            psi   = random.uniform(0, math.pi)         # Phase offset
            
            kernel = get_gabor_kernel(28, sigma, theta, lambd, gamma, psi, device='cpu')
            gabor_weights[i] = kernel.reshape(-1)
            
        # Normalize weights to have Kaiming-like variance
        target_std = math.sqrt(2.0 / input_size)
        current_std = gabor_weights.std()
        gabor_weights = gabor_weights * (target_std / (current_std + 1e-8))
        
        self.frozen_layer.weight.data = gabor_weights.to(device)
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
    print(f"--- V43: GABOR STRUCTURED FROZEN PROJECTION ---")
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
    model = GaborFrozenMLP(hidden_size=HIDDEN_SIZE, device=device).to(device)
    
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
    with open("results/raw/v43_gabor_frozen.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
