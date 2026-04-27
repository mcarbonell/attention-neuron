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
# PERLIN NOISE GENERATOR (Optimized for Weight Init)
# =============================================================================

def generate_perlin_2d(shape, scale=1.0, octaves=1, persistence=0.5, device='cpu'):
    H, W = shape
    noise = torch.zeros(H, W, device=device)
    amplitude = 1.0
    frequency = scale
    
    for _ in range(octaves):
        grid_h = max(2, int(H * frequency))
        grid_w = max(2, int(W * frequency))
        
        grads = torch.randn(grid_h, grid_w, 2, device=device)
        grads = grads / (grads.norm(dim=-1, keepdim=True) + 1e-8)
        
        y = torch.linspace(0, grid_h - 1, H, device=device)
        x = torch.linspace(0, grid_w - 1, W, device=device)
        gy = torch.floor(y).long()
        gx = torch.floor(x).long()
        
        fy = (y - gy).unsqueeze(1).expand(H, W)
        fx = (x - gx).unsqueeze(0).expand(H, W)
        
        sy = fy * fy * (3 - 2 * fy)
        sx = fx * fx * (3 - 2 * fx)
        
        gy0 = gy.clamp(0, grid_h - 1)
        gy1 = (gy0 + 1).clamp(0, grid_h - 1)
        gx0 = gx.clamp(0, grid_w - 1)
        gx1 = (gx0 + 1).clamp(0, grid_w - 1)
        
        # Corner dot products
        g00 = grads[gy0[:, None], gx0[None, :]]
        d00 = g00[..., 0] * fx + g00[..., 1] * fy
        
        g10 = grads[gy0[:, None], gx1[None, :]]
        d10 = g10[..., 0] * (fx - 1) + g10[..., 1] * fy
        
        g01 = grads[gy1[:, None], gx0[None, :]]
        d01 = g01[..., 0] * fx + g01[..., 1] * (fy - 1)
        
        g11 = grads[gy1[:, None], gx1[None, :]]
        d11 = g11[..., 0] * (fx - 1) + g11[..., 1] * (fy - 1)
        
        n0 = d00 * (1 - sx) + d10 * sx
        n1 = d01 * (1 - sx) + d11 * sx
        layer = n0 * (1 - sy) + n1 * sy
        
        noise += layer * amplitude
        amplitude *= persistence
        frequency *= 2
    
    return noise

# =============================================================================
# MODEL
# =============================================================================

class PerlinFrozenMLP(nn.Module):
    """
    V42: Perlin Structured Frozen Projection.
    Each of the 2048 neurons in the first layer is initialized with 
    a different Perlin noise pattern (structured noise).
    """
    def __init__(self, input_size=784, hidden_size=2048, output_size=10, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.frozen_layer = nn.Linear(input_size, hidden_size)
        
        # Generate Perlin Weights
        print(f"Generating {hidden_size} Perlin kernels...")
        perlin_weights = torch.zeros(hidden_size, input_size)
        
        # Diversity of scales and octaves
        for i in range(hidden_size):
            # Assign different characteristics to different neurons
            if i < hidden_size // 4:
                s, oct = 0.1, 1 # Global structures
            elif i < hidden_size // 2:
                s, oct = 0.25, 2 # Medium textures
            elif i < 3 * hidden_size // 4:
                s, oct = 0.5, 3 # Fine details
            else:
                s, oct = random.uniform(0.1, 0.7), random.randint(1, 4) # Random mix
            
            p_noise = generate_perlin_2d((28, 28), scale=s, octaves=oct, device=device)
            perlin_weights[i] = p_noise.view(-1)
            
        # Normalize weights to have a healthy variance (like Kaiming)
        # Standard deviation for Kaiming is sqrt(2/input_size)
        target_std = math.sqrt(2.0 / input_size)
        current_std = perlin_weights.std()
        perlin_weights = perlin_weights * (target_std / (current_std + 1e-8))
        
        self.frozen_layer.weight.data = perlin_weights.to(device)
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
    print(f"--- V42: PERLIN STRUCTURED FROZEN PROJECTION ---")
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
    model = PerlinFrozenMLP(hidden_size=HIDDEN_SIZE, device=device).to(device)
    
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
    with open("results/raw/v42_perlin_frozen.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
