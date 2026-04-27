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
# PERLIN GENERATOR
# =============================================================================

def generate_perlin_2d(shape, scale=1.0, octaves=1, persistence=0.5, device='cpu'):
    H, W = shape
    noise = torch.zeros(H, W, device=device)
    amplitude = 1.0
    frequency = scale
    for _ in range(octaves):
        grid_h, grid_w = max(2, int(H * frequency)), max(2, int(W * frequency))
        grads = torch.randn(grid_h, grid_w, 2, device=device)
        grads /= (grads.norm(dim=-1, keepdim=True) + 1e-8)
        y = torch.linspace(0, grid_h - 1, H, device=device)
        x = torch.linspace(0, grid_w - 1, W, device=device)
        gy, gx = torch.floor(y).long(), torch.floor(x).long()
        fy, fx = (y - gy).unsqueeze(1).expand(H, W), (x - gx).unsqueeze(0).expand(H, W)
        sy, sx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        gy0, gy1 = gy.clamp(0, grid_h - 1), (gy + 1).clamp(0, grid_h - 1)
        gx0, gx1 = gx.clamp(0, grid_w - 1), (gx + 1).clamp(0, grid_w - 1)
        g00, g10 = grads[gy0[:, None], gx0[None, :]], grads[gy0[:, None], gx1[None, :]]
        g01, g11 = grads[gy1[:, None], gx0[None, :]], grads[gy1[:, None], gx1[None, :]]
        d00, d10 = g00[..., 0] * fx + g00[..., 1] * fy, g10[..., 0] * (fx - 1) + g10[..., 1] * fy
        d01, d11 = g01[..., 0] * fx + g01[..., 1] * (fy - 1), g11[..., 0] * (fx - 1) + g11[..., 1] * (fy - 1)
        n0, n1 = d00 * (1 - sx) + d10 * sx, d01 * (1 - sx) + d11 * sx
        noise += (n0 * (1 - sy) + n1 * sy) * amplitude
        amplitude *= persistence
        frequency *= 2
    return noise

# =============================================================================
# MODEL
# =============================================================================

class LocalPerlinDeepMLP(nn.Module):
    """
    V46: Local Perlin Patches + Deep Trainable Readout.
    Architecture: 784 -> 2048 (FROZEN LOCAL PERLIN) -> 512 (TRAINABLE) -> 10 (TRAINABLE)
    """
    def __init__(self, input_size=784, projection_size=2048, hidden_size=512, output_size=10, device='cpu'):
        super().__init__()
        
        # 1. Frozen Local Perlin Layer
        self.frozen_layer = nn.Linear(input_size, projection_size)
        print(f"Generating {projection_size} Local Perlin Patches...")
        weights = torch.zeros(projection_size, input_size)
        
        for i in range(projection_size):
            p_h, p_w = random.randint(8, 16), random.randint(8, 16)
            top, left = random.randint(0, 28 - p_h), random.randint(0, 28 - p_w)
            
            # Perlin inside the patch
            s = random.uniform(0.2, 0.6)
            p_noise = generate_perlin_2d((p_h, p_w), scale=s, octaves=2, device='cpu')
            
            mask = torch.zeros(28, 28)
            mask[top:top+p_h, left:left+p_w] = p_noise
            weights[i] = mask.view(-1)
            
        # Kaiming-like normalization for the active parts
        target_std = math.sqrt(2.0 / input_size)
        current_std = weights[weights != 0].std()
        weights = weights * (target_std / (current_std + 1e-8))
        
        self.frozen_layer.weight.data = weights.to(device)
        nn.init.zeros_(self.frozen_layer.bias)
        self.frozen_layer.weight.requires_grad = False
        self.frozen_layer.bias.requires_grad = False
        
        # 2. Trainable Deep Readout
        self.hidden = nn.Linear(projection_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # Frozen Projection
        with torch.no_grad():
            x = self.frozen_layer(x)
            x = torch.relu(x)
        # Trainable layers
        x = self.hidden(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V46: LOCAL PERLIN + DEEP READOUT ---")
    print(f"Device: {device}")

    # Hyperparameters
    PROJ_SIZE = 2048
    HIDDEN_SIZE = 512
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

    model = LocalPerlinDeepMLP(projection_size=PROJ_SIZE, hidden_size=HIDDEN_SIZE, device=device).to(device)
    
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

    print(f"\n🚀 Final Accuracy: {acc:.4f} | Total Time: {time.time() - t_start:.1f}s")
    
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v46_local_perlin_deep.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
