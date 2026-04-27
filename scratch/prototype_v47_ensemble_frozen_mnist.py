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
# GENERATORS
# =============================================================================

def generate_perlin_2d(shape, scale=1.0, octaves=1, device='cpu'):
    H, W = shape
    noise = torch.zeros(H, W, device=device)
    amplitude, frequency = 1.0, scale
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
        amplitude *= 0.5
        frequency *= 2
    return noise

def get_fourier_kernel(size, u, v, phase, device='cpu'):
    d0 = torch.linspace(0, 1, size, device=device)
    y, x = torch.meshgrid(d0, d0, indexing='ij')
    return torch.cos(2 * math.pi * (u * x + v * y) + phase)

# =============================================================================
# MODEL
# =============================================================================

class EnsembleFrozenMLP(nn.Module):
    """
    V47: Ensemble Frozen Projection + Deep Readout.
    First layer (2048 neurons) is a mix of:
    - 512 Local Patches
    - 512 Local Perlin
    - 512 Global Random
    - 512 Fourier Basis
    """
    def __init__(self, input_size=784, projection_size=2048, hidden_size=512, output_size=10, device='cpu'):
        super().__init__()
        self.frozen_layer = nn.Linear(input_size, projection_size)
        print(f"Generating Ensemble Projection ({projection_size} neurons)...")
        
        weights = torch.zeros(projection_size, input_size)
        chunk = projection_size // 4
        
        # 1. Local Patches (V44)
        for i in range(0, chunk):
            p_h, p_w = random.randint(8, 16), random.randint(8, 16)
            top, left = random.randint(0, 28 - p_h), random.randint(0, 28 - p_w)
            mask = torch.zeros(28, 28)
            mask[top:top+p_h, left:left+p_w] = 1.0
            weights[i] = torch.randn(input_size) * mask.view(-1)
            
        # 2. Local Perlin (V46)
        for i in range(chunk, 2 * chunk):
            p_h, p_w = random.randint(8, 16), random.randint(8, 16)
            top, left = random.randint(0, 28 - p_h), random.randint(0, 28 - p_w)
            s = random.uniform(0.2, 0.6)
            p_noise = generate_perlin_2d((p_h, p_w), scale=s, device='cpu')
            mask = torch.zeros(28, 28)
            mask[top:top+p_h, left:left+p_w] = p_noise
            weights[i] = mask.view(-1)
            
        # 3. Global Random (V41)
        for i in range(2 * chunk, 3 * chunk):
            weights[i] = torch.randn(input_size)
            
        # 4. Fourier Basis (V45)
        for i in range(3 * chunk, projection_size):
            u, v = random.uniform(-5, 5), random.uniform(-5, 5)
            phase = random.uniform(0, 2*math.pi)
            weights[i] = get_fourier_kernel(28, u, v, phase).view(-1)

        # Final normalization
        target_std = math.sqrt(2.0 / input_size)
        current_std = weights[weights != 0].std()
        weights = weights * (target_std / (current_std + 1e-8))
        
        self.frozen_layer.weight.data = weights.to(device)
        nn.init.zeros_(self.frozen_layer.bias)
        self.frozen_layer.weight.requires_grad = False
        self.frozen_layer.bias.requires_grad = False
        
        # Deep Trainable layers
        self.hidden = nn.Linear(projection_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        with torch.no_grad():
            x = torch.relu(self.frozen_layer(x))
        x = torch.relu(self.hidden(x))
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V47: THE ENSEMBLE PROJECTION ---")
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

    model = EnsembleFrozenMLP(device=device).to(device)
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
    with open("results/raw/v47_ensemble_frozen.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
