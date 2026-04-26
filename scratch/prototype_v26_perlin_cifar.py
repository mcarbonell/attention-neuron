import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import json
import os

# =============================================================================
# PERLIN NOISE GENERATOR (PyTorch, no external deps)
# =============================================================================

def generate_perlin_2d(shape, scale=1.0, octaves=1, persistence=0.5, device='cpu'):
    """
    Generate 2D Perlin-like noise using smooth interpolation of random gradients.
    shape: (H, W)
    scale: frequency of the noise (higher = more rapid variation)
    octaves: number of noise layers to sum (fractal Brownian motion)
    Returns: tensor of shape (H, W)
    """
    H, W = shape
    noise = torch.zeros(H, W, device=device)
    amplitude = 1.0
    frequency = scale
    
    for _ in range(octaves):
        # Grid size: higher frequency = smaller grid cells
        grid_h = max(2, int(H * frequency))
        grid_w = max(2, int(W * frequency))
        
        # Random gradients at grid nodes
        grads = torch.randn(grid_h, grid_w, 2, device=device)
        grads = grads / (grads.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Coordinates in the grid
        y = torch.linspace(0, grid_h - 1, H, device=device)
        x = torch.linspace(0, grid_w - 1, W, device=device)
        gy = torch.floor(y).long()
        gx = torch.floor(x).long()
        
        # Fractional parts
        fy = (y - gy).unsqueeze(1)  # (H, 1)
        fx = (x - gx).unsqueeze(0)  # (1, W)
        
        # Smoothstep interpolation
        sy = fy * fy * (3 - 2 * fy)  # (H, 1)
        sx = fx * fx * (3 - 2 * fx)  # (1, W)
        
        # Clamp grid indices
        gy0 = gy.clamp(0, grid_h - 1)
        gy1 = (gy0 + 1).clamp(0, grid_h - 1)
        gx0 = gx.clamp(0, grid_w - 1)
        gx1 = (gx0 + 1).clamp(0, grid_w - 1)
        
        # Distance vectors from grid nodes
        # For each pixel, compute dot product with the 4 surrounding gradients
        # We'll do this vectorized by expanding
        y_idx = torch.arange(H, device=device).unsqueeze(1).expand(H, W)
        x_idx = torch.arange(W, device=device).unsqueeze(0).expand(H, W)
        
        # Normalized local coordinates within each cell
        ly = fy.expand(H, W)  # (H, W)
        lx = fx.expand(H, W)  # (H, W)
        
        # Dot products at corners
        # Corner (0,0): (gx0, gy0)
        g00 = grads[gy0[:, None].expand(H, W), gx0[None, :].expand(H, W)]  # (H, W, 2)
        d00 = g00[..., 0] * lx + g00[..., 1] * ly
        
        # Corner (1,0): (gx1, gy0)
        g10 = grads[gy0[:, None].expand(H, W), gx1[None, :].expand(H, W)]
        d10 = g10[..., 0] * (lx - 1) + g10[..., 1] * ly
        
        # Corner (0,1): (gx0, gy1)
        g01 = grads[gy1[:, None].expand(H, W), gx0[None, :].expand(H, W)]
        d01 = g01[..., 0] * lx + g01[..., 1] * (ly - 1)
        
        # Corner (1,1): (gx1, gy1)
        g11 = grads[gy1[:, None].expand(H, W), gx1[None, :].expand(H, W)]
        d11 = g11[..., 0] * (lx - 1) + g11[..., 1] * (ly - 1)
        
        # Bilinear interpolation with smoothstep
        sx_exp = sx.expand(H, W)
        sy_exp = sy.expand(H, W)
        
        n0 = d00 * (1 - sx_exp) + d10 * sx_exp
        n1 = d01 * (1 - sx_exp) + d11 * sx_exp
        layer = n0 * (1 - sy_exp) + n1 * sy_exp
        
        noise += layer * amplitude
        amplitude *= persistence
        frequency *= 2
    
    return noise


def create_perlin_kernels(out_channels, in_channels, kernel_size=3, scale=1.0, octaves=2, device='cpu'):
    """
    Create a conv kernel (out, in, k, k) where each spatial slice is Perlin noise.
    We generate independent Perlin realizations per (out, in) pair with slight offsets.
    """
    kernels = torch.zeros(out_channels, in_channels, kernel_size, kernel_size, device=device)
    for o in range(out_channels):
        for i in range(in_channels):
            # Offset the noise seed implicitly by using different scales per pair
            s = scale * (1.0 + 0.1 * math.sin(o * 3.7 + i * 1.3))
            kernels[o, i] = generate_perlin_2d((kernel_size, kernel_size), scale=s, octaves=octaves, device=device)
    
    # Normalize to Kaiming std
    std = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
    kernels = kernels / (kernels.std() + 1e-8) * std
    return kernels


# =============================================================================
# V26: PERLIN SPECTRUM NET
# =============================================================================

class PerlinSpectrumLayer(nn.Module):
    """
    V26 Component: Kernels initialized with Perlin noise at different spatial scales.
    Instead of white noise, the substrates have spatially correlated structure.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, rank=16, num_substrates=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.rank = rank
        self.num_substrates = num_substrates
        
        # 4 Perlin substrates at different scales
        # Scale 1: low freq (large structures / smooth gradients)
        # Scale 2: medium-low freq
        # Scale 3: medium-high freq
        # Scale 4: high freq (fine textures)
        scales = [0.3, 0.6, 1.2, 2.4]
        for k in range(num_substrates):
            w = create_perlin_kernels(
                out_channels, in_channels, kernel_size,
                scale=scales[k], octaves=2, device='cpu'
            )
            self.register_buffer(f'w_init_{k}', w)
        
        # Channel-wise Library Attention
        self.library_logits = nn.Parameter(torch.zeros(out_channels, num_substrates))
        
        # Dual Channel Modulation (Rank-r)
        self.delta_in_m = nn.Parameter(torch.randn(out_channels, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_channels) * 0.01)
        self.delta_in_a = nn.Parameter(torch.zeros(out_channels, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_channels))
        
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 1. Mix the substrates per output channel
        mix_weights = torch.softmax(self.library_logits, dim=1)
        mix_weights = mix_weights.view(self.out_channels, self.num_substrates, 1, 1, 1)
        
        w_mixed = 0
        for k in range(self.num_substrates):
            w_init_k = getattr(self, f'w_init_{k}')
            w_mixed += mix_weights[:, k] * w_init_k
            
        # 2. Apply channel-wise modulation
        m_chan = torch.matmul(self.delta_in_m, self.delta_out_m).view(self.out_channels, self.in_channels, 1, 1)
        a_chan = torch.matmul(self.delta_in_a, self.delta_out_a).view(self.out_channels, self.in_channels, 1, 1)
        
        w_evolved = w_mixed * (1.0 + m_chan) + a_chan
        
        y = F.conv2d(x, w_evolved, padding=self.padding) + self.bias.view(1, -1, 1, 1)
        return self.bn(y)


class AttentionLinearLayer(nn.Module):
    """Simple Attention Layer for the classifier"""
    def __init__(self, in_features, out_features, rank=16):
        super().__init__()
        std = math.sqrt(2.0 / in_features)
        self.register_buffer('w_init', torch.randn(out_features, in_features) * std)
        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.delta_in_a = nn.Parameter(torch.zeros(out_features, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        w_evolved = self.w_init * (1.0 + w_m) + w_a
        return self.bn(torch.matmul(x, w_evolved.t()) + self.bias)


class PerlinSpectrumNet(nn.Module):
    def __init__(self, rank=16, num_substrates=4):
        super().__init__()
        # Block 1
        self.conv1 = PerlinSpectrumLayer(3, 64, kernel_size=5, padding=2, rank=rank, num_substrates=num_substrates)
        self.conv2 = PerlinSpectrumLayer(64, 64, rank=rank, num_substrates=num_substrates)
        self.pool1 = nn.MaxPool2d(2)
        
        # Block 2
        self.conv3 = PerlinSpectrumLayer(64, 128, rank=rank, num_substrates=num_substrates)
        self.conv4 = PerlinSpectrumLayer(128, 128, rank=rank, num_substrates=num_substrates)
        self.pool2 = nn.MaxPool2d(2)
        
        # Block 3
        self.conv5 = PerlinSpectrumLayer(128, 256, rank=rank, num_substrates=num_substrates)
        self.conv6 = PerlinSpectrumLayer(256, 256, rank=rank, num_substrates=num_substrates)
        self.pool3 = nn.MaxPool2d(2)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = AttentionLinearLayer(256, 10, rank=rank)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.dropout(x)
        
        x = self.gap(x).view(-1, 256)
        x = self.fc(x)
        return x


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V26 'PERLIN SPECTRUM' (CNN with Perlin substrates) on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 50
    RANK = 16
    NUM_SUBSTRATES = 4
    MAX_LR = 0.003
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2)
    
    os.makedirs("results", exist_ok=True)
    
    model = PerlinSpectrumNet(rank=RANK, num_substrates=NUM_SUBSTRATES).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR / 10, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=len(train_loader)*EPOCHS, 
        pct_start=0.2, anneal_strategy='cos'
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0
    t_start = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        t_epoch_start = time.time()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = correct / 10000
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "results/v26_perlin_best.pt")
            
        t_now = time.time()
        elapsed = t_now - t_start
        epoch_time = t_now - t_epoch_start
        eta = (elapsed / epoch) * (EPOCHS - epoch)
        
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] Epoch {epoch:2d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f} | Time: {epoch_time:.1f}s | ETA: {eta/60:.1f}m")

    # Library Usage Analysis
    print("\n--- Perlin Analysis: Library Usage per Layer ---")
    for i, name in enumerate(["conv1", "conv2", "conv3", "conv4", "conv5", "conv6"]):
        layer = getattr(model, name)
        weights = torch.softmax(layer.library_logits, dim=1).detach().cpu()
        mean_usage = weights.mean(dim=0).numpy()
        print(f"Layer {name} Mean Usage: {mean_usage}")

    t_total = time.time() - t_start
    print(f"\nPERLIN SPECTRUM finished in {t_total:.1f}s")
    print(f"Final Best Test Accuracy: {best_acc:.4f}")
    
    results = {
        "model": "V26_Perlin_Spectrum",
        "rank": RANK,
        "num_substrates": NUM_SUBSTRATES,
        "trainable_params": params,
        "best_acc": best_acc,
        "epochs": EPOCHS,
        "wall_clock_time": t_total,
        "dataset": "CIFAR-10"
    }
    
    with open("results/raw/v26_perlin_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
