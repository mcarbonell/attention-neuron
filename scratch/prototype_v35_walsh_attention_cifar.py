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

# --- Fast Walsh-Hadamard Transform (Correct Vectorized Version) ---

def fwht(x):
    """
    Computes the Fast Walsh-Hadamard Transform of a batch of vectors.
    Input x: (B, C, N) where N must be a power of 2.
    """
    B, C, N = x.shape
    h = 1
    while h < N:
        x = x.view(B, C, N // (2 * h), 2, h)
        a = x[:, :, :, 0, :]
        b = x[:, :, :, 1, :]
        # Vectorized butterfly
        x = torch.stack([a + b, a - b], dim=3)
        h *= 2
    return x.view(B, C, N)

def ifwht(x):
    """
    Inverse Fast Walsh-Hadamard Transform.
    """
    N = x.shape[-1]
    return fwht(x) / N

# --- Walsh Attention Layer ---

class WalshFilterLayer(nn.Module):
    def __init__(self, channels, spatial_size=32):
        super().__init__()
        self.channels = channels
        self.N = spatial_size * spatial_size
        
        # Initializing deltas small to avoid exploding Walsh coeffs
        self.delta_m = nn.Parameter(torch.randn(channels, self.N) * 0.01)
        self.delta_a = nn.Parameter(torch.zeros(channels, self.N))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W)
        
        # Dominio Walsh
        x_walsh = fwht(x_flat)
        # Modulación
        x_filtered = x_walsh * (1.0 + self.delta_m) + self.delta_a
        # Vuelta al espacio
        x_spatial = ifwht(x_filtered)
        
        out = x_spatial.view(B, C, H, W)
        return self.bn(out)

# --- Hybrid Walsh-CNN Architecture ---

class WalshResBlock(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.walsh = WalshFilterLayer(planes, spatial_size=32)
        self.conv1x1 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        res = x
        out = F.relu(self.walsh(x))
        out = self.bn(self.conv1x1(out))
        return F.relu(out + res)

class WalshNet(nn.Module):
    def __init__(self, num_blocks=3, channels=64):
        super().__init__()
        # Initial projection (the only spatial conv to map RGB to rich features)
        self.init_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.blocks = nn.ModuleList([
            WalshResBlock(channels) for _ in range(num_blocks)
        ])
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.init_conv(x)))
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def main():
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V35 'THE WALSH FILTER' (FWHT) on: {device}")
    
    BATCH_SIZE = 64 # Reduced for better CPU stability
    EPOCHS = 50
    LR = 0.003
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_loader = DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transform_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(datasets.CIFAR10('./data', train=False, transform=transform_test), batch_size=512, num_workers=2)
    
    model = WalshNet(num_blocks=3, channels=64).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR/10, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=len(train_loader)*EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # FAST FEEDBACK: Log every 20 batches in the first epoch
            if epoch == 1 and batch_idx % 20 == 0:
                print(f"  > Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = correct / 10000
        if acc > best_acc: best_acc = acc
        
        t_now = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch:2d}/{EPOCHS} | Acc: {acc:.4f} | Best: {best_acc:.4f} | Time: {t_now-t0:.1f}s")

if __name__ == "__main__":
    main()
