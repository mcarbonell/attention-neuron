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

class ContinuousSplatLayer(nn.Module):
    """
    V29 Component: Fully Continuous Spatial Attention (The Splat).
    Extracts spatial features using learnable 2D Gaussian ovals.
    """
    def __init__(self, in_channels, out_channels, spatial_size, splats_per_channel=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_size = spatial_size
        self.K = splats_per_channel
        
        # Grid of coordinates [-1, 1]
        y = torch.linspace(-1, 1, spatial_size)
        x = torch.linspace(-1, 1, spatial_size)
        self.register_buffer('grid_y', y.view(spatial_size, 1).expand(spatial_size, spatial_size))
        self.register_buffer('grid_x', x.view(1, spatial_size).expand(spatial_size, spatial_size))
        
        # Splat Parameters: (out_channels, in_channels, K)
        # Centers [-0.8, 0.8] to avoid extreme edges initially
        self.cx = nn.Parameter(torch.rand(out_channels, in_channels, self.K) * 1.6 - 0.8)
        self.cy = nn.Parameter(torch.rand(out_channels, in_channels, self.K) * 1.6 - 0.8)
        
        # Log-spread (Initialize small: exp(-2) ~ 0.13)
        self.log_sig_x = nn.Parameter(torch.ones(out_channels, in_channels, self.K) * -2.0)
        self.log_sig_y = nn.Parameter(torch.ones(out_channels, in_channels, self.K) * -2.0)
        
        # Rotation (raw before tanh)
        self.rho_raw = nn.Parameter(torch.zeros(out_channels, in_channels, self.K))
        
        # Amplitude (Weight) - Kaiming-like init
        std = math.sqrt(2.0 / (in_channels * self.K))
        self.amp = nn.Parameter(torch.randn(out_channels, in_channels, self.K) * std)
        
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.spatial_size and W == self.spatial_size, f"Expected spatial size {self.spatial_size}, got {H}x{W}"
        
        # Constrain parameters
        sx = torch.exp(self.log_sig_x) + 1e-4
        sy = torch.exp(self.log_sig_y) + 1e-4
        r = torch.tanh(self.rho_raw) * 0.95
        
        # Reshape for broadcasting: (1, 1, 1, H, W)
        gx = self.grid_x.view(1, 1, 1, H, W)
        gy = self.grid_y.view(1, 1, 1, H, W)
        
        # Reshape params: (out_C, in_C, K, 1, 1)
        cx = self.cx.view(self.out_channels, self.in_channels, self.K, 1, 1)
        cy = self.cy.view(self.out_channels, self.in_channels, self.K, 1, 1)
        sx = sx.view(self.out_channels, self.in_channels, self.K, 1, 1)
        sy = sy.view(self.out_channels, self.in_channels, self.K, 1, 1)
        r = r.view(self.out_channels, self.in_channels, self.K, 1, 1)
        a = self.amp.view(self.out_channels, self.in_channels, self.K, 1, 1)
        
        # Gaussian formula
        dx = (gx - cx) / sx
        dy = (gy - cy) / sy
        z = (dx**2 - 2*r*dx*dy + dy**2) / (1 - r**2)
        gaussians = a * torch.exp(-0.5 * z) # (out_C, in_C, K, H, W)
        
        # Sum splats per filter: (out_C, in_C, H, W)
        filters = torch.sum(gaussians, dim=2)
        
        # Apply spatial attention (Global dot product per channel pair)
        # x: (B, in_C, H*W)
        # filters: (out_C, in_C, H*W)
        x_flat = x.view(B, self.in_channels, -1)
        f_flat = filters.view(self.out_channels, self.in_channels, -1)
        
        # y: (B, out_C) = sum_{in_C, HW} (x * filters)
        y = torch.einsum('bil,cil->bc', x_flat, f_flat) + self.bias
        
        # Output shape: (B, out_C, 1, 1) to simulate a 1x1 spatial map
        return y.view(B, self.out_channels, 1, 1)

class SplatterBlock(nn.Module):
    """
    Residual Block combining Global Continuous Spatial Attention (Splat) 
    and Channel Mixing (1x1 Conv).
    NO 3x3 Convolutions.
    """
    def __init__(self, in_channels, out_channels, spatial_size, splats=2):
        super().__init__()
        
        # 1. Spatial Phase: Extract global continuous features
        self.splat = ContinuousSplatLayer(in_channels, out_channels, spatial_size, splats_per_channel=splats)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 2. Channel Phase: Mix features (Plastic 1x1 Conv)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut (if channel dimension changes)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            # We use a simple 1x1 projection for the shortcut.
            # Note: the shortcut will collapse spatial dimensions to 1x1 using AvgPool first
            # because the main branch (Splat) outputs 1x1.
            self.shortcut = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential(
                nn.AdaptiveAvgPool2d(1)
            )

    def forward(self, x):
        # Splat phase collapses H, W -> 1, 1
        out = F.relu(self.bn1(self.splat(x)))
        # Mix phase operates on the 1x1 vectors
        out = self.bn2(self.conv1x1(out))
        # Add collapsed shortcut
        out += self.shortcut(x)
        return F.relu(out)

class SplatterResNet(nn.Module):
    """
    V29: The Fully Continuous Network.
    Zero 3x3 convolutions. Pure spatial Gaussian splatting + 1x1 mixing.
    """
    def __init__(self, splats_per_channel=2):
        super().__init__()
        # Initial projection: Map 3 channels to 32 to give the first splats enough room.
        # We keep spatial size 32x32.
        self.init_conv = nn.Conv2d(3, 32, kernel_size=1, bias=False)
        self.init_bn = nn.BatchNorm2d(32)
        
        # Splatter Blocks (Spatial size remains 32 for the input of each splat, 
        # but the output is 1x1. So subsequent blocks see 1x1 inputs!)
        # Wait, if block 1 outputs 1x1, block 2 cannot do spatial splatting on 1x1.
        # Architecture fix: We do spatial splatting ONLY when we have spatial data.
        # Once data is 1x1, it's just MLPs.
        
        # So we use Splatting to extract a rich set of 256 global features directly from 32x32.
        self.splat_layer = SplatterBlock(in_channels=32, out_channels=256, spatial_size=32, splats=splats_per_channel)
        
        # Now we have (B, 256, 1, 1). We build a deep ResNet using ONLY 1x1 convolutions (MLPs).
        self.mlp_block1 = self._make_mlp_block(256, 256)
        self.mlp_block2 = self._make_mlp_block(256, 256)
        self.mlp_block3 = self._make_mlp_block(256, 256)
        
        self.classifier = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def _make_mlp_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        # Prepare channels
        x = self.relu(self.init_bn(self.init_conv(x)))
        
        # EXTRACT GLOBAL SPATIAL FEATURES (The Ovals)
        # Input: 32x32. Output: 1x1
        x = self.splat_layer(x)
        
        # DEEP PROCESSING (The Brain)
        res = x
        x = self.relu(self.mlp_block1(x) + res)
        res = x
        x = self.relu(self.mlp_block2(x) + res)
        res = x
        x = self.relu(self.mlp_block3(x) + res)
        
        # Classify
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V29 'THE SPLATTER-RESNET' (Fully Continuous Vision) on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 50
    SPLATS = 4 # 4 ovals per 256 features = 1024 ovals searching the image!
    MAX_LR = 0.005 # High LR to let the ovals move fast
    
    # Data Augmentation: We avoid RandomCrop because ovals learn absolute spatial semantics
    # e.g., "sky is usually at the top". If we crop, the top moves.
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
    
    model = SplatterResNet(splats_per_channel=SPLATS).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR / 10, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=len(train_loader)*EPOCHS, 
        pct_start=0.3, anneal_strategy='cos'
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
            torch.save(model.state_dict(), "results/v29_splatter_resnet_best.pt")
            
        t_now = time.time()
        elapsed = t_now - t_start
        epoch_time = t_now - t_epoch_start
        eta = (elapsed / epoch) * (EPOCHS - epoch)
        
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] Epoch {epoch:2d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f} | Time: {epoch_time:.1f}s | ETA: {eta/60:.1f}m")

    t_total = time.time() - t_start
    print(f"\nSPLATTER RESNET finished in {t_total:.1f}s")
    print(f"Final Best Test Accuracy: {best_acc:.4f}")
    
    results = {
        "model": "V29_Splatter_ResNet",
        "splats_per_channel": SPLATS,
        "trainable_params": params,
        "best_acc": best_acc,
        "epochs": EPOCHS,
        "wall_clock_time": t_total,
        "dataset": "CIFAR-10"
    }
    
    with open("results/raw/v29_splatter_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
