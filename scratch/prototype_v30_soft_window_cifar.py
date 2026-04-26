import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os

class SoftWindowLayer(nn.Module):
    """
    V30 Component: The Framer (Soft Window Attention).
    Uses 1D separable sigmoid masks to create a differentiable bounding box 
    with O(H+W) complexity instead of O(H*W).
    """
    def __init__(self, in_channels, out_channels, spatial_size=32, windows_per_channel=4, temperature=10.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_size = spatial_size
        self.K = windows_per_channel
        self.temp = temperature
        
        # 1D Grids
        # Shape: (spatial_size)
        x = torch.linspace(-1, 1, spatial_size)
        y = torch.linspace(-1, 1, spatial_size)
        self.register_buffer('grid_x', x)
        self.register_buffer('grid_y', y)
        
        # Window Parameters: (out_channels, in_channels, K)
        # We initialize min bounds randomly in [-0.8, 0.0] and max bounds in [0.0, 0.8]
        self.x_min = nn.Parameter(torch.rand(out_channels, in_channels, self.K) * 0.8 - 0.8)
        self.x_max = nn.Parameter(torch.rand(out_channels, in_channels, self.K) * 0.8)
        self.y_min = nn.Parameter(torch.rand(out_channels, in_channels, self.K) * 0.8 - 0.8)
        self.y_max = nn.Parameter(torch.rand(out_channels, in_channels, self.K) * 0.8)
        
        # Amplitude
        self.amp = nn.Parameter(torch.randn(out_channels, in_channels, self.K) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def _generate_windows(self):
        # Enforce valid windows (min < max) using softplus
        x_min = self.x_min
        x_max = x_min + F.softplus(self.x_max - x_min) + 1e-3
        y_min = self.y_min
        y_max = y_min + F.softplus(self.y_max - y_min) + 1e-3
        
        # Reshape for broadcasting with 1D grids
        # Params: (out, in, K, 1)
        x_min = x_min.unsqueeze(-1)
        x_max = x_max.unsqueeze(-1)
        y_min = y_min.unsqueeze(-1)
        y_max = y_max.unsqueeze(-1)
        
        # Grids: (1, 1, 1, spatial_size)
        gx = self.grid_x.view(1, 1, 1, self.spatial_size)
        gy = self.grid_y.view(1, 1, 1, self.spatial_size)
        
        # Calculate 1D masks using Sigmoid
        # M_x = sigmoid(T * (x - x_min)) - sigmoid(T * (x - x_max))
        mask_x = torch.sigmoid(self.temp * (gx - x_min)) - torch.sigmoid(self.temp * (gx - x_max))
        mask_y = torch.sigmoid(self.temp * (gy - y_min)) - torch.sigmoid(self.temp * (gy - y_max))
        
        # Outer product to get 2D window: (out, in, K, H, W)
        # mask_y is (out, in, K, H, 1) and mask_x is (out, in, K, 1, W)
        windows_2d = mask_y.unsqueeze(-1) * mask_x.unsqueeze(-2)
        
        # Apply amplitude and sum splats per channel pair
        # amp: (out, in, K, 1, 1)
        amp = self.amp.view(self.out_channels, self.in_channels, self.K, 1, 1)
        
        # Sum over K: (out, in, H, W)
        filters = torch.sum(amp * windows_2d, dim=2)
        return filters

    def forward(self, x):
        B, C, H, W = x.shape
        # Generate spatial filters dynamically
        filters = self._generate_windows() # (out_C, in_C, H, W)
        
        x_flat = x.view(B, self.in_channels, -1)
        f_flat = filters.view(self.out_channels, self.in_channels, -1)
        
        y = torch.einsum('bil,cil->bc', x_flat, f_flat) + self.bias
        return y.view(B, self.out_channels, 1, 1)

class SoftWindowNet(nn.Module):
    def __init__(self, windows=4):
        super().__init__()
        # Extractor (The Framer)
        self.window_layer = SoftWindowLayer(in_channels=3, out_channels=256, spatial_size=32, windows_per_channel=windows)
        self.bn1 = nn.BatchNorm1d(256)
        
        # Brain (MLP)
        self.fc1 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.window_layer(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc1(x)))
        return self.fc2(x)

def main():
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V30 'THE FRAMER' (Soft Window Attention) on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 50
    WINDOWS = 4
    MAX_LR = 0.01
    
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
    test_loader = DataLoader(datasets.CIFAR10('./data', train=False, transform=transform_test), batch_size=1024, num_workers=2)
    
    os.makedirs("results", exist_ok=True)
    
    model = SoftWindowNet(windows=WINDOWS).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR / 10, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, total_steps=len(train_loader)*EPOCHS, pct_start=0.2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = correct / 10000
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "results/v30_framer_best.pt")
            
        t_now = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch:2d}/{EPOCHS} | Acc: {acc:.4f} | Best: {best_acc:.4f} | Time: {t_now-t0:.1f}s | ETA: {(t_now-t_start)/epoch*(EPOCHS-epoch)/60:.1f}m")

if __name__ == "__main__":
    main()
