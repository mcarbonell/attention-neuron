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

# --- Noise Generators (Spectrum) ---
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
        fy = (y - gy).unsqueeze(1)
        fx = (x - gx).unsqueeze(0)
        sy = fy * fy * (3 - 2 * fy)
        sx = fx * fx * (3 - 2 * fx)
        gy0 = gy.clamp(0, grid_h - 1)
        gy1 = (gy0 + 1).clamp(0, grid_h - 1)
        gx0 = gx.clamp(0, grid_w - 1)
        gx1 = (gx0 + 1).clamp(0, grid_w - 1)
        ly, lx = fy.expand(H, W), fx.expand(H, W)
        g00 = grads[gy0[:, None].expand(H, W), gx0[None, :].expand(H, W)]
        d00 = g00[..., 0] * lx + g00[..., 1] * ly
        g10 = grads[gy0[:, None].expand(H, W), gx1[None, :].expand(H, W)]
        d10 = g10[..., 0] * (lx - 1) + g10[..., 1] * ly
        g01 = grads[gy1[:, None].expand(H, W), gx0[None, :].expand(H, W)]
        d01 = g01[..., 0] * lx + g01[..., 1] * (ly - 1)
        g11 = grads[gy1[:, None].expand(H, W), gx1[None, :].expand(H, W)]
        d11 = g11[..., 0] * (lx - 1) + g11[..., 1] * (ly - 1)
        sx_exp, sy_exp = sx.expand(H, W), sy.expand(H, W)
        n0 = d00 * (1 - sx_exp) + d10 * sx_exp
        n1 = d01 * (1 - sx_exp) + d11 * sx_exp
        layer = n0 * (1 - sy_exp) + n1 * sy_exp
        noise += layer * amplitude
        amplitude *= persistence
        frequency *= 2
    return noise

def create_spectrum_bases(out_channels, in_channels, spatial_size=32, device='cpu'):
    """
    Creates 4 base substrates for the entire spatial area (32x32).
    0: White Noise, 1: Perlin Low, 2: Perlin High, 3: Blue Noise (checkerboard)
    """
    bases = []
    std = math.sqrt(2.0 / (in_channels * 9)) # Scaled roughly for typical local receptive fields
    
    # 0. White Noise
    bases.append(torch.randn(out_channels, in_channels, spatial_size, spatial_size, device=device) * std)
    
    # 1. Perlin Low Freq
    b_pl = torch.zeros(out_channels, in_channels, spatial_size, spatial_size, device=device)
    for o in range(out_channels):
        for i in range(in_channels):
            b_pl[o, i] = generate_perlin_2d((spatial_size, spatial_size), scale=2.0, device=device)
    bases.append(b_pl / (b_pl.std() + 1e-8) * std)
    
    # 2. Perlin High Freq
    b_ph = torch.zeros(out_channels, in_channels, spatial_size, spatial_size, device=device)
    for o in range(out_channels):
        for i in range(in_channels):
            b_ph[o, i] = generate_perlin_2d((spatial_size, spatial_size), scale=8.0, device=device)
    bases.append(b_ph / (b_ph.std() + 1e-8) * std)
    
    # 3. Blue Noise / Checkerboard
    x = torch.arange(spatial_size).view(1, -1).expand(spatial_size, spatial_size).float()
    y = torch.arange(spatial_size).view(-1, 1).expand(spatial_size, spatial_size).float()
    checker = torch.sin(math.pi * x) * torch.sin(math.pi * y)
    checker = checker.view(1, 1, spatial_size, spatial_size).expand(out_channels, in_channels, spatial_size, spatial_size)
    b_blue = (checker + torch.randn_like(checker) * 0.5)
    bases.append(b_blue / (b_blue.std() + 1e-8) * std)
    
    return bases

class WindowedSpectrumLayer(nn.Module):
    """
    V33 Component: Applies a Differentiable Soft Window to EACH of the 4 Spectrum Substrates.
    Composes a custom spatial filter per channel pair by cropping and summing noise.
    """
    def __init__(self, in_channels, out_channels, spatial_size=32, temperature=10.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_size = spatial_size
        self.temp = temperature
        self.num_substrates = 4
        
        # Load the 4 full-image spectrums
        k_list = create_spectrum_bases(out_channels, in_channels, spatial_size)
        for i, k_tensor in enumerate(k_list):
            self.register_buffer(f'base_{i}', k_tensor)
            
        # 1D Grids for Windows
        x = torch.linspace(-1, 1, spatial_size)
        y = torch.linspace(-1, 1, spatial_size)
        self.register_buffer('grid_x', x)
        self.register_buffer('grid_y', y)
        
        # Window Parameters per substrate: (out, in, num_substrates)
        self.x_min = nn.Parameter(torch.rand(out_channels, in_channels, self.num_substrates) * 0.8 - 0.8)
        self.x_max = nn.Parameter(torch.rand(out_channels, in_channels, self.num_substrates) * 0.8)
        self.y_min = nn.Parameter(torch.rand(out_channels, in_channels, self.num_substrates) * 0.8 - 0.8)
        self.y_max = nn.Parameter(torch.rand(out_channels, in_channels, self.num_substrates) * 0.8)
        
        # Amplitude (Mixing weight per windowed substrate)
        self.amp = nn.Parameter(torch.randn(out_channels, in_channels, self.num_substrates) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def _generate_composed_filters(self):
        # Softplus ensures x_max > x_min
        x_min = self.x_min.unsqueeze(-1)
        x_max = (self.x_min + F.softplus(self.x_max - self.x_min) + 1e-3).unsqueeze(-1)
        y_min = self.y_min.unsqueeze(-1)
        y_max = (self.y_min + F.softplus(self.y_max - self.y_min) + 1e-3).unsqueeze(-1)
        
        gx = self.grid_x.view(1, 1, 1, self.spatial_size)
        gy = self.grid_y.view(1, 1, 1, self.spatial_size)
        
        # 1D Sigmoid Masks
        mask_x = torch.sigmoid(self.temp * (gx - x_min)) - torch.sigmoid(self.temp * (gx - x_max))
        mask_y = torch.sigmoid(self.temp * (gy - y_min)) - torch.sigmoid(self.temp * (gy - y_max))
        
        # 2D Windows: (out, in, substrates, H, W)
        windows_2d = mask_y.unsqueeze(-1) * mask_x.unsqueeze(-2)
        amp = self.amp.view(self.out_channels, self.in_channels, self.num_substrates, 1, 1)
        
        # Apply windows and amplitudes to the 4 frozen substrates
        composed_filters = 0
        for k in range(self.num_substrates):
            base_k = getattr(self, f'base_{k}') # (out, in, H, W)
            # Add the windowed chunk of this specific noise to the final filter
            composed_filters += amp[:, :, k, :, :] * windows_2d[:, :, k, :, :] * base_k.unsqueeze(2)
            
        # Sum out the substrate dimension (already done implicitly above if we squeeze)
        # Actually, composed_filters shape is (out, in, 1, H, W). We squeeze it.
        return composed_filters.squeeze(2)

    def forward(self, x):
        B, C, H, W = x.shape
        filters = self._generate_composed_filters() # (out_C, in_C, H, W)
        
        x_flat = x.view(B, self.in_channels, -1)
        f_flat = filters.view(self.out_channels, self.in_channels, -1)
        
        y = torch.einsum('bil,cil->bc', x_flat, f_flat) + self.bias
        return y.view(B, self.out_channels, 1, 1)

class WindowedSpectrumNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Extractor (The Composing Eye)
        self.eye = WindowedSpectrumLayer(in_channels=3, out_channels=256, spatial_size=32)
        self.bn1 = nn.BatchNorm1d(256)
        
        # Brain (MLP)
        self.fc1 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.eye(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc1(x)))
        return self.fc2(x)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V33 'THE WINDOWED SPECTRUM' (Soft Windows over Mixed Noise) on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 50
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
    
    model = WindowedSpectrumNet().to(device)
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
            torch.save(model.state_dict(), "results/v33_windowed_spectrum_best.pt")
            
        t_now = time.time()
        elapsed = t_now - t_start
        eta = (elapsed / epoch) * (EPOCHS - epoch)
        print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch:2d}/{EPOCHS} | Acc: {acc:.4f} | Best: {best_acc:.4f} | Time: {t_now-t0:.1f}s | ETA: {eta/60:.1f}m")

if __name__ == "__main__":
    main()
