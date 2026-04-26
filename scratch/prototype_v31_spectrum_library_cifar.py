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

# --- Noise Generators ---
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
        
        ly = fy.expand(H, W)
        lx = fx.expand(H, W)
        
        g00 = grads[gy0[:, None].expand(H, W), gx0[None, :].expand(H, W)]
        d00 = g00[..., 0] * lx + g00[..., 1] * ly
        g10 = grads[gy0[:, None].expand(H, W), gx1[None, :].expand(H, W)]
        d10 = g10[..., 0] * (lx - 1) + g10[..., 1] * ly
        g01 = grads[gy1[:, None].expand(H, W), gx0[None, :].expand(H, W)]
        d01 = g01[..., 0] * lx + g01[..., 1] * (ly - 1)
        g11 = grads[gy1[:, None].expand(H, W), gx1[None, :].expand(H, W)]
        d11 = g11[..., 0] * (lx - 1) + g11[..., 1] * (ly - 1)
        
        sx_exp = sx.expand(H, W)
        sy_exp = sy.expand(H, W)
        n0 = d00 * (1 - sx_exp) + d10 * sx_exp
        n1 = d01 * (1 - sx_exp) + d11 * sx_exp
        layer = n0 * (1 - sy_exp) + n1 * sy_exp
        
        noise += layer * amplitude
        amplitude *= persistence
        frequency *= 2
    return noise

def create_spectrum_kernels(out_channels, in_channels, kernel_size=3, device='cpu'):
    """
    Creates 4 substrates with different noise spectrums:
    0: White Noise (High Frequency random)
    1: Perlin Scale 0.5 (Low Frequency smooth)
    2: Perlin Scale 1.5 (Medium Frequency)
    3: Blue Noise approximation (Edge detectors / Alternating signs)
    """
    kernels = []
    std = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
    
    # 0. White Noise
    k_white = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=device) * std
    kernels.append(k_white)
    
    # 1. Perlin Low Freq
    k_perlin_low = torch.zeros(out_channels, in_channels, kernel_size, kernel_size, device=device)
    for o in range(out_channels):
        for i in range(in_channels):
            s = 0.5 * (1.0 + 0.1 * math.sin(o + i))
            k_perlin_low[o, i] = generate_perlin_2d((kernel_size, kernel_size), scale=s, device=device)
    k_perlin_low = k_perlin_low / (k_perlin_low.std() + 1e-8) * std
    kernels.append(k_perlin_low)
    
    # 2. Perlin Med Freq
    k_perlin_med = torch.zeros(out_channels, in_channels, kernel_size, kernel_size, device=device)
    for o in range(out_channels):
        for i in range(in_channels):
            s = 1.5 * (1.0 + 0.1 * math.cos(o - i))
            k_perlin_med[o, i] = generate_perlin_2d((kernel_size, kernel_size), scale=s, device=device)
    k_perlin_med = k_perlin_med / (k_perlin_med.std() + 1e-8) * std
    kernels.append(k_perlin_med)
    
    # 3. Blue Noise approximation (Checkerboard pattern base + noise)
    x = torch.arange(kernel_size).view(1, -1).expand(kernel_size, kernel_size).float()
    y = torch.arange(kernel_size).view(-1, 1).expand(kernel_size, kernel_size).float()
    checker = torch.sin(math.pi * x) * torch.sin(math.pi * y)
    checker = checker.view(1, 1, kernel_size, kernel_size).expand(out_channels, in_channels, kernel_size, kernel_size)
    k_blue = (checker + torch.randn_like(checker) * 0.5)
    k_blue = k_blue / (k_blue.std() + 1e-8) * std
    kernels.append(k_blue)
    
    return kernels

# --- Model ---
class SpectrumConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, rank=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.num_substrates = 4
        
        # Load the 4 spectrums
        k_list = create_spectrum_kernels(out_channels, in_channels, kernel_size)
        for i, k_tensor in enumerate(k_list):
            self.register_buffer(f'w_init_{i}', k_tensor)
            
        self.library_logits = nn.Parameter(torch.zeros(out_channels, self.num_substrates))
        
        self.delta_in_m = nn.Parameter(torch.randn(out_channels, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_channels) * 0.01)
        self.delta_in_a = nn.Parameter(torch.zeros(out_channels, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_channels))
        
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        mix = torch.softmax(self.library_logits, dim=1).view(self.out_channels, self.num_substrates, 1, 1, 1)
        w_mixed = 0
        for i in range(self.num_substrates):
            w_mixed += mix[:, i] * getattr(self, f'w_init_{i}')
            
        m_chan = torch.matmul(self.delta_in_m, self.delta_out_m).view(self.out_channels, self.in_channels, 1, 1)
        a_chan = torch.matmul(self.delta_in_a, self.delta_out_a).view(self.out_channels, self.in_channels, 1, 1)
        
        w_evolved = w_mixed * (1.0 + m_chan) + a_chan
        return self.bn(F.conv2d(x, w_evolved, stride=self.stride, padding=self.padding) + self.bias.view(1, -1, 1, 1))

class SpectrumBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, rank=16):
        super().__init__()
        self.conv1 = SpectrumConvLayer(in_planes, planes, stride=stride, rank=rank)
        self.conv2 = SpectrumConvLayer(planes, planes, stride=1, rank=rank)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        return F.relu(out)

class SpectrumResNet(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2, 2], rank=16):
        super().__init__()
        self.in_planes = 64
        # First layer uses 5x5 to capture Perlin structures well
        self.conv1 = SpectrumConvLayer(3, 64, kernel_size=5, stride=1, padding=2, rank=rank)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, rank=rank)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, rank=rank)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, rank=rank)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2, rank=rank)
        self.linear = nn.Linear(512, 10)

    def _make_layer(self, planes, num_blocks, stride, rank):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(SpectrumBasicBlock(self.in_planes, planes, s, rank))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return self.linear(out.view(out.size(0), -1))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V31 'THE SPECTRUM LIBRARY' (White, Perlin, Blue Noises) on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 50
    LR = 0.003
    
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
    
    train_loader = DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transform_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(datasets.CIFAR10('./data', train=False, transform=transform_test), batch_size=1024, num_workers=2)
    
    os.makedirs("results", exist_ok=True)
    
    model = SpectrumResNet().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR/10)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=len(train_loader)*EPOCHS)
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
        if acc > best_acc: best_acc = acc
        
        t_now = time.time()
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] Epoch {epoch:2d}/{EPOCHS} | Acc: {acc:.4f} | Best: {best_acc:.4f} | Time: {t_now-t0:.1f}s | ETA: {(t_now-t_start)/epoch*(EPOCHS-epoch)/60:.1f}m")

if __name__ == "__main__":
    main()
