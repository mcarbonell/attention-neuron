import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import time
import json
import os
import math

# --- Data Loading ---

class MNISTRaw(Dataset):
    def __init__(self, train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.dataset = datasets.MNIST('./data', train=train, download=True, transform=transform)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx]

class InvariantSpiralSampler:
    def __init__(self, num_samples=1024, rotations=15, q=4.0, p=1.5, device='cpu'):
        self.num_samples = num_samples
        self.device = device
        t = torch.linspace(0, 1, num_samples, device=device)
        r = t ** p
        theta = rotations * 2 * math.pi * (t ** q)
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        self.grid = torch.stack([x, y], dim=1).view(1, 1, num_samples, 2)
        
    def sample(self, x):
        B = x.size(0)
        grid_expanded = self.grid.expand(B, -1, -1, -1)
        # Using bilinear for speed and some anti-aliasing
        samples = F.grid_sample(x, grid_expanded, mode='bilinear', padding_mode='zeros', align_corners=True)
        return samples.view(B, -1)

# --- Spectral Layer Optimized for High Resolution ---

class FastSpectralLayer(nn.Module):
    def __init__(self, in_features, out_features, k=16):
        super().__init__()
        # Parameters only depend on k
        self.spectral_core = nn.Parameter(torch.randn(out_features, k) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Precompute basis (Walsh-like)
        basis = torch.zeros(k, in_features)
        for freq in range(k):
            for i in range(in_features):
                val = math.cos(math.pi * freq * (i / in_features))
                basis[freq, i] = 1.0 if val >= 0 else -1.0
        self.register_buffer('basis', basis)

    def forward(self, x):
        # Optimization: (Batch, In) @ (In, k) -> (Batch, k) then (Batch, k) @ (k, Out)
        # This is much faster when In >> k
        spectral_proj = torch.matmul(x, self.basis.t()) # (B, k)
        return torch.matmul(spectral_proj, self.spectral_core.t()) + self.bias

class InfiniteResNet(nn.Module):
    def __init__(self, num_samples, h=64, k=16, q=4.0, device='cpu'):
        super().__init__()
        self.sampler = InvariantSpiralSampler(num_samples=num_samples, q=q, device=device)
        self.path = nn.Sequential(
            FastSpectralLayer(num_samples, h, k=k),
            nn.ReLU(),
            nn.Linear(h, 10)
        )
    def forward(self, x):
        return self.path(self.sampler.sample(x))

def evaluate_rotation(model, test_loader, angle_deg, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if angle_deg != 0:
                angle_rad = angle_deg * math.pi / 180
                matrix = torch.tensor([[math.cos(angle_rad), -math.sin(angle_rad), 0],
                                     [math.sin(angle_rad), math.cos(angle_rad), 0]], device=device).unsqueeze(0).repeat(data.size(0), 1, 1)
                grid = F.affine_grid(matrix[:, :2, :3], data.size(), align_corners=True)
                data = F.grid_sample(data, grid, align_corners=True)
            outputs = model(data)
            pred = outputs.argmax(dim=1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
    return 100. * correct / total

if __name__ == "__main__":
    device = torch.device("cpu")
    train_data = MNISTRaw(train=True)
    test_data = MNISTRaw(train=False)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)
    
    n_samples_list = [4096, 8192, 16384, 32768]
    q = 4.0 # High redundancy at center
    angles = [0, 15, 30, 45]
    
    results = []
    
    for ns in n_samples_list:
        print(f"\nTesting: num_samples={ns}, q={q} (H=64, k=16)...")
        model = InfiniteResNet(num_samples=ns, q=q, h=64, device=device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        criterion = nn.CrossEntropyLoss()
        
        # Train 3 epochs
        for epoch in range(3):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        row = {'ns': ns}
        for angle in angles:
            acc = evaluate_rotation(model, test_loader, angle, device)
            row[f'acc_{angle}'] = acc
            print(f"  Angle {angle:2d}°: {acc:.2f}%")
        results.append(row)

    print("\n" + "="*60)
    print(f"{'Samples':<8} | {'0°':<8} | {'15°':<8} | {'30°':<8} | {'45°':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['ns']:<8} | {r['acc_0']:<8.2f} | {r['acc_15']:<8.2f} | {r['acc_30']:<8.2f} | {r['acc_45']:<8.2f}")
