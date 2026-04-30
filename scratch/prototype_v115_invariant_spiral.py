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

# --- Data Loading (Raw MNIST for real-time sampling) ---

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
        img, label = self.dataset[idx]
        return img, label

# --- Advanced Invariant Spiral Sampler ---

class InvariantSpiralSampler:
    def __init__(self, num_samples=1024, rotations=15, q=2.0, p=1.5, device='cpu'):
        """
        q: Angular acceleration factor (q>1 means slow start)
        p: Radial factor
        """
        self.num_samples = num_samples
        self.device = device
        
        t = torch.linspace(0, 1, num_samples, device=device)
        
        # 1. Radial path (from center out)
        r = t ** p
        
        # 2. Angular path (slow start, fast end)
        # theta = TotalRotations * 2pi * t^q
        theta = rotations * 2 * math.pi * (t ** q)
        
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        
        # Grid sample expects coords in [-1, 1]
        self.grid = torch.stack([x, y], dim=1).view(1, 1, num_samples, 2)
        
    def sample(self, x):
        """ x: (B, 1, H, W) """
        B = x.size(0)
        grid_expanded = self.grid.expand(B, -1, -1, -1)
        # We use bilinear interpolation which already provides some "averaging"
        samples = F.grid_sample(x, grid_expanded, mode='bilinear', padding_mode='zeros', align_corners=True)
        return samples.view(B, -1)

# --- Custom Neuron Layers (from v111) ---

class SpectralLayer(nn.Module):
    def __init__(self, in_features, out_features, k=16):
        super().__init__()
        self.spectral_core = nn.Parameter(torch.randn(out_features, k) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        basis = torch.zeros(k, in_features)
        for freq in range(k):
            for i in range(in_features):
                val = math.cos(math.pi * freq * (i / in_features))
                basis[freq, i] = 1.0 if val >= 0 else -1.0
        self.register_buffer('basis', basis)

    def forward(self, x):
        w = torch.matmul(self.spectral_core, self.basis)
        return torch.matmul(x, w.t()) + self.bias

# --- Invariant Model ---

class InvariantSpiralNet(nn.Module):
    def __init__(self, h=128, k=16, device='cpu'):
        super().__init__()
        self.sampler = InvariantSpiralSampler(num_samples=1024, device=device)
        self.path = nn.Sequential(
            SpectralLayer(1024, h, k=k),
            nn.ReLU(),
            nn.Linear(h, 10)
        )
    
    def forward(self, x):
        # 1. Sample from raw image
        x_spiral = self.sampler.sample(x)
        # 2. Process through spectral neurons
        return self.path(x_spiral)

# --- Training and Invariance Test ---

def evaluate_rotation(model, test_loader, angle_deg, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # Apply rotation
            if angle_deg != 0:
                # Rotate image
                angle_rad = angle_deg * math.pi / 180
                matrix = torch.tensor([
                    [math.cos(angle_rad), -math.sin(angle_rad), 0],
                    [math.sin(angle_rad), math.cos(angle_rad), 0]
                ], device=device).unsqueeze(0).repeat(data.size(0), 1, 1)
                grid = F.affine_grid(matrix[:, :2, :3], data.size(), align_corners=True)
                data = F.grid_sample(data, grid, align_corners=True)
            
            outputs = model(data)
            pred = outputs.argmax(dim=1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()
    return 100. * correct / total

def run_invariance_test():
    device = torch.device("cpu")
    train_data = MNISTRaw(train=True)
    test_data = MNISTRaw(train=False)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)
    
    model = InvariantSpiralNet(h=128, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("Training Invariant Spiral Model (5 epochs)...")
    for epoch in range(5):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed.")

    # Invariance Test
    angles = [0, 15, 30, 45, 90]
    print("\nRotation Invariance Test:")
    print(f"{'Angle':<10} | {'Accuracy':<10}")
    print("-" * 25)
    for angle in angles:
        acc = evaluate_rotation(model, test_loader, angle, device)
        print(f"{angle:3d}°       | {acc:.2f}%")

if __name__ == "__main__":
    run_invariance_test()
