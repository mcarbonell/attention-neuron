import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import time
import math
import os

# --- Data Loading (Raw MNIST) ---

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

# --- Concentric Rings Sampler ---

class ConcentricRingsSampler:
    def __init__(self, num_rings=32, samples_per_ring=64, device='cpu'):
        self.num_rings = num_rings
        self.samples_per_ring = samples_per_ring
        self.device = device
        
        # Radii from 0 to 1
        radii = torch.linspace(0.05, 1.0, num_rings, device=device)
        # Angles from 0 to 2pi
        theta = torch.linspace(0, 2*math.pi, samples_per_ring + 1, device=device)[:-1]
        
        # Grid: (num_rings, samples_per_ring, 2)
        grid = []
        for r in radii:
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            grid.append(torch.stack([x, y], dim=1))
        
        self.grid = torch.stack(grid, dim=0).view(1, num_rings * samples_per_ring, 2)
        
    def sample_and_fft(self, x):
        """
        x: (B, 1, 28, 28)
        Returns: (B, num_rings * (samples_per_ring//2 + 1))
        """
        B = x.size(0)
        # Pad to 32x32 for better sampling
        x_padded = F.pad(x, (2, 2, 2, 2))
        
        grid_expanded = self.grid.expand(B, -1, -1, -1).to(x.device)
        samples = F.grid_sample(x_padded, grid_expanded, mode='bilinear', padding_mode='zeros', align_corners=True)
        # Reshape to (B, num_rings, samples_per_ring)
        samples = samples.view(B, self.num_rings, self.samples_per_ring)
        
        # Compute FFT per ring and take magnitude
        # fft.rfft works on the last dimension
        spectral_rings = torch.fft.rfft(samples, dim=2)
        magnitude = torch.abs(spectral_rings) # (B, num_rings, samples_per_ring//2 + 1)
        
        return magnitude.view(B, -1)

# --- Custom Layers ---

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
        spectral_proj = torch.matmul(x, self.basis.t())
        return torch.matmul(spectral_proj, self.spectral_core.t()) + self.bias

# --- Invariant Rings Net ---

class SpectralRingsNet(nn.Module):
    def __init__(self, h=128, k=16, device='cpu'):
        super().__init__()
        self.sampler = ConcentricRingsSampler(num_rings=32, samples_per_ring=64, device=device)
        # Input size: 32 rings * (64/2 + 1) coeffs = 32 * 33 = 1056
        self.path = nn.Sequential(
            SpectralLayer(1056, h, k=k),
            nn.ReLU(),
            nn.Linear(h, 10)
        )
    
    def forward(self, x):
        # 1. Extract rotation-invariant spectral signature
        x_sig = self.sampler.sample_and_fft(x)
        # 2. Classify
        return self.path(x_sig)

# --- Training and Rotation Test ---

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

def run_experiment():
    device = torch.device("cpu")
    train_loader = DataLoader(MNISTRaw(train=True), batch_size=128, shuffle=True)
    test_loader = DataLoader(MNISTRaw(train=False), batch_size=1000)
    
    model = SpectralRingsNet(h=128, k=16, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    print("Training Spectral Rings Model (5 epochs)...")
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

    # Rotation Test
    angles = [0, 15, 30, 45, 90, 180]
    print("\nSpectral Rings Rotation Invariance Test:")
    print(f"{'Angle':<10} | {'Accuracy':<10}")
    print("-" * 25)
    for angle in angles:
        acc = evaluate_rotation(model, test_loader, angle, device)
        print(f"{angle:3d}°       | {acc:.2f}%")

if __name__ == "__main__":
    run_experiment()
