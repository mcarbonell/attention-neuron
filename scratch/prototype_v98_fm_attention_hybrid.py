import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
import json
import math

# --- Fast Walsh-Hadamard Transform (Recursive/Iterative) ---
def fwht(x):
    """
    Iterative implementation of Fast Walsh-Hadamard Transform.
    x: (B, N) where N is a power of 2.
    """
    B, N = x.shape
    levels = int(math.log2(N))
    for i in range(levels):
        h = 2**i
        x = x.view(B, N // (2 * h), 2, h)
        x_top = x[:, :, 0, :]
        x_bot = x[:, :, 1, :]
        x = torch.stack([x_top + x_bot, x_top - x_bot], dim=2)
    return x.view(B, N) / math.sqrt(N)

# --- Fourier-Mellin Transform Layer ---
class FourierMellinTransform(nn.Module):
    def __init__(self, num_samples=1024, device='cpu'):
        super().__init__()
        self.num_samples = num_samples
        self.device = device
        
        # Pre-calculate Log-Polar grid
        t = torch.linspace(0, 1, num_samples, device=device)
        r = t ** 1.2
        theta = 12 * math.pi * t
        
        x_grid = r * torch.cos(theta)
        y_grid = r * torch.sin(theta)
        self.grid = torch.stack([x_grid, y_grid], dim=1).view(1, 1, num_samples, 2)

    def forward(self, x):
        B = x.size(0)
        # 1. FFT2 Magnitude
        X = torch.fft.fft2(x)
        X_mag = torch.abs(X)
        X_mag = torch.fft.fftshift(X_mag, dim=(-2, -1))
        
        # 2. Log-Polar Sampling
        X_lp = F.grid_sample(X_mag, self.grid.expand(B, -1, -1, -1), 
                             mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # 3. Second FFT Magnitude (Mellin)
        X_lp = X_lp.view(B, 1, self.num_samples)
        X_final = torch.fft.fft(X_lp)
        X_final_mag = torch.abs(X_final)
        
        return X_final_mag.view(B, -1)

# --- Spectral Attention Layer ---
class SpectralAttentionLayer(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.dim = dim
        # Learned frequency mask (Attention)
        self.mask = nn.Parameter(torch.ones(1, dim))
        
    def forward(self, x):
        # x is the invariant signature (1024)
        # 1. Project to Walsh domain
        w = fwht(x)
        # 2. Apply Attention Mask
        w_attended = w * self.mask
        # 3. Back to spatial domain
        out = fwht(w_attended)
        return out

# --- Hybrid V98 Model ---
class HybridAttentionNet(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super().__init__()
        self.fm = FourierMellinTransform(num_samples=input_dim)
        self.attention = SpectralAttentionLayer(dim=input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        z_inv = self.fm(x)
        z_att = self.attention(z_inv)
        return self.classifier(z_att)

def run_experiment(mode="hybrid", epochs=10, lr=0.001, device='cpu'):
    print(f"\n--- Running Experiment: {mode.upper()} ---", flush=True)
    
    BATCH_SIZE = 128
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    # TORTURE TEST: Heavy rotations and shifts
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(90),
        transforms.RandomAffine(0, translate=(0.2, 0.2)),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_set = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_set = datasets.MNIST('./data', train=False, transform=test_transform)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000)
    
    if mode == "hybrid":
        model = HybridAttentionNet().to(device)
    else:
        # Simple FM + MLP (V97)
        from prototype_v97_fourier_mellin_mnist import StandardMLP, FourierMellinTransform as FM_Layer
        class V97Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fm = FM_Layer(num_samples=1024)
                self.classifier = StandardMLP(input_dim=1024)
            def forward(self, x):
                return self.classifier(self.fm(x))
        model = V97Model().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = F.pad(data, (2, 2, 2, 2))
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = F.pad(data, (2, 2, 2, 2))
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = correct / 10000
        print(f"  Epoch {epoch} | TEST ACC (Torture): {acc:.4f}", flush=True)
        history.append({"epoch": epoch, "acc": acc})
        
    return {"mode": mode, "final_acc": acc, "history": history}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Info: {device}", flush=True)
    
    results = {}
    
    # 1. Run V97 (FM + MLP)
    results['v97'] = run_experiment(mode="v97", epochs=10, device=device)
    
    # 2. Run V98 (FM + Spectral Attention)
    results['v98'] = run_experiment(mode="hybrid", epochs=10, device=device)
    
    os.makedirs('results/raw', exist_ok=True)
    with open('results/raw/v98_hybrid_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- FINAL HYBRID RESULTS ---", flush=True)
    print(f"V97 (FM + MLP) Final Acc:       {results['v97']['final_acc']:.4f}", flush=True)
    print(f"V98 (FM + Attention) Final Acc: {results['v98']['final_acc']:.4f}", flush=True)
    
    improvement = (results['v98']['final_acc'] - results['v97']['final_acc']) * 100
    print(f"Spectral Attention Gain: {improvement:+.2f}%", flush=True)

if __name__ == "__main__":
    main()
