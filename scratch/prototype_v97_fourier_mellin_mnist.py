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

# --- Fourier-Mellin Transform Pipeline ---
class FourierMellinTransform(nn.Module):
    def __init__(self, num_samples=1024, size=32, device='cpu'):
        super().__init__()
        self.num_samples = num_samples
        self.size = size
        self.device = device
        
        # Pre-calculate Log-Polar grid for sampling the FFT spectrum
        # We sample frequencies in a log-polar way
        t = torch.linspace(0, 1, num_samples, device=device)
        r = t ** 1.2  # Radial spacing
        theta = 12 * math.pi * t # Spiral rotations
        
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        
        self.grid = torch.stack([x, y], dim=1).view(1, 1, num_samples, 2)

    def forward(self, x):
        """
        x: (B, 1, 32, 32)
        Returns: Invariant signature (B, num_samples)
        """
        B = x.size(0)
        
        # 1. FFT2 + Magnitude (Translation Invariance)
        # Use rfft2 for real input to be faster, but fft2 is more general for Fourier-Mellin
        X = torch.fft.fft2(x)
        X_mag = torch.abs(X)
        X_mag = torch.fft.fftshift(X_mag, dim=(-2, -1)) # Center the DC component
        
        # 2. Log-Polar Mapping (Rotation/Scale -> Shift)
        # We sample the centered magnitude spectrum
        X_lp = F.grid_sample(X_mag, self.grid.expand(B, -1, -1, -1), 
                             mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # 3. Second FFT (Magnitude) - Mellin Transform part
        # This removes the "shift" in log-polar space caused by rotation/scale
        # We treat the log-polar map as a 1D signal
        X_lp = X_lp.view(B, 1, self.num_samples)
        X_final = torch.fft.fft(X_lp)
        X_final_mag = torch.abs(X_final)
        
        return X_final_mag.view(B, -1)

# --- Standard MLP for Classification ---
class StandardMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        return self.net(x)

def run_experiment(mode="raster", epochs=10, lr=0.001, device='cpu'):
    print(f"\n--- Running Experiment: {mode.upper()} ---", flush=True)
    
    BATCH_SIZE = 128
    
    # Training transform: Normal MNIST
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    # TEST TRANSFORM: TORTURE TEST (Rotations and Shifts)
    # We want to see if the model generalizes to unseen orientations
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(90), # HEAVY Rotation
        transforms.RandomAffine(0, translate=(0.2, 0.2)), # HEAVY Shift
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_set = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_set = datasets.MNIST('./data', train=False, transform=test_transform)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000)
    
    fm_layer = FourierMellinTransform(num_samples=1024, device=device)
    model = StandardMLP(input_dim=1024).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    t_start = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = F.pad(data, (2, 2, 2, 2)) # to 32x32
            
            if mode == "fm":
                data_in = fm_layer(data)
            else:
                data_in = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            output = model(data_in)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 200 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx:3d} | Loss: {loss.item():.4f}", flush=True)

        # Eval on TORTURE TEST
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = F.pad(data, (2, 2, 2, 2))
                if mode == "fm":
                    data_in = fm_layer(data)
                else:
                    data_in = data.view(data.size(0), -1)
                pred = model(data_in).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = correct / 10000
        print(f"  Epoch {epoch} | TEST ACC (Rotated/Shifted): {acc:.4f}", flush=True)
        history.append({"epoch": epoch, "acc": acc})
        
    return {"mode": mode, "final_acc": acc, "history": history}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Info: {device}", flush=True)
    
    results = {}
    
    # Run Baseline (Raster) - Should fail on rotated test set
    results['raster'] = run_experiment(mode="raster", epochs=10, device=device)
    
    # Run Fourier-Mellin - Should be robust
    results['fm'] = run_experiment(mode="fm", epochs=10, device=device)
    
    os.makedirs('results/raw', exist_ok=True)
    with open('results/raw/v97_fourier_mellin_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- TORTURE TEST RESULTS SUMMARY ---", flush=True)
    print(f"Raster Final Acc (Rotated/Shifted): {results['raster']['final_acc']:.4f}", flush=True)
    print(f"FM Final Acc (Rotated/Shifted):     {results['fm']['final_acc']:.4f}", flush=True)
    
    improvement = (results['fm']['final_acc'] - results['raster']['final_acc']) * 100
    print(f"Invariance Boost: {improvement:+.2f}%", flush=True)

if __name__ == "__main__":
    main()
