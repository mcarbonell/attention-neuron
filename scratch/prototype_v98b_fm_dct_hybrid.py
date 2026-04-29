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

# --- 1D DCT Implementation ---
class DCT1D(nn.Module):
    def __init__(self, N, device='cpu'):
        super().__init__()
        self.N = N
        # Precompute DCT matrix
        matrix = torch.zeros(N, N, device=device)
        for k in range(N):
            for n in range(N):
                matrix[k, n] = math.cos(math.pi * k * (n + 0.5) / N)
        # Normalization factors
        alpha = torch.ones(N, device=device) * math.sqrt(2.0 / N)
        alpha[0] = math.sqrt(1.0 / N)
        self.register_buffer('matrix', matrix * alpha.view(N, 1))
        self.register_buffer('inv_matrix', self.matrix.t())

    def forward(self, x):
        # x: (B, N)
        return x @ self.matrix.t()

    def inverse(self, X):
        # X: (B, N)
        return X @ self.inv_matrix.t()

# --- Fourier-Mellin Transform Layer ---
class FourierMellinTransform(nn.Module):
    def __init__(self, num_samples=1024, device='cpu'):
        super().__init__()
        self.num_samples = num_samples
        self.device = device
        t = torch.linspace(0, 1, num_samples, device=device)
        r = t ** 1.2
        theta = 12 * math.pi * t
        x_grid = r * torch.cos(theta)
        y_grid = r * torch.sin(theta)
        self.grid = torch.stack([x_grid, y_grid], dim=1).view(1, 1, num_samples, 2)

    def forward(self, x):
        B = x.size(0)
        X = torch.fft.fft2(x)
        X_mag = torch.abs(X)
        X_mag = torch.fft.fftshift(X_mag, dim=(-2, -1))
        X_lp = F.grid_sample(X_mag, self.grid.expand(B, -1, -1, -1), 
                             mode='bilinear', padding_mode='zeros', align_corners=True)
        X_lp = X_lp.view(B, 1, self.num_samples)
        X_final = torch.fft.fft(X_lp)
        X_final_mag = torch.abs(X_final)
        return X_final_mag.view(B, -1)

# --- DCT Attention Layer ---
class DCTAttentionLayer(nn.Module):
    def __init__(self, dim=1024, device='cpu'):
        super().__init__()
        self.dct = DCT1D(dim, device=device)
        self.mask = nn.Parameter(torch.ones(1, dim))
        
    def forward(self, x):
        # 1. Project to DCT domain
        X_dct = self.dct(x)
        # 2. Apply Mask
        X_attended = X_dct * self.mask
        # 3. Inverse DCT
        out = self.dct.inverse(X_attended)
        return out

# --- Hybrid V98b Model (DCT) ---
class HybridDCTAttentionNet(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, device='cpu'):
        super().__init__()
        self.fm = FourierMellinTransform(num_samples=input_dim, device=device)
        self.attention = DCTAttentionLayer(dim=input_dim, device=device)
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

def run_experiment(mode="dct", epochs=10, lr=0.001, device='cpu'):
    print(f"\n--- Running Experiment: {mode.upper()} ---", flush=True)
    
    BATCH_SIZE = 128
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
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
    
    if mode == "dct":
        model = HybridDCTAttentionNet(device=device).to(device)
    else:
        # Import Walsh version (V98)
        from prototype_v98_fm_attention_hybrid import HybridAttentionNet
        model = HybridAttentionNet().to(device)

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
    
    # 1. Run V98 (Walsh Hybrid)
    results['walsh'] = run_experiment(mode="walsh", epochs=10, device=device)
    
    # 2. Run V98b (DCT Hybrid)
    results['dct'] = run_experiment(mode="dct", epochs=10, device=device)
    
    os.makedirs('results/raw', exist_ok=True)
    with open('results/raw/v98b_dct_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- FINAL HYBRID COMPARISON ---", flush=True)
    print(f"V98 (Walsh) Final Acc: {results['walsh']['final_acc']:.4f}", flush=True)
    print(f"V98b (DCT) Final Acc:  {results['dct']['final_acc']:.4f}", flush=True)
    
    diff = (results['dct']['final_acc'] - results['walsh']['final_acc']) * 100
    print(f"DCT vs Walsh Delta: {diff:+.2f}%", flush=True)

if __name__ == "__main__":
    main()
