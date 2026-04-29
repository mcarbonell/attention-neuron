import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import os
import json

# --- Optimized Fast Walsh-Hadamard Transform ---
def fwht(x):
    """Computes the Fast Walsh-Hadamard Transform of a batch of vectors."""
    B, C, N = x.shape
    h = 1
    while h < N:
        x = x.view(B, C, -1, 2, h)
        a = x[:, :, :, 0, :]
        b = x[:, :, :, 1, :]
        x = torch.cat([a + b, a - b], dim=3)
        h *= 2
    return x.view(B, C, N)

def ifwht(x):
    N = x.shape[-1]
    return fwht(x) / N

# --- Spiral Index Generation ---
def get_spiral_indices(size=32):
    """Correct center-out spiral for any square grid."""
    indices = []
    r, c = size // 2, size // 2
    indices.append(r * size + c)
    
    step = 1
    while len(indices) < size * size:
        # Move Right
        for _ in range(step):
            c += 1
            if 0 <= r < size and 0 <= c < size:
                indices.append(r * size + c)
        # Move Down
        for _ in range(step):
            r += 1
            if 0 <= r < size and 0 <= c < size:
                indices.append(r * size + c)
        step += 1
        # Move Left
        for _ in range(step):
            c -= 1
            if 0 <= r < size and 0 <= c < size:
                indices.append(r * size + c)
        # Move Up
        for _ in range(step):
            r -= 1
            if 0 <= r < size and 0 <= c < size:
                indices.append(r * size + c)
        step += 1
        
    return indices[:size*size]

# --- Walsh Attention Neuron Model ---
class WalshMNISTNet(nn.Module):
    def __init__(self, hidden_dim=32, input_size=1024):
        super().__init__()
        self.N = input_size
        self.delta_m = nn.Parameter(torch.randn(hidden_dim, self.N) * 0.01)
        self.delta_a = nn.Parameter(torch.zeros(hidden_dim, self.N))
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc_final = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        # x expected to be (B, 1, N)
        x_walsh = fwht(x)
        x_filtered = x_walsh * (1.0 + self.delta_m) + self.delta_a
        x_spatial = ifwht(x_filtered)
        x_features = x_spatial.mean(dim=2)
        x = self.bn1(x_features)
        return self.fc_final(F.relu(x))

def run_experiment(mode="raster", epochs=3, lr=0.01, device='cpu'):
    print(f"\n--- Running Experiment: {mode.upper()} ---", flush=True)
    
    BATCH_SIZE = 128
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000)
    
    model = WalshMNISTNet(hidden_dim=32).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Pre-calculate indices for reordering
    spiral_indices = torch.tensor(get_spiral_indices(32), device=device)
    
    history = []
    wall_clock_start = time.time()
    total_evals = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = F.pad(data, (2, 2, 2, 2)) 
            B = data.size(0)
            data_flat = data.view(B, 1, 1024)
            
            if mode == "spiral":
                data_flat = data_flat[:, :, spiral_indices]
            
            optimizer.zero_grad()
            output = model(data_flat)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_evals += B
            
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx:3d} | Loss: {loss.item():.4f}", flush=True)

        # Eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = F.pad(data, (2, 2, 2, 2))
                data_flat = data.view(data.size(0), 1, 1024)
                if mode == "spiral":
                    data_flat = data_flat[:, :, spiral_indices]
                pred = model(data_flat).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = correct / 10000
        t_epoch = time.time() - t0
        print(f"  Epoch {epoch} | Acc: {acc:.4f} | Time: {t_epoch:.1f}s", flush=True)
        history.append({"epoch": epoch, "acc": acc, "loss": loss.item()})
        
    wall_clock_time = time.time() - wall_clock_start
    return {
        "mode": mode,
        "final_acc": acc,
        "total_evaluations": total_evals,
        "wall_clock_time": wall_clock_time,
        "history": history
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Info: {device}", flush=True)
    
    results = {}
    
    # Run Baseline
    results['raster'] = run_experiment(mode="raster", epochs=3, device=device)
    
    # Run Spiral
    results['spiral'] = run_experiment(mode="spiral", epochs=3, device=device)
    
    # Save results
    os.makedirs('results/raw', exist_ok=True)
    with open('results/raw/v93_spiral_vs_raster.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- Final Results Summary ---", flush=True)
    print(f"Raster Final Acc: {results['raster']['final_acc']:.4f}", flush=True)
    print(f"Spiral Final Acc: {results['spiral']['final_acc']:.4f}", flush=True)
    
    improvement = (results['spiral']['final_acc'] - results['raster']['final_acc']) * 100
    print(f"Improvement: {improvement:+.2f}%", flush=True)

if __name__ == "__main__":
    main()
