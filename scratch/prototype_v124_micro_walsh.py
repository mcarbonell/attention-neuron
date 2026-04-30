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

# --- BASIS GENERATION ---

def get_walsh_matrix_sequency(N):
    def get_walsh(n):
        if n == 1: return torch.tensor([[1.0]])
        h_prev = get_walsh(n // 2)
        return torch.cat([torch.cat([h_prev, h_prev], dim=1),
                          torch.cat([h_prev, -h_prev], dim=1)], dim=0)
    H = get_walsh(N)
    crossings = [( (H[i, :-1] * H[i, 1:] < 0).sum().item(), i) for i in range(N)]
    crossings.sort()
    return H[[idx for _, idx in crossings]]

def iwalsh_2d(coeffs, H):
    N = H.shape[0]
    return torch.matmul(H.t(), torch.matmul(coeffs, H)) / (N * N)

# --- LAYERS ---

class MicroWalshLayer(nn.Module):
    def __init__(self, out_features, K=2, N=32, mode='smooth'):
        super().__init__()
        self.out_features = out_features
        self.K = K
        self.N = N
        self.mode = mode # 'smooth' or 'blocky'
        self.spectral_core = nn.Parameter(torch.randn(out_features, K, K) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('H_K', get_walsh_matrix_sequency(K))

    def get_weights(self):
        w_mini = iwalsh_2d(self.spectral_core, self.H_K).unsqueeze(1)
        interp_mode = 'bilinear' if self.mode == 'smooth' else 'nearest'
        w_32 = F.interpolate(w_mini, size=(self.N, self.N), mode=interp_mode, align_corners=False if interp_mode=='bilinear' else None)
        return w_32.view(self.out_features, -1)

    def forward(self, x):
        return F.linear(x, self.get_weights(), self.bias)

# --- MODEL ---

class MicroNet(nn.Module):
    def __init__(self, hidden_dim=128, mode='smooth', K=2):
        super().__init__()
        self.layer1 = MicroWalshLayer(hidden_dim, K=K, mode=mode)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2)).view(x.size(0), -1)
        x = F.relu(self.bn1(self.layer1(x)))
        return self.fc_out(x)

# --- BENCHMARK ---

def run_experiment(name, mode, K, epochs=10):
    device = torch.device('cpu')
    print(f"\n>>> Running: {name} (K={K})")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)
    
    model = MicroNet(mode=mode, K=K).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params:,}")
    
    best_acc = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if epoch == 1 and batch_idx < 5:
                print(f"  B{batch_idx} Loss: {loss.item():.4f}")
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        acc = correct / 10000
        if acc > best_acc: best_acc = acc
        print(f"  Epoch {epoch} Acc: {acc:.4f}")
        
    return {"acc": best_acc, "params": params}

def main():
    os.makedirs("results/raw", exist_ok=True)
    results = {}
    
    configs = [
        ("Blocky K=2", "blocky", 2),
        ("Smooth K=2", "smooth", 2),
        ("Blocky K=4", "blocky", 4),
        ("Smooth K=4", "smooth", 4),
        ("Smooth K=8", "smooth", 8),
    ]
    
    for name, mode, k in configs:
        results[name] = run_experiment(name, mode, k, epochs=10)
        
    with open("results/raw/v124_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*50)
    print(f"{'Method':20} | {'Acc':8} | {'Params':8}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:20} | {res['acc']:.4f} | {res['params']:8}")

if __name__ == "__main__":
    main()
