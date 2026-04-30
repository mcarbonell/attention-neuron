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

def get_dct_matrix(N):
    D = torch.zeros(N, N)
    for k in range(N):
        for n in range(N):
            if k == 0: D[k, n] = 1.0 / math.sqrt(N)
            else: D[k, n] = math.sqrt(2.0 / N) * math.cos(math.pi * k * (2 * n + 1) / (2 * N))
    return D

def idct_2d(coeffs, D):
    return torch.matmul(D.t(), torch.matmul(coeffs, D))

def iwalsh_2d(coeffs, H):
    N = H.shape[0]
    return torch.matmul(H.t(), torch.matmul(coeffs, H)) / (N * N)

# --- LAYERS ---

class SpectralBaseLayer(nn.Module):
    def __init__(self, out_features, K=8, N=32):
        super().__init__()
        self.out_features = out_features
        self.K = K
        self.N = N
        self.spectral_core = nn.Parameter(torch.randn(out_features, K, K) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

class WalshFairLayer(SpectralBaseLayer):
    def __init__(self, out_features, K=8, N=32, mode='smooth'):
        super().__init__(out_features, K, N)
        self.mode = mode # 'smooth' or 'blocky'
        self.register_buffer('H_K', get_walsh_matrix_sequency(K))

    def get_weights(self):
        w_mini = iwalsh_2d(self.spectral_core, self.H_K).unsqueeze(1)
        interp_mode = 'bilinear' if self.mode == 'smooth' else 'nearest'
        w_32 = F.interpolate(w_mini, size=(self.N, self.N), mode=interp_mode, align_corners=False if interp_mode=='bilinear' else None)
        return w_32.view(self.out_features, -1)

    def forward(self, x):
        return F.linear(x, self.get_weights(), self.bias)

class DCTFairLayer(SpectralBaseLayer):
    def __init__(self, out_features, K=8, N=32, mode='smooth'):
        super().__init__(out_features, K, N)
        self.mode = mode # 'smooth' or 'pure' (spectral padding)
        self.register_buffer('D_K', get_dct_matrix(K))
        self.register_buffer('D_N', get_dct_matrix(N))

    def get_weights(self):
        if self.mode == 'smooth':
            w_mini = idct_2d(self.spectral_core, self.D_K).unsqueeze(1)
            w_32 = F.interpolate(w_mini, size=(self.N, self.N), mode='bilinear', align_corners=False)
        else: # 'pure' - Spectral padding
            spectrum_padded = torch.zeros(self.out_features, self.N, self.N, device=self.spectral_core.device)
            spectrum_padded[:, :self.K, :self.K] = self.spectral_core
            w_32 = idct_2d(spectrum_padded, self.D_N)
            
        return w_32.view(self.out_features, -1)

    def forward(self, x):
        return F.linear(x, self.get_weights(), self.bias)

# --- MODEL ---

class ComparisonNet(nn.Module):
    def __init__(self, hidden_dim=128, layer_type='walsh_smooth', K=8):
        super().__init__()
        l_type, l_mode = layer_type.split('_')
        if l_type == 'walsh':
            self.layer1 = WalshFairLayer(hidden_dim, K=K, mode=l_mode)
        else: # dct
            self.layer1 = DCTFairLayer(hidden_dim, K=K, mode=l_mode)
            
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2)).view(x.size(0), -1)
        x = F.relu(self.bn1(self.layer1(x)))
        return self.fc_out(x)

# --- BENCHMARK ---

def run_experiment(name, layer_type, K, epochs=10):
    device = torch.device('cpu')
    print(f"\n>>> Running: {name} (K={K})")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)
    
    model = ComparisonNet(layer_type=layer_type, K=K).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params:,}")
    
    best_acc = 0
    t_start = time.time()
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
        
    return {"acc": best_acc, "params": params, "time": time.time() - t_start}

def main():
    os.makedirs("results/raw", exist_ok=True)
    results = {}
    
    configs = [
        ("Walsh Blocky K=8", "walsh_blocky", 8),
        ("Walsh Smooth K=8", "walsh_smooth", 8),
        ("Walsh Blocky K=16", "walsh_blocky", 16),
        ("Walsh Smooth K=16", "walsh_smooth", 16),
        ("DCT Pure K=8", "dct_pure", 8),
        ("DCT Smooth K=8", "dct_smooth", 8),
        ("DCT Pure K=16", "dct_pure", 16),
        ("DCT Smooth K=16", "dct_smooth", 16),
    ]
    
    for name, ltype, k in configs:
        results[name] = run_experiment(name, ltype, k, epochs=10)
        
    with open("results/raw/v123_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*50)
    print(f"{'Method':20} | {'Acc':8} | {'Params':8}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:20} | {res['acc']:.4f} | {res['params']:8}")

if __name__ == "__main__":
    main()
