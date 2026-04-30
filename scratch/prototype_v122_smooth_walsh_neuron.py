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

# --- SEQUENCY-ORDERED WALSH BASIS GENERATION ---

def get_walsh_matrix(N):
    if N == 1: return torch.tensor([[1.0]])
    H_prev = get_walsh_matrix(N // 2)
    top = torch.cat([H_prev, H_prev], dim=1)
    bottom = torch.cat([H_prev, -H_prev], dim=1)
    return torch.cat([top, bottom], dim=0)

def get_walsh_matrix_sequency(N):
    H = get_walsh_matrix(N)
    crossings = []
    for i in range(N):
        row = H[i]
        num_crossings = (row[:-1] * row[1:] < 0).sum().item()
        crossings.append((num_crossings, i))
    crossings.sort()
    indices = [idx for _, idx in crossings]
    return H[indices]

def iwalsh_2d_transform(coeffs, H):
    """ Inverse: I = (1/N^2) * H^T * W * H """
    N = H.shape[0]
    return torch.matmul(H.T, torch.matmul(coeffs, H)) / (N * N)

# --- CUSTOM LAYERS ---

class BlockyWalshLayer(nn.Module):
    """ Standard Walsh Neuron: Weights are learned in Walsh space, no smoothing. """
    def __init__(self, in_features, out_features, N=32):
        super().__init__()
        self.in_features = in_features # Expecting 1024 (32x32)
        self.out_features = out_features
        self.N = N
        # We learn the 2D spectrum directly
        self.spectral_core = nn.Parameter(torch.randn(out_features, N, N) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('H', get_walsh_matrix_sequency(N))

    def get_weights(self):
        # Inverse transform to get spatial weights
        w_2d = iwalsh_2d_transform(self.spectral_core, self.H)
        # Flatten to match input features
        return w_2d.view(self.out_features, -1)

    def forward(self, x):
        w = self.get_weights()
        return F.linear(x, w, self.bias)

class SmoothWalshLayer(nn.Module):
    """ Smooth Walsh Neuron: Weights learned in low-res Walsh space + Interpolation. """
    def __init__(self, in_features, out_features, N=32, K=8):
        super().__init__()
        self.in_features = in_features # Target size (e.g. 1024)
        self.out_features = out_features
        self.N = N
        self.K = K
        # We learn a smaller spectrum
        self.spectral_core = nn.Parameter(torch.randn(out_features, K, K) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('H_K', get_walsh_matrix_sequency(K))

    def get_weights(self):
        # 1. Inverse transform to get low-res spatial weights (KxK)
        # Note: scale adjustment is handled by the division in iwalsh_2d_transform
        w_mini = iwalsh_2d_transform(self.spectral_core, self.H_K) # (out, K, K)
        
        # 2. Smooth upscaling using Bilinear Interpolation
        # Shape: (B, C, H, W) -> (out_features, 1, K, K)
        w_mini_4d = w_mini.unsqueeze(1)
        w_smooth_4d = F.interpolate(w_mini_4d, size=(self.N, self.N), mode='bilinear', align_corners=False)
        
        # 3. Flatten
        return w_smooth_4d.view(self.out_features, -1)

    def forward(self, x):
        w = self.get_weights()
        return F.linear(x, w, self.bias)

# --- MODELS ---

class WalshNet(nn.Module):
    def __init__(self, hidden_dim=128, mode='smooth', K=8):
        super().__init__()
        self.N = 32
        if mode == 'smooth':
            self.layer1 = SmoothWalshLayer(1024, hidden_dim, N=32, K=K)
        elif mode == 'blocky':
            self.layer1 = BlockyWalshLayer(1024, hidden_dim, N=32)
        else:
            self.layer1 = nn.Linear(1024, hidden_dim)
            
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = F.pad(x, (2, 2, 2, 2)) # 28x28 -> 32x32
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        return self.fc_out(x)

# --- TRAINING & BENCHMARK ---

def run_training(mode, hidden_dim=128, K=8, epochs=10):
    device = torch.device('cpu') # Following rules: small nets usually faster on CPU
    print(f"\n--- Starting Experiment: Mode={mode} (K={K if mode=='smooth' else 'N/A'}) ---")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)
    
    model = WalshNet(hidden_dim=hidden_dim, mode=mode, K=K).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params:,}")
    
    metrics = {
        "wall_clock_time": 0,
        "eval_time": 0,
        "final_acc": 0,
        "params": params
    }
    
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            t0 = time.perf_counter()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            t1 = time.perf_counter()
            
            metrics["eval_time"] += (t1 - t0)
            
            if batch_idx < 5 and epoch == 1:
                print(f"  Batch {batch_idx} | Loss: {loss.item():.4f}")
                
        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = correct / 10000
        print(f"  Epoch {epoch}/{epochs} | Acc: {acc:.4f}")
        metrics["final_acc"] = acc

    metrics["wall_clock_time"] = time.time() - t_start
    metrics["overhead_time"] = metrics["wall_clock_time"] - metrics["eval_time"]
    
    return metrics, model

def main():
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    results = {}
    
    # 1. Baseline Dense
    m_dense, _ = run_training(mode='dense', epochs=10)
    results['dense'] = m_dense
    
    # 2. Blocky Walsh (Full Res)
    m_blocky, model_blocky = run_training(mode='blocky', epochs=10)
    results['blocky'] = m_blocky
    
    # 3. Smooth Walsh (Low Res K=8)
    m_smooth8, model_smooth8 = run_training(mode='smooth', K=8, epochs=10)
    results['smooth_k8'] = m_smooth8
    
    # 4. Smooth Walsh (Medium Res K=16)
    m_smooth16, model_smooth16 = run_training(mode='smooth', K=16, epochs=10)
    results['smooth_k16'] = m_smooth16
    
    # Save Results
    with open("results/raw/v122_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n=== FINAL SUMMARY ===")
    for mode, m in results.items():
        print(f"{mode:12} | Acc: {m['final_acc']:.4f} | Params: {m['params']:6} | Time: {m['wall_clock_time']:.1f}s")

    # Visualize Weights
    import matplotlib.pyplot as plt
    
    def plot_weights(model, title, filename):
        weights = model.layer1.get_weights().detach().cpu()
        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(16):
            ax = axs[i//4, i%4]
            w_img = weights[i].view(32, 32)
            vabs = torch.abs(w_img).max().item()
            ax.imshow(w_img, cmap='RdBu', vmin=-vabs, vmax=vabs)
            ax.axis('off')
        plt.suptitle(title)
        plt.savefig(f"results/figures/{filename}")
        plt.close()

    # We need to handle Dense model separately as it has no get_weights()
    # (Actually we could just use model.layer1.weight for dense)
    
    print("\nGenerating weight visualizations...")
    plot_weights(model_blocky, "Walsh Weights (Blocky)", "v122_weights_blocky.png")
    plot_weights(model_smooth8, "Smooth Walsh Weights (K=8)", "v122_weights_smooth_k8.png")
    plot_weights(model_smooth16, "Smooth Walsh Weights (K=16)", "v122_weights_smooth_k16.png")
    
    print(f"Results saved to results/raw/v122_results.json")
    print(f"Figures saved to results/figures/")

if __name__ == "__main__":
    main()
