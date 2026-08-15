"""
v370_mnist_spectral_dct_coefficients.py
======================================
Experimento v370 (Barrido Espectral Completo): K in [4, 8, 16, 32, 64, 128, 256, 512, 784] vs Píxeles Crudos.
Evaluación estadística sobre 5 semillas independientes para trazar la curva de tasa-distorsión completa.
"""

import os
import sys
import io
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def log(msg):
    print(msg, flush=True)

# ---------------------------------------------------------------------
# 1. 2D-DCT Basis & Frequency Ordering
# ---------------------------------------------------------------------

def get_dct2d_basis(N=28, device='cpu'):
    D = torch.zeros(N, N, device=device)
    for i in range(N):
        for j in range(N):
            if i == 0:
                D[i, j] = math.sqrt(1.0 / N)
            else:
                D[i, j] = math.sqrt(2.0 / N) * math.cos((math.pi * i * (2.0 * j + 1.0)) / (2.0 * N))
    return D

def get_frequency_ordering(N=28):
    coords = []
    for u in range(N):
        for v in range(N):
            r2 = u**2 + v**2
            coords.append((r2, u, v, u * N + v))
    coords.sort(key=lambda item: (item[0], item[1]))
    sorted_flat_indices = [item[3] for item in coords]
    return torch.tensor(sorted_flat_indices, dtype=torch.long)


# ---------------------------------------------------------------------
# 2. Classifier Network
# ---------------------------------------------------------------------

class StandardMLP(nn.Module):
    def __init__(self, in_features, hidden_dim=128, num_classes=10):
        super().__init__()
        self.in_features = in_features
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)


# ---------------------------------------------------------------------
# 3. Pre-extracting & Caching Spectral Features
# ---------------------------------------------------------------------

def extract_features(loader, D_mat, sorted_idx, device='cpu'):
    all_raw = []
    all_dct = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.view(-1, 28, 28).to(device)
            dct_2d = torch.matmul(D_mat, torch.matmul(x, D_mat.t()))
            dct_flat = dct_2d.view(x.shape[0], -1)
            dct_sorted = dct_flat[:, sorted_idx]
            
            all_raw.append(x.view(x.shape[0], -1).cpu())
            all_dct.append(dct_sorted.cpu())
            all_labels.append(y)
            
    return torch.cat(all_raw, dim=0), torch.cat(all_dct, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------------------
# 4. Single-Seed Training Harness
# ---------------------------------------------------------------------

def train_single_seed(train_x, train_y, test_x, test_y, in_features, seed, epochs=5, lr=0.003, batch_size=128):
    torch.manual_seed(seed)
    device = 'cpu'
    
    model = StandardMLP(in_features=in_features, hidden_dim=128, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    n_samples = train_x.shape[0]
    n_batches = math.ceil(n_samples / batch_size)
    
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            bx = train_x[idx].to(device)
            by = train_y[idx].to(device)
            
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        test_logits = model(test_x.to(device))
        preds = test_logits.argmax(dim=-1)
        acc = (preds == test_y.to(device)).float().mean().item() * 100.0
        
    return acc


# ---------------------------------------------------------------------
# 5. Full Spectrum Sweep Benchmark
# ---------------------------------------------------------------------

def run_full_sweep_benchmark():
    t_global_start = time.time()
    device = 'cpu'
    seeds = [42, 100, 2024, 777, 999]
    n_seeds = len(seeds)
    
    log("="*105)
    log("📋 [METADATA] EXPERIMENTO v370: BARRIDO ESPECTRAL COMPLETO 2D-DCT (K=4 a K=784)")
    log("="*105)
    log("  • Coeficientes K:  [4, 8, 16, 32, 64, 128, 256, 512, 784] vs Píxeles Crudos (784)")
    log(f"  • Semillas:        {seeds} (5 semillas x 10 configuraciones = 50 runs)")
    log("  • Arquitectura:    MLP Estandarizado (Input -> 128 ReLU -> 10) | Épocas: 5")
    log(f"  • Dispositivo:      {device.upper()} | Inicio: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("="*105 + "\n")
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    log("[00:00] Precalculando bases 2D-DCT y ordenando por frecuencia...")
    D_mat = get_dct2d_basis(N=28, device=device)
    sorted_idx = get_frequency_ordering(N=28)
    
    train_raw, train_dct, train_labels = extract_features(train_loader, D_mat, sorted_idx, device=device)
    test_raw, test_dct, test_labels = extract_features(test_loader, D_mat, sorted_idx, device=device)
    log(f"[00:01] Extracción completada. Iniciando barrido...\n")
    
    k_values = [4, 8, 16, 32, 64, 128, 256, 512, 784]
    
    experiments = [("Píxeles Crudos (Base)", train_raw, test_raw, 784, 1.0)]
    for k in k_values:
        name = f"DCT Top-{k:<3} Coefs" if k < 784 else "DCT Todos (784 Coefs)"
        comp = 784.0 / float(k)
        experiments.append((name, train_dct[:, :k], test_dct[:, :k], k, comp))
        
    final_stats = []
    
    for name, tr_x, te_x, k_dim, comp_factor in experiments:
        seed_accs = []
        in_weights = k_dim * 128
        
        for s_idx, s in enumerate(seeds, 1):
            acc = train_single_seed(tr_x, train_labels, te_x, test_labels, in_features=k_dim, seed=s, epochs=5)
            seed_accs.append(acc)
            
            elapsed = time.time() - t_global_start
            mins, secs = divmod(int(elapsed), 60)
            log(f"  [{mins:02d}:{secs:02d}] {name:<24} | Semilla {s_idx}/{n_seeds} -> Test Acc: {acc:.2f}%")
            
        mean_acc = sum(seed_accs) / n_seeds
        variance = sum((x - mean_acc)**2 for x in seed_accs) / (n_seeds - 1)
        std_acc = math.sqrt(variance)
        stderr_acc = std_acc / math.sqrt(n_seeds)
        
        final_stats.append((name, k_dim, comp_factor, in_weights, mean_acc, std_acc, stderr_acc))
        log(f"  --> {name} Resumen: {mean_acc:.2f}% +- {std_acc:.2f}%\n")
        
    log("="*115)
    log("📊 MATRIZ DEL BARRIDO ESPECTRAL COMPLETO: EXPERIMENTO v370")
    log("="*115)
    log(f"{'Espacio de Entrada':<25} | {'Dim (K)':<8} | {'Compresión':<12} | {'Pesos Entrada':<14} | {'Test Acc (Media +- Std)':<24} | {'SE':<8} | {'Retención':<10}")
    log("-" * 115)
    
    base_mean = final_stats[0][4]
    for name, k_dim, comp_factor, in_w, mean_a, std_a, se_a in final_stats:
        retention = (mean_a / base_mean) * 100.0
        if mean_a >= 97.5:
            acc_str = f"🟩 {mean_a:.2f}% ± {std_a:.2f}% 🌟"
        elif mean_a >= 95.0:
            acc_str = f"🟩 {mean_a:.2f}% ± {std_a:.2f}%"
        elif mean_a >= 80.0:
            acc_str = f"🟨 {mean_a:.2f}% ± {std_a:.2f}%"
        else:
            acc_str = f"🟧 {mean_a:.2f}% ± {std_a:.2f}%"
        log(f"{name:<25} | {k_dim:<8} | {comp_factor:>10.1f}x | {in_w:>12,} | {acc_str:<24} | ±{se_a:.2f}% | {retention:>8.1f}%")
        
    log("="*115)
    
    # ASCII Rate-Distortion Curve
    log("\n📈 CURVA ESPECTRAL DE PRECISIÓN VS NÚMERO DE COEFICIENTES (K):")
    log("-" * 80)
    for name, k_dim, comp_factor, in_w, mean_a, std_a, se_a in final_stats[1:]: # Skip baseline row
        bars = int(mean_a / 2.0) # 0 to 50 chars
        bar_str = "█" * bars
        log(f"K={k_dim:<4} ({comp_factor:>5.1f}x) | {mean_a:>6.2f}% {bar_str}")
    log("-" * 80)
    
    total_time = time.time() - t_global_start
    mins_t, secs_t = divmod(int(total_time), 60)
    log(f"\nTiempo Total de Evaluación (50 entrenamientos): {mins_t:02d}:{secs_t:02d} min ({total_time:.2f}s).")

if __name__ == '__main__':
    run_full_sweep_benchmark()
