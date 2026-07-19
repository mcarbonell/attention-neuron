import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import math
import json
import os

# --- 1. Fast Walsh-Hadamard Transform (FWHT) ---
def fwht(x):
    B, N = x.shape
    h = 1
    while h < N:
        x = x.view(B, N // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        h *= 2
    return x.view(B, N)

def ifwht(x):
    N = x.shape[-1]
    return fwht(x) / N

# --- 2. Discrete Cosine Transform (DCT-II) ---
def create_dct_matrix(N):
    n = torch.arange(N, dtype=torch.float32)
    k = torch.arange(N, dtype=torch.float32).unsqueeze(1)
    C = torch.cos(math.pi / (2 * N) * (2 * n + 1) * k)
    C[0, :] *= 1.0 / math.sqrt(2)
    C *= math.sqrt(2.0 / N)
    return C

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


# --- ARCHITECTURES (All Iso-Parameter ~4,096 Trainable Params) ---

class BlockyWalshMegaLayer(nn.Module):
    """ Standard Blocky Walsh Mega Layer (v87): Hard truncation in Walsh spectrum. """
    def __init__(self, N=16384, K=64):
        super().__init__()
        self.N = N
        self.K = K
        self.core = nn.Parameter(torch.randn(K, K) * 0.02)

    def forward(self, x):
        x_spec = fwht(x)
        low_freq = x_spec[:, :self.K]
        modulated = torch.matmul(low_freq, self.core)
        full_spec = torch.zeros_like(x_spec)
        full_spec[:, :self.K] = modulated
        return ifwht(full_spec)


class SmoothWalshSpectralLayer(nn.Module):
    """ 
    Smooth Walsh Mega Layer (Spectral Low-Pass Filter):
    Applies continuous low-pass filter to high Walsh sequencies to eliminate
    sharp blocky step-function aliasing artifacts.
    """
    def __init__(self, N=16384, K=64):
        super().__init__()
        self.N = N
        self.K = K
        self.core = nn.Parameter(torch.randn(K, K) * 0.02)
        
        k_idx = torch.arange(K, dtype=torch.float32)
        smooth_mask = torch.cos(0.5 * math.pi * k_idx / K)
        self.register_buffer('smooth_mask', smooth_mask)

    def forward(self, x):
        x_spec = fwht(x)
        low_freq = x_spec[:, :self.K]
        smooth_low_freq = low_freq * self.smooth_mask
        modulated = torch.matmul(smooth_low_freq, self.core)
        
        full_spec = torch.zeros_like(x_spec)
        full_spec[:, :self.K] = modulated * self.smooth_mask
        return ifwht(full_spec)


class SmoothWalshSpatialInterpolatedLayer(nn.Module):
    """
    Smooth Walsh Spatial Interpolated Layer (v122 2D Bilinear Interpolation):
    Inverse 2D Walsh transform of 64x64 core (4096 params) -> 2D Bilinear Interpolation to 128x128.
    Precomputes the upscaled smooth weight vector for maximum speed!
    """
    def __init__(self, N=16384, grid_K=64, grid_N=128):
        super().__init__()
        self.N = N
        self.grid_K = grid_K
        self.grid_N = grid_N
        
        self.core = nn.Parameter(torch.randn(grid_K, grid_K) * 0.02)
        self.register_buffer('H_K', get_walsh_matrix_sequency(grid_K))

    def get_smooth_weights(self):
        w_low_res = torch.matmul(self.H_K.T, torch.matmul(self.core, self.H_K)) / (self.grid_K * self.grid_K)
        w_4d = w_low_res.unsqueeze(0).unsqueeze(0)
        w_smooth_4d = F.interpolate(w_4d, size=(self.grid_N, self.grid_N), mode='bilinear', align_corners=False)
        return w_smooth_4d.view(-1)

    def forward(self, x):
        w_smooth = self.get_smooth_weights()
        return x * w_smooth


class DCTMegaLayer(nn.Module):
    """ Continuous Discrete Cosine Transform (DCT) Mega Layer (Optimized 256x). """
    def __init__(self, N=16384, K=64):
        super().__init__()
        self.N = N
        self.K = K
        self.core = nn.Parameter(torch.randn(K, K) * 0.02)
        C_full = create_dct_matrix(N)
        # Sliced top-K basis vectors (K x N)
        self.register_buffer('C_K', C_full[:K, :])

    def forward(self, x):
        # 1. Project input directly to top-K DCT frequencies: O(N * K) instead of O(N^2)
        low_freq = torch.matmul(x, self.C_K.T) # (B, K)
        # 2. Modulate with core: O(K^2)
        modulated = torch.matmul(low_freq, self.core) # (B, K)
        # 3. Inverse transform from top-K DCT frequencies back to 16K spatial domain: O(K * N)
        return torch.matmul(modulated, self.C_K) # (B, N)


class PCADenseLayer(nn.Module):
    """ Data-Informed PCA Subspace Baseline. """
    def __init__(self, dataset, N=16384, K=64):
        super().__init__()
        self.N = N
        self.K = K
        _, _, Vh = torch.linalg.svd(dataset, full_matrices=False)
        P_pca_in = Vh[:K, :].T
        P_pca_out = Vh[:K, :]
        self.register_buffer('P_in', P_pca_in)
        self.register_buffer('P_out', P_pca_out)
        self.core = nn.Parameter(torch.randn(K, K) * 0.02)

    def forward(self, x):
        x_proj = torch.matmul(x, self.P_in)
        modulated = torch.matmul(x_proj, self.core)
        return torch.matmul(modulated, self.P_out)


# --- DATASETS ---

def generate_dct_decay_dataset(num_samples=512, N=16384, alpha=2.0, seed=42):
    torch.manual_seed(seed)
    C = create_dct_matrix(N)
    k_idx = torch.arange(1, N + 1, dtype=torch.float32)
    weights = 1.0 / (k_idx ** alpha)
    weights = weights / weights.sum()
    rand_spec = torch.randn(num_samples, N) * torch.sqrt(weights)
    return torch.matmul(rand_spec, C)

def generate_walsh_decay_dataset(num_samples=512, N=16384, alpha=2.0, seed=42):
    torch.manual_seed(seed)
    k_idx = torch.arange(1, N + 1, dtype=torch.float32)
    weights = 1.0 / (k_idx ** alpha)
    weights = weights / weights.sum()
    rand_spec = torch.randn(num_samples, N) * torch.sqrt(weights)
    return ifwht(rand_spec)


def run_benchmark(dataset_name, dataset, N=16384, K=64, epochs=40, lr=1e-2):
    device = torch.device('cpu')
    num_samples = dataset.shape[0]
    batch_size = 32
    num_batches = math.ceil(num_samples / batch_size)
    
    models = [
        ("Blocky Walsh (v87 Hard Truncation)", lambda ds: BlockyWalshMegaLayer(N=N, K=K)),
        ("Smooth Walsh (Spectral Low-Pass)", lambda ds: SmoothWalshSpectralLayer(N=N, K=K)),
        ("Smooth Walsh (2D Bilinear Interpolation)", lambda ds: SmoothWalshSpatialInterpolatedLayer(N=N, grid_K=64, grid_N=128)),
        ("DCT Mega-Layer (Continuous Cosine)", lambda ds: DCTMegaLayer(N=N, K=K)),
        ("PCA-Informed Baseline (SVD Top-64)", lambda ds: PCADenseLayer(ds, N=N, K=K))
    ]
    
    print(f"\n" + "="*85)
    print(f"  BENCHMARK: Dataset [{dataset_name}] (N={N}, K={K}, Samples={num_samples})")
    print("="*85)
    
    results = {}
    
    for name, model_fn in models:
        torch.manual_seed(42)
        model = model_fn(dataset).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        t_start_wall = time.time()
        t_eval_total = 0.0
        loss_history = []
        
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            perm = torch.randperm(num_samples)
            shuffled = dataset[perm]
            
            for b in range(num_batches):
                batch_x = shuffled[b * batch_size : (b + 1) * batch_size].to(device)
                optimizer.zero_grad()
                
                t_eval_start = time.time()
                output = model(batch_x)
                loss = criterion(output, batch_x)
                t_eval_end = time.time()
                t_eval_total += (t_eval_end - t_eval_start)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_x.size(0)
                
                if epoch == 1 and b == 0:
                    print(f"  [{name[:35]:<35}] Ep 1 Batch 1 Loss: {loss.item():.6f}")
                    
            avg_loss = epoch_loss / num_samples
            loss_history.append(avg_loss)
            
        wall_time = time.time() - t_start_wall
        pei = (1.0 / loss_history[-1]) / math.log10(trainable_params + 1)
        
        results[name] = {
            "model_name": name,
            "trainable_params": trainable_params,
            "final_objective": loss_history[-1],
            "wall_clock_time": wall_time,
            "function_evaluation_time": t_eval_total,
            "internal_overhead_time": wall_time - t_eval_total,
            "PEI": pei,
            "loss_history": loss_history
        }
        print(f"  > {name:<40} | Loss: {loss_history[-1]:.8f} | PEI: {pei:.2f}")
        
    return results


def main():
    print("--- V87d: SMOOTH WALSH VS BLOCKY WALSH BENCHMARK (16K MEGA-LAYER) ---")
    N = 16384
    K = 64
    EPOCHS = 40
    SAMPLES = 512
    ALPHA = 2.0
    
    # 1. Dataset 1: Continuous DCT Decay Data
    print("\n[1/2] Generating Continuous Signal Dataset (DCT Decay)...")
    ds_dct = generate_dct_decay_dataset(num_samples=SAMPLES, N=N, alpha=ALPHA, seed=42)
    res_dct = run_benchmark("Continuous DCT Signal", ds_dct, N=N, K=K, epochs=EPOCHS)
    
    # 2. Dataset 2: Step Walsh Decay Data
    print("\n[2/2] Generating Step Signal Dataset (Walsh Decay)...")
    ds_walsh = generate_walsh_decay_dataset(num_samples=SAMPLES, N=N, alpha=ALPHA, seed=42)
    res_walsh = run_benchmark("Step Walsh Signal", ds_walsh, N=N, K=K, epochs=EPOCHS)
    
    summary = {
        "continuous_dct_signal": res_dct,
        "step_walsh_signal": res_walsh
    }
    
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v87d_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[SUCCESS] Benchmark complete! Results written to results/raw/v87d_results.json")

if __name__ == "__main__":
    main()
