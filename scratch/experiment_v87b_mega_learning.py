import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import json
import os

# --- Fast Walsh-Hadamard Transform (Vectorized) ---
def fwht(x):
    """
    Computes the Fast Walsh-Hadamard Transform of a batch of vectors.
    Input x: (B, N) where N must be a power of 2.
    """
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

# --- Architectures (all iso-parameter ~4,096 trainable params) ---

class SpectralMegaLayer(nn.Module):
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

class ProjectedDenseLayer(nn.Module):
    def __init__(self, N=16384, K=64):
        super().__init__()
        self.N = N
        self.K = K
        # Fixed random orthogonal spatial projections (frozen)
        P_in = torch.randn(N, K) / math.sqrt(N)
        P_out = torch.randn(K, N) / math.sqrt(K)
        self.register_buffer('P_in', P_in)
        self.register_buffer('P_out', P_out)
        self.core = nn.Parameter(torch.randn(K, K) * 0.02)

    def forward(self, x):
        x_proj = torch.matmul(x, self.P_in)
        modulated = torch.matmul(x_proj, self.core)
        return torch.matmul(modulated, self.P_out)

class ProjectedUVLayer(nn.Module):
    def __init__(self, N=16384, K=64, r=32):
        super().__init__()
        self.N = N
        self.K = K
        P_in = torch.randn(N, K) / math.sqrt(N)
        P_out = torch.randn(K, N) / math.sqrt(K)
        self.register_buffer('P_in', P_in)
        self.register_buffer('P_out', P_out)
        self.U = nn.Parameter(torch.randn(K, r) * 0.02)
        self.V = nn.Parameter(torch.randn(r, K) * 0.02)

    def forward(self, x):
        x_proj = torch.matmul(x, self.P_in)
        low_rank = torch.matmul(x_proj, self.U)
        modulated = torch.matmul(low_rank, self.V)
        return torch.matmul(modulated, self.P_out)

class SpatialSliceLayer(nn.Module):
    def __init__(self, N=16384, K=64):
        super().__init__()
        self.N = N
        self.K = K
        self.core = nn.Parameter(torch.randn(K, K) * 0.02)

    def forward(self, x):
        slice_in = x[:, :self.K]
        modulated = torch.matmul(slice_in, self.core)
        out = torch.zeros_like(x)
        out[:, :self.K] = modulated
        return out


def generate_spectral_decay_dataset(num_samples=512, N=16384, alpha=1.5, seed=42):
    """
    Generates synthetic signals with power-law spectral decay in Walsh domain (typical of real-world signals/images).
    Energy is concentrated in low-frequency Walsh spectrum.
    """
    torch.manual_seed(seed)
    # Generate random spectrum with decay 1 / (1 + k)^alpha
    k_idx = torch.arange(1, N + 1, dtype=torch.float32)
    weights = 1.0 / (k_idx ** alpha)
    weights = weights / weights.sum() # Normalize
    
    rand_spec = torch.randn(num_samples, N) * torch.sqrt(weights)
    # Transform to spatial domain to create realistic continuous signals
    signals = ifwht(rand_spec)
    return signals


def run_experiment(exp_name, dataset, N=16384, K=64, epochs=40, lr=5e-3):
    device = torch.device('cpu')
    num_samples = dataset.shape[0]
    batch_size = 32
    num_batches = math.ceil(num_samples / batch_size)
    
    models = [
        ("Spectral Mega-Layer (FWHT)", SpectralMegaLayer),
        ("Projected Dense (Baseline A)", ProjectedDenseLayer),
        ("Projected Low-Rank UV (Baseline B)", ProjectedUVLayer),
        ("Spatial Slicing (Baseline C)", SpatialSliceLayer)
    ]
    
    exp_results = {}
    print(f"\n======================================================================")
    print(f"  EXP: {exp_name} (N={N}, K={K}, Samples={num_samples}, Epochs={epochs})")
    print(f"======================================================================")
    
    for name, cls in models:
        torch.manual_seed(42)
        model = cls(N=N, K=K).to(device)
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
                
                # Fast feedback on first batch of epoch 1
                if epoch == 1 and b == 0:
                    print(f"  [{name[:22]}] Ep 1 Batch 1 Loss: {loss.item():.6f}")
                    
            avg_loss = epoch_loss / num_samples
            loss_history.append(avg_loss)
            
        wall_time = time.time() - t_start_wall
        pei = (1.0 / loss_history[-1]) / math.log10(trainable_params + 1)
        
        exp_results[name] = {
            "model_name": name,
            "trainable_params": trainable_params,
            "final_objective": loss_history[-1],
            "wall_clock_time": wall_time,
            "function_evaluation_time": t_eval_total,
            "internal_overhead_time": wall_time - t_eval_total,
            "PEI": pei,
            "loss_history": loss_history
        }
        print(f"  > {name:<35} | Final Loss: {loss_history[-1]:.6f} | PEI: {pei:.2f}")
        
    return exp_results


def main():
    print("--- V87b: HIGH-DIMENSIONAL LEARNING CAPACITY & SPECTRAL DECAY BENCHMARK ---")
    N = 16384
    K = 64
    
    # Dataset 1: Moderate Spectral Decay (alpha = 1.0)
    ds_mod = generate_spectral_decay_dataset(num_samples=512, N=N, alpha=1.0, seed=42)
    res_mod = run_experiment("Natural Spectral Signal (decay alpha=1.0)", ds_mod, N=N, K=K, epochs=40, lr=1e-2)
    
    # Dataset 2: Strong Spectral Decay (alpha = 2.0, high concentration in low frequencies)
    ds_strong = generate_spectral_decay_dataset(num_samples=512, N=N, alpha=2.0, seed=42)
    res_strong = run_experiment("Highly Structured Spectral Signal (decay alpha=2.0)", ds_strong, N=N, K=K, epochs=40, lr=1e-2)
    
    all_summary = {
        "spectral_decay_alpha_1.0": res_mod,
        "spectral_decay_alpha_2.0": res_strong
    }
    
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v87b_results.json", "w") as f:
        json.dump(all_summary, f, indent=2)
    print("\n[SUCCESS] Execution complete. JSON saved to results/raw/v87b_results.json")

if __name__ == "__main__":
    main()
