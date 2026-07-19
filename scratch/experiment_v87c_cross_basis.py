import torch
import torch.nn as nn
import torch.optim as optim
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
    return C # Orthogonal matrix C^T C = I

def dct(x, C):
    return torch.matmul(x, C.T)

def idct(x, C):
    return torch.matmul(x, C)

# --- Architectures (all iso-parameter 4,096 trainable params) ---

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

class PCADenseLayer(nn.Module):
    def __init__(self, dataset, N=16384, K=64):
        super().__init__()
        self.N = N
        self.K = K
        
        # Perform SVD/PCA on dataset to get optimal top-K principal directions
        # dataset: (S, N)
        print(f"  [PCA-Init] Computing SVD on dataset {dataset.shape} for top {K} components...")
        _, _, Vh = torch.linalg.svd(dataset, full_matrices=False)
        # Vh: (K_svd, N). Top K right singular vectors:
        P_pca_in = Vh[:K, :].T # (N, K)
        P_pca_out = Vh[:K, :] # (K, N)
        
        self.register_buffer('P_in', P_pca_in)
        self.register_buffer('P_out', P_pca_out)
        self.core = nn.Parameter(torch.randn(K, K) * 0.02)

    def forward(self, x):
        x_proj = torch.matmul(x, self.P_in)
        modulated = torch.matmul(x_proj, self.core)
        return torch.matmul(modulated, self.P_out)

class ProjectedDenseLayer(nn.Module):
    def __init__(self, N=16384, K=64):
        super().__init__()
        self.N = N
        self.K = K
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


# --- Dataset Generators for Cross-Basis Evaluation ---

def generate_walsh_decay_dataset(num_samples=512, N=16384, alpha=2.0, seed=42):
    torch.manual_seed(seed)
    k_idx = torch.arange(1, N + 1, dtype=torch.float32)
    weights = 1.0 / (k_idx ** alpha)
    weights = weights / weights.sum()
    rand_spec = torch.randn(num_samples, N) * torch.sqrt(weights)
    return ifwht(rand_spec)

def generate_dct_decay_dataset(num_samples=512, N=16384, alpha=2.0, seed=42):
    torch.manual_seed(seed)
    C = create_dct_matrix(N)
    k_idx = torch.arange(1, N + 1, dtype=torch.float32)
    weights = 1.0 / (k_idx ** alpha)
    weights = weights / weights.sum()
    rand_spec = torch.randn(num_samples, N) * torch.sqrt(weights)
    return idct(rand_spec, C)

def generate_random_ortho_decay_dataset(num_samples=512, N=16384, alpha=2.0, seed=42):
    torch.manual_seed(seed)
    # Generate random orthogonal matrix Q via QR factorization
    M = torch.randn(N, N)
    Q, _ = torch.linalg.qr(M)
    
    k_idx = torch.arange(1, N + 1, dtype=torch.float32)
    weights = 1.0 / (k_idx ** alpha)
    weights = weights / weights.sum()
    rand_spec = torch.randn(num_samples, N) * torch.sqrt(weights)
    return torch.matmul(rand_spec, Q.T)


def run_cross_basis_benchmark(dataset_name, dataset, N=16384, K=64, epochs=40, lr=1e-2):
    device = torch.device('cpu')
    num_samples = dataset.shape[0]
    batch_size = 32
    num_batches = math.ceil(num_samples / batch_size)
    
    models = [
        ("Spectral Mega-Layer (FWHT)", lambda ds: SpectralMegaLayer(N=N, K=K)),
        ("PCA-Informed Dense (PCA Baseline)", lambda ds: PCADenseLayer(ds, N=N, K=K)),
        ("Projected Dense (Random Ortho)", lambda ds: ProjectedDenseLayer(N=N, K=K)),
        ("Projected Low-Rank UV (Random Ortho)", lambda ds: ProjectedUVLayer(N=N, K=K))
    ]
    
    print(f"\n" + "="*80)
    print(f"  CROSS-BASIS BENCHMARK: Data generated in [{dataset_name}] (N={N}, K={K}, Samples={num_samples})")
    print("="*80)
    
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
                
                # Fast feedback rule
                if epoch == 1 and b == 0:
                    print(f"  [{name[:32]:<32}] Ep 1 Batch 1 Loss: {loss.item():.6f}")
                    
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
        print(f"  > {name:<36} | Loss: {loss_history[-1]:.8f} | PEI: {pei:.2f}")
        
    return results


def main():
    print("--- V87c: CROSS-BASIS EXPERIMENT & PCA OPTIMAL SUBSPACE BENCHMARK ---")
    N = 16384
    K = 64
    EPOCHS = 40
    SAMPLES = 512
    ALPHA = 2.0
    
    # 1. Dataset 1: Walsh Domain Decay
    print("\n[1/3] Generating Walsh Decay Dataset...")
    ds_walsh = generate_walsh_decay_dataset(num_samples=SAMPLES, N=N, alpha=ALPHA, seed=42)
    res_walsh = run_cross_basis_benchmark("Walsh Basis", ds_walsh, N=N, K=K, epochs=EPOCHS)
    
    # 2. Dataset 2: DCT Domain Decay (Non-native for FWHT!)
    print("\n[2/3] Generating DCT Decay Dataset...")
    ds_dct = generate_dct_decay_dataset(num_samples=SAMPLES, N=N, alpha=ALPHA, seed=42)
    res_dct = run_cross_basis_benchmark("DCT Basis", ds_dct, N=N, K=K, epochs=EPOCHS)
    
    # 3. Dataset 3: Random Orthogonal Domain Decay
    print("\n[3/3] Generating Random Orthogonal Decay Dataset...")
    ds_ortho = generate_random_ortho_decay_dataset(num_samples=SAMPLES, N=N, alpha=ALPHA, seed=42)
    res_ortho = run_cross_basis_benchmark("Random Orthogonal Basis", ds_ortho, N=N, K=K, epochs=EPOCHS)
    
    summary = {
        "walsh_basis_data": res_walsh,
        "dct_basis_data": res_dct,
        "random_ortho_basis_data": res_ortho
    }
    
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v87c_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[SUCCESS] Cross-Basis Benchmark complete! Results written to results/raw/v87c_results.json")

if __name__ == "__main__":
    main()
