import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import json
import os
import numpy as np

# --- CONFIGURACIÓN DE DISPOSITIVO ---
# Priorizamos CPU para consistencia en benchmarks de latencia, 
# pero permitimos DirectML si está disponible para ver el escalado de VRAM.
device = torch.device('cpu')
HAS_DIRECTML = False
try:
    import torch_directml
    device = torch_directml.device()
    HAS_DIRECTML = True
    print(f"Using DirectML device: {device}")
except ImportError:
    print("torch-directml not found, using CPU")

# --- UTILIDADES ESPECTRALES ---
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

# --- CAPAS A COMPARAR ---

class SmoothWalshLayer(nn.Module):
    def __init__(self, in_features, out_features, N=32, K=8, use_cache=False):
        super().__init__()
        self.in_features = in_features 
        self.out_features = out_features
        self.N, self.K = N, K
        self.use_cache = use_cache
        self.spectral_core = nn.Parameter(torch.randn(out_features, K, K) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('H_K', get_walsh_matrix_sequency(K))
        self._cached_w = None

    def get_weights(self):
        if self.use_cache and self._cached_w is not None:
            return self._cached_w
            
        w_mini = torch.matmul(self.H_K.t(), torch.matmul(self.spectral_core, self.H_K)) / (self.K * self.K)
        w_mini_4d = w_mini.unsqueeze(1) 
        w_smooth_4d = F.interpolate(w_mini_4d, size=(self.N, self.N), mode='bilinear', align_corners=False)
        w_final = w_smooth_4d.view(self.out_features, -1)
        
        if self.use_cache:
            self._cached_w = w_final
        return w_final

    def forward(self, x):
        w = self.get_weights()
        return F.linear(x, w, self.bias)

# --- ENGINE DE BENCHMARK ---

def benchmark_step(model, optimizer, x, target, iters=50):
    model.train()
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        if hasattr(model, '_cached_w'): model._cached_w = None
        loss = F.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()

    def sync():
        if HAS_DIRECTML: pass

    t_fwd, t_bwd, t_opt = 0, 0, 0
    
    for _ in range(iters):
        # Limpiar caché si existe para medir el coste de síntesis real por paso
        if hasattr(model, '_cached_w'): model._cached_w = None

        # FORWARD
        optimizer.zero_grad()
        sync()
        t0 = time.perf_counter()
        pred = model(x)
        loss = F.mse_loss(pred, target)
        sync()
        t_fwd += (time.perf_counter() - t0)
        
        # BACKWARD
        sync()
        t0 = time.perf_counter()
        loss.backward()
        sync()
        t_bwd += (time.perf_counter() - t0)
        
        # OPTIMIZER STEP
        sync()
        t0 = time.perf_counter()
        optimizer.step()
        sync()
        t_opt += (time.perf_counter() - t0)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params": params,
        "fwd_ms": (t_fwd / iters) * 1000,
        "bwd_ms": (t_bwd / iters) * 1000,
        "opt_ms": (t_opt / iters) * 1000,
        "total_step_ms": ((t_fwd + t_bwd + t_opt) / iters) * 1000
    }

def run_saturation_test():
    in_dim = 1024 
    hidden_dims = [128, 1024, 4096, 8192] # Reducido para brevedad
    batch_size = 64
    
    results = []
    dev_name = "gpu" if HAS_DIRECTML else "cpu"
    
    print("\n" + "="*95)
    print(f"BENCHMARK SATURACIÓN EN {dev_name.upper()}")
    print(f"{'DIM':>6} | {'MODEL':>15} | {'PARAMS':>12} | {'FWD(ms)':>8} | {'OPT(ms)':>8} | {'TOTAL(ms)':>10}")
    print("="*95)

    for h_dim in hidden_dims:
        models = [
            ("Dense", nn.Linear(in_dim, h_dim).to(device)),
            ("SmoothWalsh", SmoothWalshLayer(in_dim, h_dim, use_cache=False).to(device)),
            ("SW_Cached", SmoothWalshLayer(in_dim, h_dim, use_cache=True).to(device))
        ]

        for name, model in models:
            opt = optim.Adam(model.parameters(), lr=0.001)
            x = torch.randn(batch_size, in_dim).to(device)
            target = torch.randn(batch_size, h_dim).to(device)
            
            res = benchmark_step(model, opt, x, target)
            res["mode"] = name
            res["h_dim"] = h_dim
            res["device"] = dev_name
            results.append(res)
            
            print(f"{h_dim:6} | {name:>15} | {res['params']:12,} | {res['fwd_ms']:8.2f} | {res['opt_ms']:8.2f} | {res['total_step_ms']:10.2f}")
        print("-" * 95)

    # Guardar resultados con sufijo de dispositivo
    os.makedirs("results/raw", exist_ok=True)
    filename = f"results/raw/v136_saturation_{dev_name}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResultados guardados en {filename}")
if __name__ == "__main__":
    run_saturation_test()
