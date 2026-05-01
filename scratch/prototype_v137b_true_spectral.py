import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import os
import numpy as np

# --- CONFIGURACIÓN DE DISPOSITIVO ---
device = torch.device('cpu')
HAS_DIRECTML = False
try:
    import torch_directml
    device = torch_directml.device()
    HAS_DIRECTML = True
    print(f"Using DirectML device: {device}")
except ImportError:
    print("torch-directml not found, using CPU")

import traceback

# --- TRANSFORMADA DE WALSH-HADAMARD RÁPIDA (FWHT) ---
def fwht(x):
    """ 
    Versión optimizada para evitar stack/view excesivos que confunden a DirectML.
    """
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        # Reestructuramos para operar en pares
        res = res.view(b, n // (2 * h), 2, h)
        # Operación in-place para ahorrar memoria y evitar fragmentación
        a = res[:, :, 0, :]
        b_ = res[:, :, 1, :]
        # Usamos una forma más directa de asignar
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

class TrueSpectralLayer(nn.Module):
    def __init__(self, dim, K=64):
        super().__init__()
        self.dim = dim
        self.K = K
        self.spectral_weights = nn.Parameter(torch.randn(dim, K) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x_spec = fwht(x)
        x_low = x_spec[:, :self.K]
        return torch.matmul(x_low, self.spectral_weights.t()) + self.bias

def run_true_spectral_benchmark():
    dims = [4096, 16384, 32768, 65536, 131072] 
    batch_size = 8
    
    print("\n" + "="*95)
    print(f"BENCHMARK: TRUE SPECTRAL V2 (RESISTENCIA EXTREMA)")
    print(f"{'DIM':>7} | {'MODEL':>10} | {'EST. MEM':>12} | {'PARAMS':>12} | {'STEP(ms)':>10}")
    print("="*95)

    for d in dims:
        # Asegurar potencia de 2
        d = 2**int(np.ceil(np.log2(d)))
        
        # TEST DENSE
        try:
            mem_dense = (d * d * 3 * 4) / (1024**2)
            print(f"{d:7} | {'Dense':>10} | {mem_dense:9.1f} MB | ", end="", flush=True)
            
            if mem_dense > 12000: raise MemoryError("Preventive OOM")
            
            model = nn.Linear(d, d).to(device)
            optimizer = optim.Adam(model.parameters())
            x = torch.randn(batch_size, d).to(device)
            
            t0 = time.perf_counter()
            model(x).sum().backward()
            optimizer.step()
            print(f"{d*d:12,} | {(time.perf_counter()-t0)*1000:10.2f} ms")
            del model, optimizer, x
        except:
            print(f"{'FAILED':>12} | OOM/Limit")

        # TEST TRUE SPECTRAL
        try:
            K = 64
            mem_spec = (d * K * 3 * 4) / (1024**2)
            print(f"{d:7} | {'Spectral':>10} | {mem_spec:9.1f} MB | ", end="", flush=True)
            
            model = TrueSpectralLayer(d, K=K).to(device)
            optimizer = optim.Adam(model.parameters())
            x = torch.randn(batch_size, d).to(device)
            
            t0 = time.perf_counter()
            model(x).sum().backward()
            optimizer.step()
            dt = (time.perf_counter() - t0) * 1000
            print(f"{d*K:12,} | {dt:10.2f} ms")
            del model, optimizer, x
        except Exception as e:
            err_msg = str(e)
            if not err_msg: err_msg = "Unknown Error (Check Console)"
            print(f"{'FAILED':>12} | {err_msg[:20]}")
            # traceback.print_exc() # Descomentar si sigue fallando sin mensaje
        
        print("-" * 95)

if __name__ == "__main__":
    run_true_spectral_benchmark()
