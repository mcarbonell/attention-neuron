import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

# --- CAPAS ---

class SpectralSynthesisLayer(nn.Module):
    """
    Capa espectral diseñada para dimensiones extremas.
    En lugar de sintetizar la matriz completa (que causaría OOM),
    sintetiza solo lo necesario para el cálculo o usa una representación comprimida.
    """
    def __init__(self, in_dim, out_dim, K=16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K
        # Parámetros: Solo el núcleo espectral (proporcional a out_dim, no a in*out)
        self.spectral_core = nn.Parameter(torch.randn(out_dim, K, K) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
    def forward(self, x):
        # Para el benchmark de 'estrés', simulamos el coste de reconstrucción
        # de una forma que sea justa: tomamos el core y lo expandimos linealmente 
        # hasta el ancho de entrada.
        # Esto evita el OOM de una matriz gigante pero mide el coste de computar con pocos parámetros.
        
        # Simulación de expansión espectral 1D (más eficiente que bilinear 2D para dims gigantes)
        w_spectral = self.spectral_core.view(self.out_features if hasattr(self, 'out_features') else self.out_dim, -1)
        # Repetimos o expandimos el core para llenar in_dim
        repeats = (self.in_dim // (self.K * self.K)) + 1
        w_expanded = w_spectral.repeat(1, repeats)[:, :self.in_dim]
        
        return F.linear(x, w_expanded, self.bias)

def run_extreme_benchmark():
    # Dimensiones que llevan al límite a un MLP denso (16k x 16k = 268M params)
    # 268M params * 4 bytes/param * 3 (Adam) = ~3.2 GB de memoria de estados.
    # A 32k x 32k, sube a 12.8 GB.
    dims = [4096, 8192, 16384, 32768, 49152]
    batch_size = 8 # Batch pequeño para enfocarnos en la memoria de los pesos
    
    results = []
    
    print("\n" + "="*85)
    print(f"EXPLORANDO EL LÍMITE: TERRENO IMPOSIBLE (D_in = D_out)")
    print(f"{'DIM':>6} | {'MODEL':>10} | {'EST. MEM':>12} | {'PARAMS':>12} | {'STEP(ms)':>10}")
    print("="*85)

    for d in dims:
        # --- TEST 1: DENSE ---
        try:
            # Estimación memoria: (W + m + v) * 4 bytes
            mem_dense_mb = (d * d * 3 * 4) / (1024**2)
            print(f"{d:6} | {'Dense':>10} | {mem_dense_mb:9.1f} MB | ", end="", flush=True)
            
            t0 = time.perf_counter()
            model = nn.Linear(d, d).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            x = torch.randn(batch_size, d).to(device)
            target = torch.randn(batch_size, d).to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            
            dt = (time.perf_counter() - t0) * 1000
            params = d * d
            print(f"{params:12,} | {dt:10.2f} ms")
            
            results.append({"dim": d, "model": "dense", "params": params, "time_ms": dt, "mem_mb": mem_dense_mb, "status": "ok"})
            
            del model, optimizer, x, target
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{'FAILED':>12} | OOM/Error")
            results.append({"dim": d, "model": "dense", "status": "failed", "error": str(e)})
            # Si el denso falla, dejamos de probar el denso pero seguimos con el espectral
            pass

        # --- TEST 2: SPECTRAL ---
        try:
            K = 16
            mem_spec_mb = (d * K * K * 3 * 4) / (1024**2)
            print(f"{d:6} | {'Spectral':>10} | {mem_spec_mb:9.1f} MB | ", end="", flush=True)
            
            t0 = time.perf_counter()
            model = SpectralSynthesisLayer(d, d, K=K).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            x = torch.randn(batch_size, d).to(device)
            target = torch.randn(batch_size, d).to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            
            dt = (time.perf_counter() - t0) * 1000
            params = d * K * K
            print(f"{params:12,} | {dt:10.2f} ms")
            
            results.append({"dim": d, "model": "spectral", "params": params, "time_ms": dt, "mem_mb": mem_spec_mb, "status": "ok"})
            
            del model, optimizer, x, target
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{'FAILED':>12} | OOM/Error")
            results.append({"dim": d, "model": "spectral", "status": "failed", "error": str(e)})

        print("-" * 85)

    # Guardar resultados
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v137_extreme_scaling.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_extreme_benchmark()
