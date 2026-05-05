import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import time
import os
import json

# --- CONFIGURACIÓN DE DISPOSITIVO ---
device = torch.device('cpu')
try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using DirectML device: {device}")
except ImportError:
    print("torch-directml not found, using CPU")

# --- TRANSFORMADA DE WALSH-HADAMARD RÁPIDA (FWHT) ---
def fwht(x):
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

# --- CLUSTERING VECTORIZADO ---
def get_massive_clanes(feats, k=4096):
    N, D = feats.shape
    centers = feats[torch.randperm(N)[:k]]
    for i in range(3): # Reducimos iteraciones para velocidad en el stack
        chunk_size = 5000
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(k, device=device)
        for j in range(0, N, chunk_size):
            end = min(j + chunk_size, N)
            batch_feats = feats[j:end]
            sims = torch.mm(batch_feats, centers.t())
            assignments = torch.argmax(sims, dim=1)
            new_centers.index_add_(0, assignments, batch_feats)
            counts.index_add_(0, assignments, torch.ones(batch_feats.size(0), device=device))
        centers = F.normalize(new_centers / (counts.unsqueeze(1) + 1e-8), p=2, dim=1)
    return centers

def run_deep_holographic_test():
    print(f"\n--- EXPERIMENTO V164b: DEEP HOLOGRAPHIC STACK (2 LAYERS) ---")
    
    # 1. PREPARACIÓN DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    
    sample_idx = torch.randperm(60000)[:15000]
    sample_data = train_ds.data[sample_idx].float().to(device) / 255.0
    padded_sample = torch.zeros(len(sample_idx), 32, 32).to(device)
    padded_sample[:, :28, :28] = sample_data
    w_sample = F.normalize(fwht(padded_sample.reshape(len(sample_idx), 1024)), p=2, dim=1)
    
    # 2. ENTRENAMIENTO CAPA 1 (GENERAL)
    K = 2048 
    print(f"Entrenando Capa 1: Intuicin ({K} expertos)...")
    clanes_l1 = get_massive_clanes(w_sample, k=K)
    
    # 3. ENTRENAMIENTO CAPA 2 (RESIDUO)
    print("Calculando Residuos Espectrales...")
    sims_l1 = torch.mm(w_sample, clanes_l1.t())
    best_l1 = clanes_l1[torch.argmax(sims_l1, dim=1)]
    residuals = F.normalize(w_sample - best_l1, p=2, dim=1)
    
    print(f"Entrenando Capa 2: Refinamiento ({K} expertos de error)...")
    clanes_l2 = get_massive_clanes(residuals, k=K)
    
    # 4. INFERENCIA CON CONTEXTO
    print("Evaluando Deep Stack con Contexto Hologrfico...")
    test_raw = test_ds.data.unsqueeze(1).float().to(device) / 255.0
    test_padded = torch.zeros(10000, 32, 32).to(device)
    test_padded[:, :28, :28] = test_raw.squeeze()
    test_walsh = F.normalize(fwht(test_padded.reshape(10000, 1024)), p=2, dim=1)
    
    # Simulamos acumulador hologrfico
    hologram = F.normalize(test_walsh + torch.roll(test_walsh, shifts=1, dims=1), p=2, dim=1)

    t0 = time.perf_counter()
    
    # --- CAPA 1 ---
    h1 = torch.mm(hologram, clanes_l1.t())
    # --- RESIDUO ---
    best_l1_idx = torch.argmax(h1, dim=1)
    res_test = F.normalize(hologram - clanes_l1[best_l1_idx], p=2, dim=1)
    # --- CAPA 2 ---
    h2 = torch.mm(res_test, clanes_l2.t())
    
    dt = time.perf_counter() - t0
    
    print("\n" + "="*60)
    print(f"RESULTADO DEEP HOLOGRAPHIC STACK (V164b)")
    print(f"="*60)
    print(f"Capas Apiladas:     2")
    print(f"Expertos Totales:   {K*2:,}")
    print(f"Latencia Deep Step: {dt*1000:.2f} ms")
    print(f"Sparsity por Capa:  {1/K*100:.4f}%")
    print("="*60)

    # Guardar resultados
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v164b_deep_holographic.json", "w") as f:
        json.dump({
            "layers": 2,
            "latency_ms": dt * 1000,
            "experts_per_layer": K
        }, f, indent=4)

if __name__ == "__main__":
    run_deep_holographic_test()
