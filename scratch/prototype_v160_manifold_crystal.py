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
    """ Walsh-Hadamard Transform 1D batch """
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

def fwht_3d(cube):
    """ Transformada de Walsh en las 3 dimensiones (X, Y, Z) """
    C, H, W, D = cube.shape
    flat_spatial = cube.permute(0, 3, 1, 2).reshape(C*D, H*W)
    w_spatial = fwht(flat_spatial).reshape(C, D, H, W).permute(0, 2, 3, 1)
    flat_depth = w_spatial.reshape(C*H*W, D)
    w_3d = fwht(flat_depth).reshape(C, H, W, D)
    return w_3d

# --- CLUSTERING + ORDENAMIENTO (TSP Greedy) ---
def get_ordered_clanes(feats, k=64):
    """ Encuentra k centros y los ordena por similitud espectral para crear un continuo """
    N, D = feats.shape
    centers = feats[torch.randperm(N)[:k]]
    for _ in range(5):
        sims = torch.mm(feats, centers.t())
        assignments = torch.argmax(sims, dim=1)
        for i in range(k):
            mask = assignments == i
            if mask.any(): 
                centers[i] = F.normalize(feats[mask].mean(0), p=2, dim=0)
    
    ordered_idx = [0]
    remaining = list(range(1, k))
    while remaining:
        last = centers[ordered_idx[-1]]
        sims = torch.mm(last.unsqueeze(0), centers[remaining].t())
        best_rel_idx = torch.argmax(sims).item()
        ordered_idx.append(remaining.pop(best_rel_idx))
    
    return centers[ordered_idx]

def run_experiment():
    print(f"\n--- EXPERIMENTO V160: MANIFOLD SPECTRAL CRYSTAL (3D FLOW) ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    raw_test = test_ds.data.unsqueeze(1).float().to(device) / 255.0
    all_train_targets = train_ds.targets.to(device) # <--- FIX: Mover etiquetas a GPU

    # 2. CONSTRUCCIÓN DEL MANIFOLD
    print("Muestreando el Manifold: Extrayendo y ordenando 64 clanes globales...")
    sample_idx = torch.randperm(60000)[:10000]
    sample_data = train_ds.data[sample_idx].float().to(device) / 255.0
    padded_sample = torch.zeros(10000, 32, 32).to(device)
    padded_sample[:, :28, :28] = sample_data
    w_sample = F.normalize(fwht(padded_sample.reshape(10000, 1024)), p=2, dim=1)
    
    clanes = get_ordered_clanes(w_sample, k=64) # (64, 1024)
    
    # Asignación de etiquetas a clanes
    clan_labels = []
    sims_train = torch.mm(clanes, w_sample.t())
    assignments = torch.argmax(sims_train, dim=0)
    sample_targets = all_train_targets[sample_idx]
    
    for i in range(64):
        mask = assignments == i
        if mask.any():
            targets = sample_targets[mask]
            clan_labels.append(torch.bincount(targets, minlength=10).argmax().item())
        else:
            clan_labels.append(0)
    clan_labels = torch.tensor(clan_labels, device=device)

    # 3. QUEMADO DEL CRISTAL 3D
    print("Tejiendo el Cristal 3D mediante Walsh-Manifold...")
    crystal_cube = clanes.permute(1, 0).reshape(1, 32, 32, 64)
    # crystal_3d = fwht_3d(crystal_cube) # Reservado para uso avanzado en V161
    
    # 4. EVALUACIÓN POR RESONANCIA
    print("Iniciando prueba de resonancia en el Manifold...")
    test_padded = torch.zeros(10000, 32, 32).to(device)
    test_padded[:, :28, :28] = raw_test.squeeze()
    test_walsh = F.normalize(fwht(test_padded.reshape(10000, 1024)), p=2, dim=1)
    
    t0 = time.perf_counter()
    scores = torch.mm(test_walsh, clanes.t())
    preds = clan_labels[torch.argmax(scores, dim=1)]
    acc = (preds == test_ds.targets.to(device)).float().mean().item()
    dt = time.perf_counter() - t0
    
    print("\n" + "="*55)
    print(f"RESULTADO CRISTAL DE MANIFOLD (V160)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Estructura:      Manifold Ordenado (Greedy TSP)")
    print(f"Compresión:      937x (1 cristal vs 60k imágenes)")
    print(f"Tiempo Inferencia: {dt:.4f}s")
    print("="*55)

if __name__ == "__main__":
    run_experiment()
