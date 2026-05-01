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

# --- CLUSTERING + ORDENAMIENTO (TSP Greedy) ---
def get_ordered_clanes(feats, k=64):
    """ Encuentra k centros y los ordena por similitud espectral """
    N, D = feats.shape
    centers = feats[torch.randperm(N)[:k]]
    for _ in range(8):
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
    print(f"\n--- EXPERIMENTO V161: SPECTRAL HOPFIELD RESONANCE ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    raw_test = test_ds.data.unsqueeze(1).float().to(device) / 255.0
    all_train_targets = train_ds.targets.to(device)

    # 2. CONSTRUCCIÓN DEL MANIFOLD (64 Clanes)
    print("Muestreando el Manifold: Extrayendo y ordenando 64 clanes globales...")
    sample_idx = torch.randperm(60000)[:15000]
    sample_data = train_ds.data[sample_idx].float().to(device) / 255.0
    padded_sample = torch.zeros(15000, 32, 32).to(device)
    padded_sample[:, :28, :28] = sample_data
    w_sample = F.normalize(fwht(padded_sample.reshape(15000, 1024)), p=2, dim=1)
    
    clanes = get_ordered_clanes(w_sample, k=64) # (64, 1024)
    
    # Etiquetas de clanes (Voto mayoritario)
    clan_labels_list = []
    sims_train = torch.mm(clanes, w_sample.t())
    assignments = torch.argmax(sims_train, dim=0)
    sample_targets = all_train_targets[sample_idx]
    
    for i in range(64):
        mask = assignments == i
        if mask.any():
            targets = sample_targets[mask]
            clan_labels_list.append(torch.bincount(targets.cpu(), minlength=10).argmax().item())
        else:
            clan_labels_list.append(0)
            
    # FIX: One-hot en CPU para evitar error de DirectML
    clan_labels_cpu = torch.tensor(clan_labels_list, dtype=torch.long)
    clan_onehot = F.one_hot(clan_labels_cpu, num_classes=10).float().to(device)

    # 3. RESONANCIA DE HOPFIELD (NO LINEAL)
    print("Iniciando Inferencia por Resonancia No Lineal (Potencia S^12)...")
    test_padded = torch.zeros(10000, 32, 32).to(device)
    test_padded[:, :28, :28] = raw_test.squeeze()
    test_walsh = F.normalize(fwht(test_padded.reshape(10000, 1024)), p=2, dim=1)
    
    t0 = time.perf_counter()
    
    # Paso 1: Similitud Coseno Lineal
    sims = torch.mm(test_walsh, clanes.t()) # (10000, 64)
    
    # Paso 2: Amplificación No Lineal (Filtro de Atención)
    power = 12
    # Clamp para evitar gradientes negativos o inestabilidad si hubiera
    resonance_energy = torch.pow(torch.clamp(sims, min=0.0), power)
    
    # Paso 3: Reflexión sobre el espacio de etiquetas (Votación Ponderada)
    final_resonance = torch.mm(resonance_energy, clan_onehot)
    
    preds = torch.argmax(final_resonance, dim=1)
    acc = (preds == test_ds.targets.to(device)).float().mean().item()
    dt = time.perf_counter() - t0
    
    print("\n" + "="*55)
    print(f"RESULTADO HOPFIELD ESPECTRAL (V161)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Mecánica:        Resonancia No Lineal (S^{power})")
    print(f"Compresión:      937x (1 cristal compartido)")
    print(f"Tiempo Total:    {dt:.4f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v161_spectral_hopfield.json", "w") as f:
        json.dump({"accuracy": acc, "power": power}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
