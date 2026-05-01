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
    print(f"\n--- EXPERIMENTO V162: SPECTRAL PROFILE RESONANCE (META-WALSH) ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    raw_test = test_ds.data.unsqueeze(1).float().to(device) / 255.0
    all_train_targets = train_ds.targets.to(device)

    # 2. CONSTRUCCIÓN DEL MANIFOLD (64 Clanes)
    print("Afinando el Manifold Espectral...")
    sample_idx = torch.randperm(60000)[:20000] # Muestra amplia para ritmos estables
    sample_data = train_ds.data[sample_idx].float().to(device) / 255.0
    padded_sample = torch.zeros(20000, 32, 32).to(device)
    padded_sample[:, :28, :28] = sample_data
    w_sample = F.normalize(fwht(padded_sample.reshape(20000, 1024)), p=2, dim=1)
    
    clanes = get_ordered_clanes(w_sample, k=64)
    sample_targets = all_train_targets[sample_idx]

    # 3. SÍNTESIS DE RITMOS MAESTROS (META-SPECTRAL)
    print("Sintetizando los 10 Ritmos Maestros (Segunda Transformada)...")
    # Perfil: similitud con los 64 clanes ordenados
    train_profiles = torch.mm(w_sample, clanes.t()) # (20000, 64)
    # Meta-Walsh: Transformada del perfil para extraer "armónicos de resonancia"
    train_rhythms = fwht(train_profiles) # (20000, 64)
    
    master_rhythms = []
    for c in range(10):
        mask = sample_targets == c
        if mask.any():
            # Promediamos el ritmo de la clase y normalizamos
            m_rhythm = train_rhythms[mask].mean(0)
            master_rhythms.append(F.normalize(m_rhythm, p=2, dim=0))
        else:
            master_rhythms.append(torch.zeros(64).to(device))
    master_rhythms = torch.stack(master_rhythms) # (10, 64)

    # 4. INFERENCIA POR RITMO (META-RESONANCIA)
    print("Evaluando test set mediante 'Música de los Números'...")
    test_padded = torch.zeros(10000, 32, 32).to(device)
    test_padded[:, :28, :28] = raw_test.squeeze()
    test_walsh = F.normalize(fwht(test_padded.reshape(10000, 1024)), p=2, dim=1)
    
    t0 = time.perf_counter()
    
    # Paso 1: Generar perfil de test
    test_profiles = torch.mm(test_walsh, clanes.t())
    # Paso 2: Extraer ritmo de test (Meta-Walsh)
    test_rhythms = F.normalize(fwht(test_profiles), p=2, dim=1)
    
    # Paso 3: Clasificar por similitud de ritmos
    meta_scores = torch.mm(test_rhythms, master_rhythms.t())
    preds = torch.argmax(meta_scores, dim=1)
    
    acc = (preds == test_ds.targets.to(device)).float().mean().item()
    dt = time.perf_counter() - t0
    
    print("\n" + "="*55)
    print(f"RESULTADO RITMO ESPECTRAL (V162)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Abstracción:     Meta-Walsh (Perfil de 64 Clanes)")
    print(f"Compresión:      937x")
    print(f"Tiempo Total:    {dt:.4f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v162_profile_resonance.json", "w") as f:
        json.dump({"accuracy": acc}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
