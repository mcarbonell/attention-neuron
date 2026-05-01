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

# --- CLUSTERING ---
def get_ordered_clanes(feats, k=256):
    N, D = feats.shape
    centers = feats[torch.randperm(N)[:k]]
    for _ in range(8):
        sims = torch.mm(feats, centers.t())
        assignments = torch.argmax(sims, dim=1)
        for i in range(k):
            mask = assignments == i
            if mask.any(): 
                centers[i] = F.normalize(feats[mask].mean(0), p=2, dim=0)
    return centers

def run_experiment():
    print(f"\n--- EXPERIMENTO V168: DYNAMIC MODULATION (SELF-TUNING CRYSTALS) ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    raw_train = train_ds.data.float().to(device) / 255.0
    raw_test = test_ds.data.float().to(device) / 255.0
    targets_train = train_ds.targets.to(device)

    # 2. ENTRENAMIENTO CAPA BASE
    K = 256
    print(f"Paso 1: Entrenando Capa Base ({K} clanes)...")
    sample_idx = torch.randperm(60000)[:15000]
    padded_train = torch.zeros(15000, 32, 32).to(device)
    padded_train[:, :28, :28] = raw_train[sample_idx]
    w_train = F.normalize(fwht(padded_train.reshape(15000, 1024)), p=2, dim=1)
    
    clanes = get_ordered_clanes(w_train, k=K)
    
    # Análisis de fidelidad por clan
    sims_train = torch.mm(w_train, clanes.t())
    assignments = torch.argmax(sims_train, dim=1)
    clan_votes = []
    targets_sample = targets_train[sample_idx].cpu()
    for i in range(K):
        mask = (assignments == i).cpu()
        if mask.any():
            clan_votes.append(torch.bincount(targets_sample[mask], minlength=10).argmax().item())
        else: clan_votes.append(0)
    clan_votes = torch.tensor(clan_votes, device=device)
    
    # FIX: Move to CPU for one_hot to avoid DirectML scatter error
    v_matrix = F.one_hot(clan_votes.cpu(), num_classes=10).float().to(device)

    # 3. SÍNTESIS DE MÁSCARAS DE GANANCIA (MODULACIÓN)
    print("Paso 2: Sintetizando Cables de Modulación (Ecualización por Clase)...")
    gain_masks = torch.ones(10, K).to(device)
    
    for c in range(10):
        # Clanes que pertenecen a la clase c
        correct_clanes_mask = (clan_votes == c).float()
        # Modulación: potenciamos el canal correcto y silenciamos el ruido
        gain_masks[c] = 0.2 + 1.3 * correct_clanes_mask

    # 4. INFERENCIA MODULADA
    print("Paso 3: Inferencia con Modulación Dinámica...")
    test_padded = torch.zeros(10000, 32, 32).to(device)
    test_padded[:, :28, :28] = raw_test
    test_walsh = F.normalize(fwht(test_padded.reshape(10000, 1024)), p=2, dim=1)
    
    t0 = time.perf_counter()
    
    # ETAPA A: Pre-escucha
    sims_raw = torch.mm(test_walsh, clanes.t())
    votes_raw = torch.mm(torch.pow(torch.clamp(sims_raw, min=0.0), 12), v_matrix)
    pre_preds = torch.argmax(votes_raw, dim=1)
    
    # ETAPA B: Modulación de Ganancia
    selected_masks = gain_masks[pre_preds] # (10000, 256)
    modulated_sims = sims_raw * selected_masks
    
    # ETAPA C: Voto Final
    final_votes = torch.mm(torch.pow(torch.clamp(modulated_sims, min=0.0), 12), v_matrix)
    
    preds = torch.argmax(final_votes, dim=1)
    acc = (preds == test_ds.targets.to(device)).float().mean().item()
    dt = time.perf_counter() - t0
    
    print("\n" + "="*55)
    print(f"RESULTADO DYNAMIC MODULATION (V168)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Mecánica:        Modulación de Ganancia (Self-Tuning)")
    print(f"Clanes Base:     {K}")
    print(f"Tiempo Total:    {dt:.4f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v168_dynamic_modulation.json", "w") as f:
        json.dump({"accuracy": acc}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
