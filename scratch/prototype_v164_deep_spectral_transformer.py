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
def get_ordered_clanes(feats, k=256):
    """ Encuentra k centros y los ordena para crear un manifold de alta densidad """
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

def get_v_matrix(clanes, feats, targets):
    """ Genera la matriz de síntesis (Down-Projection) para un cristal """
    K = clanes.size(0)
    sims = torch.mm(feats, clanes.t())
    assignments = torch.argmax(sims, dim=1)
    
    clan_labels = []
    for i in range(K):
        mask = assignments == i
        if mask.any():
            clan_labels.append(torch.bincount(targets[mask].cpu(), minlength=10).argmax().item())
        else:
            clan_labels.append(0)
    
    labels_cpu = torch.tensor(clan_labels, dtype=torch.long)
    return F.one_hot(labels_cpu, num_classes=10).float().to(device)

def run_experiment():
    print(f"\n--- EXPERIMENTO V164: DEEP SPECTRAL TRANSFORMER (2-LAYER REFINE) ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    raw_test = test_ds.data.unsqueeze(1).float().to(device) / 255.0
    all_train_targets = train_ds.targets.to(device)

    # 2. ENTRENAMIENTO CAPA 1 (GENERAL)
    K_DIM = 256
    print(f"Capa 1: Construyendo el Cristal de Intuición ({K_DIM} clanes)...")
    sample_idx = torch.randperm(60000)[:15000]
    sample_data = train_ds.data[sample_idx].float().to(device) / 255.0
    padded_sample = torch.zeros(15000, 32, 32).to(device)
    padded_sample[:, :28, :28] = sample_data
    w_sample = F.normalize(fwht(padded_sample.reshape(15000, 1024)), p=2, dim=1)
    
    clanes_l1 = get_ordered_clanes(w_sample, k=K_DIM)
    v_matrix_l1 = get_v_matrix(clanes_l1, w_sample, all_train_targets[sample_idx])

    # 3. CÁLCULO DE RESIDUOS (EL ERROR DE COMPRENSIÓN)
    print("Calculando Residuos Espectrales para el Refinamiento...")
    # Reconstrucción L1: Q_hat = (Q @ K.T) @ K
    # (Usamos similitud coseno para la energía de reconstrucción)
    energies = torch.mm(w_sample, clanes_l1.t())
    # Top-1 reconstrucción para el residuo más puro
    best_clan_idx = torch.argmax(energies, dim=1)
    reconstruction = clanes_l1[best_clan_idx]
    
    residuals = w_sample - reconstruction
    residuals = F.normalize(residuals, p=2, dim=1) # Normalizamos el error para que sea una firma
    
    # 4. ENTRENAMIENTO CAPA 2 (ESPECIALISTA EN DETALLE)
    print(f"Capa 2: Construyendo el Cristal de Detalle ({K_DIM} clanes de error)...")
    clanes_l2 = get_ordered_clanes(residuals, k=K_DIM)
    v_matrix_l2 = get_v_matrix(clanes_l2, residuals, all_train_targets[sample_idx])

    # 5. INFERENCIA PROFUNDA (CASCADA)
    print("Iniciando Inferencia Multicapa sobre el Test Set...")
    test_padded = torch.zeros(10000, 32, 32).to(device)
    test_padded[:, :28, :28] = raw_test.squeeze()
    test_walsh = F.normalize(fwht(test_padded.reshape(10000, 1024)), p=2, dim=1)
    
    t0 = time.perf_counter()
    
    # --- CAPA 1 ---
    h1 = torch.mm(test_walsh, clanes_l1.t())
    h1_act = torch.pow(torch.clamp(h1, min=0.0), 16)
    votes_l1 = torch.mm(h1_act, v_matrix_l1)
    
    # --- RESIDUO ---
    best_l1_idx = torch.argmax(h1, dim=1)
    recon_l1 = clanes_l1[best_l1_idx]
    test_residuals = F.normalize(test_walsh - recon_l1, p=2, dim=1)
    
    # --- CAPA 2 ---
    h2 = torch.mm(test_residuals, clanes_l2.t())
    h2_act = torch.pow(torch.clamp(h2, min=0.0), 16)
    votes_l2 = torch.mm(h2_act, v_matrix_l2)
    
    # --- FUSIÓN (Intuición + Detalle) ---
    # Damos un peso ligeramente menor al detalle para no sobre-corregir
    final_output = votes_l1 + 0.5 * votes_l2
    
    preds = torch.argmax(final_output, dim=1)
    acc = (preds == test_ds.targets.to(device)).float().mean().item()
    dt = time.perf_counter() - t0
    
    print("\n" + "="*55)
    print(f"RESULTADO DEEP SPECTRAL TRANSFORMER (V164)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Capas:           2 (Intuición + Detalle)")
    print(f"Canales Totales: {K_DIM * 2} clanes")
    print(f"Tiempo Total:    {dt:.4f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v164_deep_spectral.json", "w") as f:
        json.dump({"accuracy": acc, "layers": 2}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
