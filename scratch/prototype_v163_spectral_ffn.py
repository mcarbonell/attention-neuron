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
    for _ in range(10): # Más refinamiento para 256 clanes
        sims = torch.mm(feats, centers.t())
        assignments = torch.argmax(sims, dim=1)
        for i in range(k):
            mask = assignments == i
            if mask.any(): 
                centers[i] = F.normalize(feats[mask].mean(0), p=2, dim=0)
    
    # Ordenamiento TSP Greedy
    ordered_idx = [0]
    remaining = list(range(1, k))
    while remaining:
        last = centers[ordered_idx[-1]]
        sims = torch.mm(last.unsqueeze(0), centers[remaining].t())
        best_rel_idx = torch.argmax(sims).item()
        ordered_idx.append(remaining.pop(best_rel_idx))
    
    return centers[ordered_idx]

def run_experiment():
    print(f"\n--- EXPERIMENTO V163: SPECTRAL-FFN (HOLOGRAPHIC TRANSFORMER BLOCK) ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    raw_test = test_ds.data.unsqueeze(1).float().to(device) / 255.0
    all_train_targets = train_ds.targets.to(device)

    # 2. CONSTRUCCIÓN DEL CRISTAL DE ALTA DENSIDAD (256 Canales)
    K_FFN = 256
    print(f"Sintonizando FFN Espectral: {K_FFN} clanes globales...")
    sample_idx = torch.randperm(60000)[:20000]
    sample_data = train_ds.data[sample_idx].float().to(device) / 255.0
    padded_sample = torch.zeros(20000, 32, 32).to(device)
    padded_sample[:, :28, :28] = sample_data
    w_sample = F.normalize(fwht(padded_sample.reshape(20000, 1024)), p=2, dim=1)
    
    clanes = get_ordered_clanes(w_sample, k=K_FFN) # (256, 1024)
    
    # Matriz de Síntesis (V-Matrix del Transformer)
    print("Sintetizando Matriz de Salida (Down-Projection)...")
    sims_train = torch.mm(clanes, w_sample.t())
    assignments = torch.argmax(sims_train, dim=0)
    sample_targets = all_train_targets[sample_idx]
    
    clan_labels_list = []
    for i in range(K_FFN):
        mask = assignments == i
        if mask.any():
            targets = sample_targets[mask]
            clan_labels_list.append(torch.bincount(targets.cpu(), minlength=10).argmax().item())
        else:
            clan_labels_list.append(0)
            
    # Matrix de Síntesis en formato One-Hot (Down-Projection)
    clan_labels_cpu = torch.tensor(clan_labels_list, dtype=torch.long)
    v_matrix = F.one_hot(clan_labels_cpu, num_classes=10).float().to(device) # (256, 10)

    # 3. EJECUCIÓN DEL BLOQUE SPECTRAL-FFN
    print("Evaluando Bloque Spectral-FFN en el Test Set...")
    test_padded = torch.zeros(10000, 32, 32).to(device)
    test_padded[:, :28, :28] = raw_test.squeeze()
    test_walsh = F.normalize(fwht(test_padded.reshape(10000, 1024)), p=2, dim=1)
    
    t0 = time.perf_counter()
    
    # PASO 1: Up-Projection (Resonancia con el Manifold)
    # X (1024) @ W_up (1024, 256) -> H (256)
    h = torch.mm(test_walsh, clanes.t())
    
    # PASO 2: Activación No Lineal (Potencia de Atención Hopfield)
    # H_act = H^P
    power = 16 # Mayor potencia para compensar la mayor densidad de clanes
    h_act = torch.pow(torch.clamp(h, min=0.0), power)
    
    # PASO 3: Down-Projection (Síntesis Holográfica)
    # H_act (256) @ W_down (256, 10) -> Y (10)
    output = torch.mm(h_act, v_matrix)
    
    preds = torch.argmax(output, dim=1)
    acc = (preds == test_ds.targets.to(device)).float().mean().item()
    dt = time.perf_counter() - t0
    
    print("\n" + "="*55)
    print(f"RESULTADO SPECTRAL-FFN (V163)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Dimensión Oculta: {K_FFN} clanes")
    print(f"Parámetros:      {K_FFN * 1024 + K_FFN * 10} (Compresión 234x)")
    print(f"Tiempo Inferencia: {dt:.4f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v163_spectral_ffn.json", "w") as f:
        json.dump({"accuracy": acc, "hidden_dim": K_FFN}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
