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
def get_ordered_clanes(feats, k=128):
    """ Encuentra k centros representativos """
    N, D = feats.shape
    if N < k: k = N
    centers = feats[torch.randperm(N)[:k]]
    for _ in range(8):
        sims = torch.mm(feats, centers.t())
        assignments = torch.argmax(sims, dim=1)
        for i in range(k):
            mask = assignments == i
            if mask.any(): 
                centers[i] = F.normalize(feats[mask].mean(0), p=2, dim=0)
    return centers

def get_v_matrix(clanes, feats, targets):
    """ Genera la matriz de etiquetas (One-Hot) para un cristal """
    K = clanes.size(0)
    sims = torch.mm(feats, clanes.t())
    assignments = torch.argmax(sims, dim=1)
    
    clan_labels = []
    targets_cpu = targets.cpu()
    for i in range(K):
        mask = (assignments == i).cpu()
        if mask.any():
            clan_labels.append(torch.bincount(targets_cpu[mask], minlength=10).argmax().item())
        else:
            clan_labels.append(0)
    
    labels_cpu = torch.tensor(clan_labels, dtype=torch.long)
    return F.one_hot(labels_cpu, num_classes=10).float().to(device)

def run_experiment():
    print(f"\n--- EXPERIMENTO V167: PROGRESSIVE NEUROGENESIS (RESIDUAL CORRECTION) ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    raw_train = train_ds.data.float().to(device) / 255.0
    raw_test = test_ds.data.float().to(device) / 255.0
    targets_train = train_ds.targets.to(device)

    # 2. ENTRENAMIENTO CAPA BASE (INTUICIÓN)
    K_BASE = 256
    print(f"Paso 1: Entrenando Capa Base ({K_BASE} clanes generales)...")
    sample_idx = torch.randperm(60000)[:15000]
    padded_train = torch.zeros(15000, 32, 32).to(device)
    padded_train[:, :28, :28] = raw_train[sample_idx]
    w_train = F.normalize(fwht(padded_train.reshape(15000, 1024)), p=2, dim=1)
    
    clanes_base = get_ordered_clanes(w_train, k=K_BASE)
    v_base = get_v_matrix(clanes_base, w_train, targets_train[sample_idx])

    # 3. AUDITORÍA DE ERRORES (DIAGNÓSTICO)
    print("Paso 2: Auditando fallos de la Capa Base...")
    # Evaluamos toda la base de entrenamiento para encontrar los "puntos ciegos"
    # (Dividimos en batches para evitar OOM)
    errors_data = []
    errors_targets = []
    
    batch_size = 5000
    for i in range(0, 60000, batch_size):
        b_data = raw_train[i:i+batch_size]
        b_padded = torch.zeros(len(b_data), 32, 32).to(device)
        b_padded[:, :28, :28] = b_data
        b_w = F.normalize(fwht(b_padded.reshape(-1, 1024)), p=2, dim=1)
        
        # Predicción Base
        sims = torch.mm(b_w, clanes_base.t())
        votes = torch.mm(torch.pow(torch.clamp(sims, min=0.0), 12), v_base)
        preds = torch.argmax(votes, dim=1)
        
        # Guardamos solo los errores
        mask_err = preds != targets_train[i:i+batch_size]
        if mask_err.any():
            errors_data.append(b_w[mask_err])
            errors_targets.append(targets_train[i:i+batch_size][mask_err])
            
    all_errors_w = torch.cat(errors_data, dim=0)
    all_errors_t = torch.cat(errors_targets, dim=0)
    print(f"Detectados {len(all_errors_w)} errores recurrentes.")

    # 4. NEUROGÉNESIS (INYECCIÓN DE ESPECIALISTAS)
    K_SPECIALIST = 256
    print(f"Paso 3: Neurogénesis de Capa 2 ({K_SPECIALIST} especialistas en errores)...")
    # Creamos clanes SOLO para las formas que fallaron
    clanes_specialist = get_ordered_clanes(all_errors_w, k=K_SPECIALIST)
    v_specialist = get_v_matrix(clanes_specialist, all_errors_w, all_errors_t)

    # 5. INFERENCIA PROGRESIVA
    print("Paso 4: Evaluando el Test Set con Refuerzo de Especialistas...")
    test_padded = torch.zeros(10000, 32, 32).to(device)
    test_padded[:, :28, :28] = raw_test
    test_walsh = F.normalize(fwht(test_padded.reshape(10000, 1024)), p=2, dim=1)
    
    t0 = time.perf_counter()
    
    # --- Resonancia Base ---
    sims_base = torch.mm(test_walsh, clanes_base.t())
    votes_base = torch.mm(torch.pow(torch.clamp(sims_base, min=0.0), 12), v_base)
    
    # --- Resonancia Especialista ---
    sims_spec = torch.mm(test_walsh, clanes_specialist.t())
    # El especialista solo vota si tiene una resonancia fuerte (foco en el error)
    votes_spec = torch.mm(torch.pow(torch.clamp(sims_spec, min=0.0), 12), v_specialist)
    
    # Fusión Dinámica: El especialista refuerza donde la base duda
    final_votes = votes_base + 0.8 * votes_spec
    
    preds = torch.argmax(final_votes, dim=1)
    acc = (preds == test_ds.targets.to(device)).float().mean().item()
    dt = time.perf_counter() - t0
    
    print("\n" + "="*55)
    print(f"RESULTADO NEUROGÉNESIS PROGRESIVA (V167)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Capa 1:          256 clanes generales")
    print(f"Capa 2 (New):    256 especialistas en errores")
    print(f"Compresión:      117x")
    print(f"Tiempo Total:    {dt:.4f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v167_progressive_neurogenesis.json", "w") as f:
        json.dump({"accuracy": acc}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
