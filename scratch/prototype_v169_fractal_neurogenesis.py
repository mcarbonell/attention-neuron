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
    K = clanes.size(0)
    sims = torch.mm(feats, clanes.t())
    assignments = torch.argmax(sims, dim=1)
    clan_labels = []
    targets_cpu = targets.cpu()
    for i in range(K):
        mask = (assignments == i).cpu()
        if mask.any():
            clan_labels.append(torch.bincount(targets_cpu[mask], minlength=10).argmax().item())
        else: clan_labels.append(0)
    labels_cpu = torch.tensor(clan_labels, dtype=torch.long)
    return F.one_hot(labels_cpu, num_classes=10).float().to(device)

def get_errors(w_data, targets, clanes_list, v_list):
    """ Encuentra los errores acumulados de todas las capas actuales """
    total_votes = torch.zeros(len(w_data), 10).to(device)
    batch_size = 5000
    for i in range(0, len(w_data), batch_size):
        b_w = w_data[i:i+batch_size]
        b_votes = torch.zeros(len(b_w), 10).to(device)
        for idx, (clanes, v_mat) in enumerate(zip(clanes_list, v_list)):
            sims = torch.mm(b_w, clanes.t())
            power = 12 + (idx * 4)
            votes = torch.mm(torch.pow(torch.clamp(sims, min=0.0), power), v_mat)
            b_votes += votes
        total_votes[i:i+batch_size] = b_votes
    
    preds = torch.argmax(total_votes, dim=1)
    mask_err = preds != targets
    return w_data[mask_err], targets[mask_err]

def run_experiment():
    print(f"\n--- EXPERIMENTO V169: FRACTAL NEUROGENESIS (3-LAYER CASCADE) ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    raw_train = train_ds.data.float().to(device) / 255.0
    raw_test = test_ds.data.float().to(device) / 255.0
    targets_train = train_ds.targets.to(device)
    
    print("Transformando Dataset de Entrenamiento...")
    padded_all = torch.zeros(60000, 32, 32).to(device)
    padded_all[:, :28, :28] = raw_train
    w_train_all = F.normalize(fwht(padded_all.reshape(60000, 1024)), p=2, dim=1)

    all_clanes = []
    all_v_matrices = []

    # PASO 1: CAPA BASE
    print("Capa 1: Formando la base (256 clanes)...")
    indices_l1 = torch.randperm(60000)[:15000] # FIX: Índices fijos
    clanes_l1 = get_ordered_clanes(w_train_all[indices_l1], k=256)
    v_l1 = get_v_matrix(clanes_l1, w_train_all[indices_l1], targets_train[indices_l1])
    all_clanes.append(clanes_l1)
    all_v_matrices.append(v_l1)

    # PASO 2: CAPA DE ERRORES L1
    print("Capa 2: Neurogénesis de primer nivel...")
    err_w, err_t = get_errors(w_train_all, targets_train, all_clanes, all_v_matrices)
    print(f"Auditando L1: {len(err_w)} errores detectados.")
    
    # Entrenamos la Capa 2 con los errores de L1
    clanes_l2 = get_ordered_clanes(err_w, k=256)
    v_l2 = get_v_matrix(clanes_l2, err_w, err_t)
    all_clanes.append(clanes_l2)
    all_v_matrices.append(v_l2)

    # PASO 3: CAPA DE SUPER-ERRORES
    print("Capa 3: Neurogénesis de segundo nivel...")
    err_w2, err_t2 = get_errors(w_train_all, targets_train, all_clanes, all_v_matrices)
    print(f"Auditando L1+L2: {len(err_w2)} errores persistentes.")
    
    clanes_l3 = get_ordered_clanes(err_w2, k=256)
    v_l3 = get_v_matrix(clanes_l3, err_w2, err_t2)
    all_clanes.append(clanes_l3)
    all_v_matrices.append(v_l3)

    # 4. INFERENCIA TRIPLE
    print("Iniciando Inferencia Fractal sobre el Test Set...")
    test_padded = torch.zeros(10000, 32, 32).to(device)
    test_padded[:, :28, :28] = raw_test
    test_walsh = F.normalize(fwht(test_padded.reshape(10000, 1024)), p=2, dim=1)
    
    t0 = time.perf_counter()
    
    final_votes = torch.zeros(10000, 10).to(device)
    for idx, (clanes, v_mat) in enumerate(zip(all_clanes, all_v_matrices)):
        sims = torch.mm(test_walsh, clanes.t())
        power = 12 + (idx * 4) 
        votes = torch.mm(torch.pow(torch.clamp(sims, min=0.0), power), v_mat)
        final_votes += votes
    
    preds = torch.argmax(final_votes, dim=1)
    acc = (preds == test_ds.targets.to(device)).float().mean().item()
    dt = time.perf_counter() - t0
    
    print("\n" + "="*55)
    print(f"RESULTADO FRACTAL NEUROGENESIS (V169)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Capas Activas:   3")
    print(f"Compresión:      78x (768 clanes)")
    print(f"Tiempo Inferencia: {dt:.4f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v169_fractal_neurogenesis.json", "w") as f:
        json.dump({"accuracy": acc}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
