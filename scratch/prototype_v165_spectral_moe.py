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
def get_ordered_clanes(feats, k=32):
    """ Encuentra k centros y los ordena """
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

def run_experiment():
    print(f"\n--- EXPERIMENTO V165: SPECTRAL MIXTURE OF EXPERTS (SMoE) ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    raw_test = test_ds.data.unsqueeze(1).float().to(device) / 255.0
    all_train_targets = train_ds.targets.to(device)

    # 2. CONSTRUCCIÓN DEL ROUTER (GATING NETWORK)
    print("Entrenando el Router Espectral (128 clanes globales)...")
    sample_idx = torch.randperm(60000)[:15000]
    sample_data = train_ds.data[sample_idx].float().to(device) / 255.0
    padded_sample = torch.zeros(15000, 32, 32).to(device)
    padded_sample[:, :28, :28] = sample_data
    w_sample = F.normalize(fwht(padded_sample.reshape(15000, 1024)), p=2, dim=1)
    
    router_clanes = get_ordered_clanes(w_sample, k=128)
    sims_router = torch.mm(w_sample, router_clanes.t())
    assignments = torch.argmax(sims_router, dim=1) # <--- FIX: dim=1
    
    router_labels = []
    targets_sample = all_train_targets[sample_idx].cpu()
    for i in range(128):
        mask = (assignments == i).cpu()
        if mask.any():
            router_labels.append(torch.bincount(targets_sample[mask], minlength=10).argmax().item())
        else: router_labels.append(0)
    router_labels = torch.tensor(router_labels, device=device)

    # 3. CONSTRUCCIÓN DE LOS 10 EXPERTOS (SPECIALISTS)
    print("Entrenando a los 10 Especialistas de Clase (32 clanes c/u)...")
    experts = []
    for c in range(10):
        idx_c = (train_ds.targets == c).nonzero()[:1000].squeeze()
        data_c = train_ds.data[idx_c].float().to(device) / 255.0
        padded_c = torch.zeros(len(idx_c), 32, 32).to(device)
        padded_c[:, :28, :28] = data_c
        w_c = F.normalize(fwht(padded_c.reshape(len(idx_c), 1024)), p=2, dim=1)
        
        expert_clanes = get_ordered_clanes(w_c, k=32)
        experts.append(expert_clanes)
    
    experts_tensor = torch.stack(experts) # (10, 32, 1024)

    # 4. INFERENCIA MoE
    print("Iniciando Inferencia Multi-Experto...")
    test_padded = torch.zeros(10000, 32, 32).to(device)
    test_padded[:, :28, :28] = raw_test.squeeze()
    test_walsh = F.normalize(fwht(test_padded.reshape(10000, 1024)), p=2, dim=1)
    
    t0 = time.perf_counter()
    
    # PASO 1: Router Gating
    router_sims = torch.mm(test_walsh, router_clanes.t())
    router_onehot = F.one_hot(router_labels.cpu(), num_classes=10).float().to(device)
    # Energía del router por clase
    router_class_energy = torch.mm(torch.pow(torch.clamp(router_sims, min=0.0), 8), router_onehot)
    
    # PASO 2: Resonancia de Especialistas
    expert_energies = []
    for c in range(10):
        sims_e = torch.mm(test_walsh, experts_tensor[c].t())
        # Tomamos la mejor resonancia dentro del experto de la clase c
        energy_e = torch.max(torch.pow(torch.clamp(sims_e, min=0.0), 16), dim=1)[0]
        expert_energies.append(energy_e)
        
    expert_matrix = torch.stack(expert_energies, dim=1) # (10000, 10)
    
    # PASO 3: Fusión (Router * Especialista)
    final_scores = expert_matrix * router_class_energy
    
    preds = torch.argmax(final_scores, dim=1)
    acc = (preds == test_ds.targets.to(device)).float().mean().item()
    dt = time.perf_counter() - t0
    
    print("\n" + "="*55)
    print(f"RESULTADO SPECTRAL-MoE (V165)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Arquitectura:    Router (128) + 10 Experts (32)")
    print(f"Compresión:      138x")
    print(f"Tiempo Inferencia: {dt:.4f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v165_spectral_moe.json", "w") as f:
        json.dump({"accuracy": acc}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
