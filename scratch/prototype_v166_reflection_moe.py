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
    print(f"\n--- EXPERIMENTO V166: REFLECTION MoE (ANALYSIS-BY-SYNTHESIS) ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    raw_test = test_ds.data.unsqueeze(1).float().to(device) / 255.0

    # 2. ENTRENAMIENTO DEL MoE BASE (Igual que V165)
    print("Entrenando Router Espectral (128 clanes)...")
    sample_idx = torch.randperm(60000)[:15000]
    sample_data = train_ds.data[sample_idx].float().to(device) / 255.0
    padded_sample = torch.zeros(15000, 32, 32).to(device)
    padded_sample[:, :28, :28] = sample_data
    w_sample = F.normalize(fwht(padded_sample.reshape(15000, 1024)), p=2, dim=1)
    
    router_clanes = get_ordered_clanes(w_sample, k=128)
    sims_router = torch.mm(w_sample, router_clanes.t())
    assignments = torch.argmax(sims_router, dim=1)
    router_labels = []
    targets_sample = train_ds.targets[sample_idx].cpu()
    for i in range(128):
        mask = assignments.cpu() == i
        if mask.any():
            router_labels.append(torch.bincount(targets_sample[mask], minlength=10).argmax().item())
        else: router_labels.append(0)
    router_labels = torch.tensor(router_labels, device=device)

    print("Entrenando 10 Especialistas (32 clanes c/u)...")
    experts = []
    for c in range(10):
        idx_c = (train_ds.targets == c).nonzero()[:1000].squeeze()
        data_c = train_ds.data[idx_c].float().to(device) / 255.0
        padded_c = torch.zeros(len(idx_c), 32, 32).to(device)
        padded_c[:, :28, :28] = data_c
        w_c = F.normalize(fwht(padded_c.reshape(len(idx_c), 1024)), p=2, dim=1)
        experts.append(get_ordered_clanes(w_c, k=32))
    experts_tensor = torch.stack(experts) # (10, 32, 1024)

    # 3. INFERENCIA CON REFLEXIÓN (ANÁLISIS POR SÍNTESIS)
    print("Iniciando Inferencia con Reflexión Generativa...")
    test_padded = torch.zeros(10000, 32, 32).to(device)
    test_padded[:, :28, :28] = raw_test.squeeze()
    test_walsh = F.normalize(fwht(test_padded.reshape(10000, 1024)), p=2, dim=1)
    
    t0 = time.perf_counter()
    
    # PASO 1: Intuición del Router
    router_sims = torch.mm(test_walsh, router_clanes.t())
    # Gating energético
    router_onehot = F.one_hot(router_labels.cpu(), num_classes=10).float().to(device)
    router_class_energy = torch.mm(torch.pow(torch.clamp(router_sims, min=0.0), 8), router_onehot)
    
    # PASO 2: Reflexión de Especialistas
    final_expert_scores = []
    for c in range(10):
        # A. Resonancia inicial
        sims_e = torch.mm(test_walsh, experts_tensor[c].t()) # (10000, 32)
        
        # B. Generación del "Sueño" (Reconstrucción)
        # Usamos pesos Softmax sobre la resonancia para crear la imagen idealizada
        weights = F.softmax(sims_e * 20.0, dim=1) # (10000, 32)
        dream = torch.mm(weights, experts_tensor[c]) # (10000, 1024)
        
        # C. Cotejo de Fidelidad (¿Se parece mi sueño a la realidad?)
        # Calculamos la similitud coseno entre el test original y el sueño del experto
        fidelity = torch.sum(test_walsh * dream, dim=1) # (10000)
        
        # D. Score final del experto = Resonancia Max * Fidelidad
        r_max = torch.max(sims_e, dim=1)[0]
        score_e = r_max * torch.pow(torch.clamp(fidelity, min=0.0), 4) # Penalizamos fuertemente la infidelidad
        final_expert_scores.append(score_e)
        
    expert_matrix = torch.stack(final_expert_scores, dim=1) # (10000, 10)
    
    # PASO 3: Fusión Final
    final_result = expert_matrix * router_class_energy
    
    preds = torch.argmax(final_result, dim=1)
    acc = (preds == test_ds.targets.to(device)).float().mean().item()
    dt = time.perf_counter() - t0
    
    print("\n" + "="*55)
    print(f"RESULTADO REFLECTION MoE (V166)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Arquitectura:    Gating + Specialists + Synthesis")
    print(f"Mecánica:        Análisis por Síntesis (Fidelidad)")
    print(f"Tiempo Total:    {dt:.4f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v166_reflection_moe.json", "w") as f:
        json.dump({"accuracy": acc}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
