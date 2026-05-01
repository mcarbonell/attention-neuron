import torch
import torch.nn as nn
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
    """ Fast Walsh-Hadamard Transform vectorizada """
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

def run_holographic_pac():
    print(f"\n--- EXPERIMENTO V140: HOLOGRAPHIC-PAC (SPECTRAL ARCHETYPES) ---")
    
    # 1. PREPARACIÓN DE DATOS
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    print("Cargando MNIST y transformando al dominio de Walsh...")
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=60000)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=10000)
    
    train_data, train_targets = next(iter(train_loader))
    test_data, test_targets = next(iter(test_loader))
    
    def to_walsh(data):
        flat = data.view(data.size(0), -1)
        padded = torch.zeros(flat.size(0), 1024).to(device)
        padded[:, :784] = flat.to(device)
        return fwht(padded)

    spec_train = to_walsh(train_data)
    spec_test = to_walsh(test_data)
    train_targets = train_targets.to(device)
    test_targets = test_targets.to(device)

    # 2. INICIALIZACIÓN DEL ALGORITMO PAC (Bifurcation Mode)
    image_cluster_assignment = train_targets.clone()
    cluster_to_label = {d: d for d in range(10)}
    next_cluster_id = 10
    
    MAX_GEN = 12 # Limitado para evitar explosión de clusters en benchmark rápido
    results_history = []

    print(f"\nIniciando purificación espectral...")
    print(f"{'Gen':>3} | {'Archs':>6} | {'Train Acc':>10} | {'Test Acc':>9} | {'Time':>7}")
    print("-" * 60)

    for gen in range(MAX_GEN):
        t0 = time.perf_counter()
        
        # A. SÍNTESIS DE ARQUETIPOS (Promedio de clusters en el dominio Walsh)
        active_clusters = torch.unique(image_cluster_assignment)
        arch_tensors = []
        arch_labels = []
        arch_ids = []
        
        for cid in active_clusters:
            mask = (image_cluster_assignment == cid)
            if mask.sum() > 0:
                # El arquetipo es el promedio espectral puro del cluster
                arch_tensors.append(spec_train[mask].mean(dim=0))
                arch_labels.append(cluster_to_label[cid.item()])
                arch_ids.append(cid.item())
        
        arch_tensors = torch.stack(arch_tensors)
        arch_labels = torch.tensor(arch_labels, device=device)
        arch_ids = torch.tensor(arch_ids, device=device)
        
        # B. CLASIFICACIÓN HOLOGRÁFICA (Similitud Coseno en Walsh Space)
        norm_train = F.normalize(spec_train, p=2, dim=1)
        norm_arch = F.normalize(arch_tensors, p=2, dim=1)
        similarities = torch.mm(norm_train, norm_arch.t())
        
        max_sim, best_arch_idx = torch.max(similarities, dim=1)
        pred_labels = arch_labels[best_arch_idx]
        
        correct = (pred_labels == train_targets)
        train_acc = correct.float().mean().item()
        
        # Validación en Test
        norm_test = F.normalize(spec_test, p=2, dim=1)
        sim_test = torch.mm(norm_test, norm_arch.t())
        test_acc = (arch_labels[torch.argmax(sim_test, dim=1)] == test_targets).float().mean().item()
        
        dt = time.perf_counter() - t0
        print(f"{gen:3d} | {len(active_clusters):6d} | {train_acc*100:9.2f}% | {test_acc*100:8.2f}% | {dt:6.2f}s")
        
        results_history.append({
            "gen": gen, 
            "num_archetypes": len(active_clusters), 
            "train_acc": train_acc, 
            "test_acc": test_acc
        })
        
        if train_acc > 0.999: break # Casi perfecto
            
        # C. BIFURCACIÓN DE ERRORES (Crecimiento dinámico de la memoria)
        # Los correctos refuerzan su arquetipo (se reasignan al más cercano de su clase)
        image_cluster_assignment[correct] = arch_ids[best_arch_idx[correct]]
        
        # Los errores engendran nuevos arquetipos para capturar la anomalía
        new_assignments = image_cluster_assignment.clone()
        for digit in range(10):
            mask_err = (~correct) & (train_targets == digit)
            err_indices = torch.nonzero(mask_err).squeeze(1)
            
            if len(err_indices) > 0:
                # Dividimos los errores en dos para acelerar la convergencia (Bifurcación)
                half = len(err_indices) // 2
                if half > 0:
                    new_assignments[err_indices[:half]] = next_cluster_id
                    cluster_to_label[next_cluster_id] = digit
                    next_cluster_id += 1
                    new_assignments[err_indices[half:]] = next_cluster_id
                    cluster_to_label[next_cluster_id] = digit
                    next_cluster_id += 1
                else:
                    new_assignments[err_indices] = next_cluster_id
                    cluster_to_label[next_cluster_id] = digit
                    next_cluster_id += 1
        
        image_cluster_assignment = new_assignments

    print("\n" + "="*60)
    print(f"RESULTADOS FINALES: HOLOGRAPHIC-PAC")
    print(f"="*60)
    print(f"Arquetipos Finales: {len(active_clusters)}")
    print(f"Precisión Test: {test_acc*100:.2f}%")
    print(f"Eficiencia de Memoria: {60000/len(active_clusters):.1f}x compresión")
    print(f"Estatus: Purificación Espectral Completada")
    print("="*60)

    # Guardar resultados
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v140_holographic_pac.json", "w") as f:
        json.dump(results_history, f, indent=4)

if __name__ == "__main__":
    run_holographic_pac()
