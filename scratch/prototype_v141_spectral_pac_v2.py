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

def run_spectral_pac_v2():
    print(f"\n--- EXPERIMENTO V141: SPECTRAL PAC-V2 (CONFUSION BIFURCATION) ---")
    
    # 1. CARGA Y TRANSFORMACIÓN
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

    # 2. INICIALIZACIÓN PAC-V2 (Semantic Mode)
    image_cluster_assignment = train_targets.clone()
    # Guardamos metadatos de cada arquetipo para interpretabilidad
    cluster_info = {d: {"label": d, "confused_with": None, "gen": 0} for d in range(10)}
    next_cluster_id = 10
    
    MAX_GEN = 15
    results_history = []

    print(f"\nIniciando PAC-V2 Espectral (Agrupación por Confusión)...")
    print(f"{'Gen':>3} | {'Archs':>6} | {'Train Acc':>10} | {'Test Acc':>9} | {'Nuevas Conf.'}")
    print("-" * 75)

    for gen in range(MAX_GEN):
        t0 = time.perf_counter()
        
        # A. SÍNTESIS DE ARQUETIPOS
        active_ids = torch.unique(image_cluster_assignment)
        arch_tensors = []
        arch_labels = []
        arch_id_list = []
        
        for cid in active_ids:
            mask = (image_cluster_assignment == cid)
            if mask.sum() > 0:
                arch_tensors.append(spec_train[mask].mean(dim=0))
                arch_labels.append(cluster_info[cid.item()]["label"])
                arch_id_list.append(cid.item())
        
        arch_tensors = torch.stack(arch_tensors)
        arch_labels = torch.tensor(arch_labels, device=device)
        arch_id_list = torch.tensor(arch_id_list, device=device)
        
        # B. EVALUACIÓN HOLOGRÁFICA
        norm_train = F.normalize(spec_train, p=2, dim=1)
        norm_arch = F.normalize(arch_tensors, p=2, dim=1)
        sims = torch.mm(norm_train, norm_arch.t())
        
        max_sim, best_idx = torch.max(sims, dim=1)
        pred_labels = arch_labels[best_idx]
        pred_arch_ids = arch_id_list[best_idx]
        
        correct = (pred_labels == train_targets)
        train_acc = correct.float().mean().item()
        
        # Test Acc
        norm_test = F.normalize(spec_test, p=2, dim=1)
        sim_test = torch.mm(norm_test, norm_arch.t())
        test_acc = (arch_labels[torch.argmax(sim_test, dim=1)] == test_targets).float().mean().item()
        
        dt = time.perf_counter() - t0
        
        # C. BIFURCACIÓN SEMÁNTICA (PAC-V2 logic)
        new_confusions = 0
        errors = ~correct
        if errors.any():
            err_real = train_targets[errors]
            err_pred = pred_labels[errors]
            
            # Identificar pares únicos (Real -> Predicha)
            # Usamos un truco de packing para unique: real * 10 + pred
            packed = err_real * 10 + err_pred
            unique_packed = torch.unique(packed)
            
            for up in unique_packed:
                r, p = up.item() // 10, up.item() % 10
                pair_mask = (train_targets == r) & (pred_labels == p)
                pair_indices = torch.nonzero(pair_mask).squeeze(1)
                
                if len(pair_indices) > 0:
                    # Creamos un nuevo arquetipo para esta confusión específica
                    image_cluster_assignment[pair_indices] = next_cluster_id
                    cluster_info[next_cluster_id] = {"label": r, "confused_with": p, "gen": gen+1}
                    next_cluster_id += 1
                    new_confusions += 1

        print(f"{gen:3d} | {len(active_ids):6d} | {train_acc*100:9.2f}% | {test_acc*100:8.2f}% | +{new_confusions}")
        
        results_history.append({
            "gen": gen, "archetypes": len(active_ids), 
            "train_acc": train_acc, "test_acc": test_acc
        })
        
        if train_acc > 0.999: break

    # D. ANÁLISIS DE LA MEMORIA SEMÁNTICA
    print("\n" + "="*75)
    print("ANÁLISIS DE ARQUETIPOS DE CONFUSIÓN (PAC-V2)")
    print("="*75)
    semantic_archs = [c for i, c in cluster_info.items() if c["confused_with"] is not None and i in active_ids]
    print(f"Total Arquetipos Semánticos: {len(semantic_archs)}")
    
    # Mostrar los arquetipos más 'antiguos' (los primeros errores detectados)
    for i, c in enumerate(semantic_archs[:8]):
        print(f"Tipo {i}: Clase {c['label']} que suena como {c['confused_with']} (Apareció Gen {c['gen']})")
    print("="*75)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v141_spectral_pac_v2.json", "w") as f:
        json.dump(results_history, f, indent=4)

if __name__ == "__main__":
    run_spectral_pac_v2()
