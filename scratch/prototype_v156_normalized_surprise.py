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

# --- EXTRACTOR HÍBRIDO UNITARIO (SIN MEDIA) ---
def get_island_signatures(images):
    binary = (images > 0.1).float()
    padded_h = F.pad(binary, (1, 0), value=0)
    diff_h = padded_h[:, :, :, 1:] - padded_h[:, :, :, :-1]
    islands_h = (diff_h == 1).float().sum(dim=3).squeeze(1)
    padded_v = F.pad(binary, (0, 0, 1, 0), value=0)
    diff_v = padded_v[:, :, 1:, :] - padded_v[:, :, :-1, :]
    islands_v = (diff_v == 1).float().sum(dim=2).squeeze(1)
    return torch.cat([islands_h, islands_v], dim=1)

def get_hybrid_feats_normalized(images_batch, batch_size=5000):
    feats_list = []
    for i in range(0, images_batch.size(0), batch_size):
        batch = images_batch[i : i+batch_size].to(device)
        flat = batch.view(batch.size(0), -1)
        padded = torch.zeros(flat.size(0), 1024).to(device)
        padded[:, :784] = flat
        w = fwht(padded)
        w[:, 0] = 0 # Eliminar DC
        w_norm = F.normalize(w, p=2, dim=1)
        isl = F.normalize(get_island_signatures(batch), p=2, dim=1)
        hybrid = torch.cat([w_norm, isl], dim=1)
        feats_list.append(F.normalize(hybrid, p=2, dim=1).cpu())
    return torch.cat(feats_list, dim=0)

def run_experiment():
    print(f"\n--- EXPERIMENTO V156: FLUID SPECTRAL BRAIN (UPDATES + SPEED) ---")
    
    # 1. CARGA
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=2000)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    raw_test = test_ds.data.unsqueeze(1).float() / 255.0

    # 2. APRENDIZAJE POR SORPRESA FLUIDA
    MAX_SLOTS = 30000 
    memory_bank = torch.zeros(MAX_SLOTS, 1080).to(device)
    memory_targets = torch.zeros(MAX_SLOTS, dtype=torch.long).to(device)
    slots_used = 0
    
    # Hiperparámetros Fluidos
    alpha = 0.2           # Tasa de actualización (EMA)
    sim_threshold = 0.95  # Umbral para creación de nuevos slots
    update_threshold = 0.98 # Umbral para actualizar arquetipo existente
    
    print(f"Iniciando aprendizaje fluido en GPU (Alpha: {alpha})...")
    t0 = time.perf_counter()
    
    processed_count = 0
    for batch_imgs, batch_targets in train_loader:
        feats = get_hybrid_feats_normalized(batch_imgs)
        
        for i in range(feats.size(0)):
            v = feats[i:i+1].to(device)
            label = batch_targets[i].item()
            
            if slots_used == 0:
                memory_bank[0] = v
                memory_targets[0] = label
                slots_used = 1
                continue
            
            # Búsqueda de resonancia sobre el banco activo
            sims = torch.mm(v, memory_bank[:slots_used].t())
            max_sim, idx = torch.max(sims, dim=1)
            max_sim, idx = max_sim.item(), idx.item()
            
            # LÓGICA FLUIDA:
            if max_sim > update_threshold and memory_targets[idx] == label:
                # ACTUALIZACIÓN: El arquetipo absorbe el nuevo conocimiento
                memory_bank[idx] = F.normalize((1-alpha)*memory_bank[idx] + alpha*v.squeeze(), p=2, dim=0)
            elif max_sim < sim_threshold or (max_sim >= sim_threshold and memory_targets[idx] != label):
                # CREACIÓN: Es nuevo o hay conflicto de clase (ambigüedad)
                if slots_used < MAX_SLOTS:
                    memory_bank[slots_used] = v
                    memory_targets[slots_used] = label
                    slots_used += 1
        
        processed_count += batch_imgs.size(0)
        print(f"Procesados {processed_count}/60,000 | Slots Usados: {slots_used}")

    # 3. EVALUACIÓN
    print("\nEvaluando cerebro fluido final...")
    test_feats = get_hybrid_feats_normalized(raw_test)
    
    all_preds = []
    q_batch = 1000
    for i in range(0, 10000, q_batch):
        q = test_feats[i:i+q_batch].to(device)
        sims = torch.mm(q, memory_bank[:slots_used].t())
        idx_match = torch.argmax(sims, dim=1)
        all_preds.append(memory_targets[idx_match])
        
    acc = (torch.cat(all_preds) == test_ds.targets.to(device)).float().mean().item()
    dt = time.perf_counter() - t0
    
    print("\n" + "="*55)
    print(f"RESULTADO CEREBRO FLUIDO (V156)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Capacidad Final: {slots_used} slots")
    print(f"Tiempo Total:    {dt:.2f}s")
    print(f"Compresión:      {60000/slots_used:.1f}x")
    print("="*55)

if __name__ == "__main__":
    run_experiment()
