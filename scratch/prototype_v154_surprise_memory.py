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

# --- EXTRACTOR HÍBRIDO ---
def get_island_signatures(images):
    binary = (images > 0.1).float()
    padded_h = F.pad(binary, (1, 0), value=0)
    diff_h = padded_h[:, :, :, 1:] - padded_h[:, :, :, :-1]
    islands_h = (diff_h == 1).float().sum(dim=3).squeeze(1)
    padded_v = F.pad(binary, (0, 0, 1, 0), value=0)
    diff_v = padded_v[:, :, 1:, :] - padded_v[:, :, :-1, :]
    islands_v = (diff_v == 1).float().sum(dim=2).squeeze(1)
    return torch.cat([islands_h, islands_v], dim=1)

def get_hybrid_feats(images_batch, batch_size=5000):
    feats_list = []
    for i in range(0, images_batch.size(0), batch_size):
        batch = images_batch[i : i+batch_size].to(device)
        flat = batch.view(batch.size(0), -1)
        padded = torch.zeros(flat.size(0), 1024).to(device)
        padded[:, :784] = flat
        w = F.normalize(fwht(padded), p=2, dim=1)
        isl = F.normalize(get_island_signatures(batch), p=2, dim=1)
        feats_list.append(torch.cat([w, isl], dim=1).cpu())
    return torch.cat(feats_list, dim=0)

def run_experiment():
    print(f"\n--- EXPERIMENTO V154: SURPRISE-DRIVEN MEMORY (SPECTRAL NOVELTY) ---")
    
    # 1. CARGA
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=2000)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    raw_test = test_ds.data.unsqueeze(1).float() / 255.0

    # 2. APRENDIZAJE POR SORPRESA
    surprise_threshold = 0.965 # Umbral de novedad (Similitud Coseno)
    memory_bank = []
    memory_targets = []
    
    print(f"Construyendo memoria selectiva (Umbral Sorpresa: {surprise_threshold})...")
    t0 = time.perf_counter()
    
    processed_count = 0
    for batch_imgs, batch_targets in train_loader:
        # Calculamos firmas del lote
        feats = get_hybrid_feats(batch_imgs)
        
        for i in range(feats.size(0)):
            v = feats[i:i+1].to(device)
            label = batch_targets[i].item()
            
            if len(memory_bank) == 0:
                memory_bank.append(v.cpu())
                memory_targets.append(label)
                continue
            
            # ¿Qué tan familiar es esta imagen?
            # Comparamos contra la memoria acumulada (Optimizamos usando mm masivo)
            # Para no ralentizar, comparamos el vector v contra todo el banco actual
            current_bank = torch.cat(memory_bank, dim=0).to(device)
            sims = torch.mm(v, current_bank.t())
            max_sim = torch.max(sims).item()
            
            if max_sim < surprise_threshold:
                # ¡SORPRESA! Este es un nuevo arquetipo natural
                memory_bank.append(v.cpu())
                memory_targets.append(label)
        
        processed_count += batch_imgs.size(0)
        if processed_count % 10000 == 0:
            print(f"Procesados {processed_count}/60,000 | Slots Usados: {len(memory_bank)}")

    memory_bank = torch.cat(memory_bank, dim=0).to(device)
    memory_targets = torch.tensor(memory_targets, device=device)
    
    # 3. EVALUACIÓN
    print("\nEvaluando resonancia sobre memoria selectiva...")
    test_feats = get_hybrid_feats(raw_test)
    
    all_preds = []
    q_batch = 1000
    for i in range(0, 10000, q_batch):
        q = test_feats[i:i+q_batch].to(device)
        sims = torch.mm(q, memory_bank.t())
        idx = torch.argmax(sims, dim=1)
        all_preds.append(memory_targets[idx])
        
    acc = (torch.cat(all_preds) == test_ds.targets.to(device)).float().mean().item()
    dt = time.perf_counter() - t0
    
    print("\n" + "="*55)
    print(f"RESULTADO MEMORIA POR SORPRESA (V154)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Capacidad Usada: {len(memory_bank)} slots")
    print(f"Ratio Compresión: {60000/len(memory_bank):.1f}x")
    print(f"Tiempo Total:    {dt:.2f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v154_surprise_memory.json", "w") as f:
        json.dump({"accuracy": acc, "slots": len(memory_bank)}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
