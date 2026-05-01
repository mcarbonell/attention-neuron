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

# --- NORMALIZADOR ROBUSTO (CPU) ---
def standardize_robust_cpu(images, target_size=20, target_mass=100.0):
    B, C, H, W = images.shape
    imgs_in = images.cpu()
    std_images = torch.zeros(B, 1, H, W)
    for i in range(B):
        img = imgs_in[i, 0]
        mask = img > 0.05
        coords = torch.nonzero(mask)
        if coords.size(0) < 5:
            std_images[i] = imgs_in[i]
            continue
        y_min, x_min = coords[:, 0].min().item(), coords[:, 1].min().item()
        y_max, x_max = coords[:, 0].max().item(), coords[:, 1].max().item()
        h_crop, w_crop = y_max - y_min + 1, x_max - x_min + 1
        scale = target_size / max(h_crop, w_crop)
        new_h, new_w = int(h_crop * scale), int(w_crop * scale)
        new_h, new_w = max(1, new_h), max(1, new_w)
        crop = img[y_min:y_max+1, x_min:x_max+1].reshape(1, 1, h_crop, w_crop)
        resized = F.interpolate(crop, size=(new_h, new_w), mode='bilinear', align_corners=False)
        mass = resized.sum()
        if mass > 0: resized = resized * (target_mass / mass)
        sy, sx = (H - new_h) // 2, (W - new_w) // 2
        std_images[i, 0, sy:sy+new_h, sx:sx+new_w] = resized.reshape(new_h, new_w)
    return std_images

# --- EXTRACTOR DE ISLAS ---
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
    print(f"\n--- EXPERIMENTO V150: DUAL HOLOGRAPHIC MEMORY (ORGANIC + STANDARD) ---")
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    
    raw_train = train_ds.data.unsqueeze(1).float() / 255.0
    raw_test = test_ds.data.unsqueeze(1).float() / 255.0
    
    print("Preparando dataset dual (CPU Robust Standardization)...")
    std_train = standardize_robust_cpu(raw_train)
    std_test = standardize_robust_cpu(raw_test)

    print("\nInyectando 120,000 recuerdos híbridos (1080D)...")
    t0 = time.perf_counter()
    bank_org = get_hybrid_feats(raw_train)
    bank_std = get_hybrid_feats(std_train)
    full_bank = torch.cat([bank_org, bank_std], dim=0)
    full_targets = torch.cat([train_ds.targets, train_ds.targets])

    print("Ejecutando búsqueda asociativa dual (Batched Search)...")
    query_org = get_hybrid_feats(raw_test)
    query_std = get_hybrid_feats(std_test)
    
    full_bank_norm = F.normalize(full_bank.to(device), p=2, dim=1)
    
    all_preds = []
    q_batch_size = 1000 # Lotes pequeños para evitar OOM en DirectML
    
    for i in range(0, query_org.size(0), q_batch_size):
        q_org_b = query_org[i : i + q_batch_size].to(device)
        q_std_b = query_std[i : i + q_batch_size].to(device)
        
        # Similitudes cruzadas
        sims_org = torch.mm(q_org_b, full_bank_norm.t())
        sims_std = torch.mm(q_std_b, full_bank_norm.t())
        
        # Máxima resonancia para cada query en el lote
        combined = torch.max(sims_org, sims_std)
        indices = torch.argmax(combined, dim=1)
        all_preds.append(full_targets[indices.cpu()])
        
        if (i + q_batch_size) % 5000 == 0:
            print(f"Procesadas {i + q_batch_size}/10,000 queries...")

    predictions = torch.cat(all_preds)
    acc = (predictions == test_ds.targets).float().mean().item()
    dt = time.perf_counter() - t0
    
    print("\n" + "="*55)
    print(f"RESULTADO MEMORIA DUAL (V150)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Capacidad:       120,000 / 131,072 slots")
    print(f"Tiempo Total:    {dt:.2f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v150_dual_memory.json", "w") as f:
        json.dump({"accuracy": acc, "time": dt}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
