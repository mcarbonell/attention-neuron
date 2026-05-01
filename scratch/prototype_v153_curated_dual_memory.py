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

# --- EXTRACTOR HÍBRIDO (WALSH + ISLAS) ---
def get_hybrid_feats(images_batch, batch_size=5000):
    feats_list = []
    for i in range(0, images_batch.size(0), batch_size):
        batch = images_batch[i : i+batch_size].to(device)
        flat = batch.view(batch.size(0), -1)
        padded = torch.zeros(flat.size(0), 1024).to(device)
        padded[:, :784] = flat
        w = F.normalize(fwht(padded), p=2, dim=1)
        binary = (batch > 0.1).float()
        padded_h = F.pad(binary, (1, 0), value=0)
        diff_h = padded_h[:, :, :, 1:] - padded_h[:, :, :, :-1]
        islands_h = (diff_h == 1).float().sum(dim=3).squeeze(1)
        padded_v = F.pad(binary, (0, 0, 1, 0), value=0)
        diff_v = padded_v[:, :, 1:, :] - padded_v[:, :, :-1, :]
        islands_v = (diff_v == 1).float().sum(dim=2).squeeze(1)
        isl = F.normalize(torch.cat([islands_h, islands_v], dim=1), p=2, dim=1)
        feats_list.append(torch.cat([w, isl], dim=1).cpu())
    return torch.cat(feats_list, dim=0)

def run_experiment():
    print(f"\n--- EXPERIMENTO V153: CURATED DUAL MEMORY (DATA PURIFICATION) ---")
    
    # 1. CARGA
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    raw_train = train_ds.data.unsqueeze(1).float() / 255.0
    raw_test = test_ds.data.unsqueeze(1).float() / 255.0

    # 2. AUDITORÍA ESPECTRAL (LIMPIEZA DE MISLABELS)
    print("Identificando anomalías espectrales para purificar el banco de memoria...")
    t_audit = time.perf_counter()
    flat_train = raw_train.view(60000, -1)
    padded_train = torch.zeros(60000, 1024).to(device)
    padded_train[:, :784] = flat_train.to(device)
    spec_bank = F.normalize(fwht(padded_train), p=2, dim=1)
    
    mislabels_mask = torch.zeros(60000, dtype=torch.bool)
    batch_audit = 1000 # Reducido a 1k para evitar OOM/TDR en DirectML
    for i in range(0, 60000, batch_audit):
        sims = torch.mm(spec_bank[i:i+batch_audit], spec_bank.t())
        for j in range(sims.size(0)): sims[j, i+j] = -1.0 # Ignorar self-match
        _, top_idx = torch.topk(sims, k=5, dim=1)
        neighbor_labels = train_ds.targets[top_idx.cpu()]
        
        # Voto de mayoría para detectar disonancia
        for j in range(sims.size(0)):
            votes = torch.bincount(neighbor_labels[j], minlength=10)
            consensus = torch.argmax(votes).item()
            if consensus != train_ds.targets[i+j].item():
                mislabels_mask[i+j] = True
        
        if (i+batch_audit) % 10000 == 0:
            print(f"Auditados {i+batch_audit}/60,000 recuerdos...")
    
    clean_indices = (~mislabels_mask).nonzero().squeeze()
    print(f"Purificación completada en {time.perf_counter()-t_audit:.2f}s.")
    print(f"Eliminados {mislabels_mask.sum().item()} recuerdos contradictorios.")
    print(f"Memoria de Confianza: {len(clean_indices)} ejemplos.")

    # 3. CONSTRUCCIÓN DE MEMORIA DUAL LIMPIA
    clean_raw_train = raw_train[clean_indices]
    clean_targets = train_ds.targets[clean_indices]
    
    print("\nPreparando dataset dual purificado...")
    std_train = standardize_robust_cpu(clean_raw_train)
    std_test = standardize_robust_cpu(raw_test)

    print("Inyectando recuerdos híbridos (Walsh + Islas)...")
    bank_org = get_hybrid_feats(clean_raw_train)
    bank_std = get_hybrid_feats(std_train)
    
    full_bank = torch.cat([bank_org, bank_std], dim=0)
    full_targets = torch.cat([clean_targets, clean_targets])
    full_bank_norm = F.normalize(full_bank.to(device), p=2, dim=1)

    # 4. BÚSQUEDA ASOCIATIVA DUAL
    print("Ejecutando resonancia dual cruzada...")
    query_org = get_hybrid_feats(raw_test)
    query_std = get_hybrid_feats(std_test)
    
    all_preds = []
    q_batch = 1000
    t_search = time.perf_counter()
    for i in range(0, 10000, q_batch):
        qo_b = query_org[i:i+q_batch].to(device)
        qs_b = query_std[i:i+q_batch].to(device)
        so = torch.mm(qo_b, full_bank_norm.t())
        ss = torch.mm(qs_b, full_bank_norm.t())
        # Resonancia máxima entre visiones
        indices = torch.argmax(torch.max(so, ss), dim=1)
        all_preds.append(full_targets[indices.cpu()])

    predictions = torch.cat(all_preds)
    acc = (predictions == test_ds.targets).float().mean().item()
    
    print("\n" + "="*55)
    print(f"RESULTADO MEMORIA DUAL PURIFICADA (V153)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Recuerdos:       {len(full_bank)} / 131,072 slots")
    print(f"Anomalías:       {mislabels_mask.sum().item()} eliminadas")
    print("="*55)

if __name__ == "__main__":
    run_experiment()
