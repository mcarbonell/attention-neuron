import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt

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

# --- NORMALIZADOR PROPORCIONAL Y DE MASA ---
def standardize_proportional_batch(images, target_size=20, target_mass=100.0, batch_size=2000):
    """
    Normaliza morfológicamente las imágenes manteniendo el Aspect Ratio.
    """
    B, C, H, W = images.shape
    std_results = []
    
    for b_idx in range(0, B, batch_size):
        batch = images[b_idx : b_idx + batch_size].to(device)
        curr_b = batch.size(0)
        processed_batch = torch.zeros_like(batch)
        
        for i in range(curr_b):
            img = batch[i, 0]
            mask = img > 0.1
            coords = torch.nonzero(mask)
            if coords.size(0) < 5:
                processed_batch[i] = batch[i]
                continue
                
            y_min, x_min = coords.min(dim=0)[0]
            y_max, x_max = coords.max(dim=0)[0]
            
            h_crop = y_max - y_min + 1
            w_crop = x_max - x_min + 1
            
            # ESCALADO PROPORCIONAL (Evita deformación)
            scale = target_size / max(h_crop, w_crop)
            new_h = int(h_crop * scale)
            new_w = int(w_crop * scale)
            
            if new_h < 1: new_h = 1
            if new_w < 1: new_w = 1
            
            crop = img[y_min:y_max+1, x_min:x_max+1].unsqueeze(0).unsqueeze(0)
            resized = F.interpolate(crop, size=(new_h, new_w), mode='bilinear', align_corners=False)
            
            # Centrar en canvas 28x28
            start_y = (H - new_h) // 2
            start_x = (W - new_w) // 2
            
            # Normalización de Masa (Contraste constante)
            current_mass = resized.sum()
            if current_mass > 0:
                resized = resized * (target_mass / current_mass)
                
            processed_batch[i, 0, start_y:start_y+new_h, start_x:start_x+new_w] = resized.squeeze()
            
        std_results.append(processed_batch.cpu())
        
    return torch.cat(std_results, dim=0)

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

def run_experiment():
    print(f"\n--- EXPERIMENTO V148: PROPORTIONAL STANDARDIZED MEMORY ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=60000)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=10000)
    
    data_train, targets_train = next(iter(train_loader))
    data_test, targets_test = next(iter(test_loader))
    
    targets_train = targets_train.to(device)
    targets_test = targets_test.to(device)

    # 2. ESTANDARIZACIÓN PROPORCIONAL
    t0 = time.perf_counter()
    print("Estandarizando dataset (Aspect Ratio Preserved)...")
    std_train = standardize_proportional_batch(data_train.to(device))
    std_test = standardize_proportional_batch(data_test.to(device))

    # Diagnóstico Visual
    print("Guardando muestras de diagnóstico en results/figures/v148_samples.png...")
    fig, axes = plt.subplots(2, 10, figsize=(18, 5))
    for i in range(10):
        axes[0, i].imshow(data_train[i, 0], cmap='gray')
        axes[0, i].set_title(f"Orig {targets_train[i].item()}")
        axes[0, i].axis('off')
        axes[1, i].imshow(std_train[i, 0], cmap='magma')
        axes[1, i].set_title("Std (Prop)")
        axes[1, i].axis('off')
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/v148_samples.png")
    
    # 3. EXTRACCIÓN HÍBRIDA
    print("Calculando firmas híbridas (Walsh + Islas)...")
    def get_features_batch(images, batch_size=5000):
        feats = []
        for i in range(0, images.size(0), batch_size):
            batch = images[i : i+batch_size].to(device)
            flat = batch.view(batch.size(0), -1)
            padded = torch.zeros(flat.size(0), 1024).to(device)
            padded[:, :784] = flat
            w = F.normalize(fwht(padded), p=2, dim=1)
            islands = get_island_signatures(batch)
            feats.append(torch.cat([w, F.normalize(islands, p=2, dim=1)], dim=1).cpu())
        return torch.cat(feats, dim=0)

    bank_feats = get_features_batch(std_train)
    query_feats = get_features_batch(std_test)

    # 4. BÚSQUEDA ASOCIATIVA
    print("Ejecutando resonancia híbrida...")
    bank_norm = F.normalize(bank_feats.to(device), p=2, dim=1)
    query_norm = F.normalize(query_feats.to(device), p=2, dim=1)
    
    sims = torch.mm(query_norm, bank_norm.t())
    best_match = torch.argmax(sims, dim=1)
    acc = (targets_train[best_match] == targets_test).float().mean().item()
    dt = time.perf_counter() - t0
    
    print("\n" + "="*55)
    print(f"RESULTADO MEMORIA ESTANDARIZADA (V148)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Tiempo Total:    {dt:.2f}s")
    print(f"Estatus: Aspect Ratio Preserved ✅")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v148_standardized_memory.json", "w") as f:
        json.dump({"accuracy": acc, "time": dt}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
