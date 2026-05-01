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
    device_gpu = torch_directml.device()
    print(f"Using DirectML device for search: {device_gpu}")
except ImportError:
    device_gpu = torch.device('cpu')

# --- TRANSFORMADA DE WALSH-HADAMARD RÁPIDA (FWHT) ---
def fwht(x):
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

# --- NORMALIZADOR ROBUSTO EN CPU ---
def standardize_robust_cpu(images, target_size=20, target_mass=100.0):
    """
    Normalización morfológica garantizada en CPU para evitar errores de DML.
    """
    B, C, H, W = images.shape
    # Aseguramos que estamos en CPU
    imgs_in = images.cpu()
    std_images = torch.zeros(B, 1, H, W)
    
    t0 = time.perf_counter()
    for i in range(B):
        img = imgs_in[i, 0]
        # Binarización para detectar bordes
        mask = img > 0.05
        coords = torch.nonzero(mask)
        
        if coords.size(0) < 5:
            std_images[i] = imgs_in[i]
            continue
            
        # Bounding box
        y_min, x_min = coords[:, 0].min().item(), coords[:, 1].min().item()
        y_max, x_max = coords[:, 0].max().item(), coords[:, 1].max().item()
        
        h_crop = y_max - y_min + 1
        w_crop = x_max - x_min + 1
        
        # Factor de escala proporcional
        scale = target_size / max(h_crop, w_crop)
        new_h, new_w = int(h_crop * scale), int(w_crop * scale)
        new_h, new_w = max(1, new_h), max(1, new_w)
        
        # Extraer y redimensionar
        crop = img[y_min:y_max+1, x_min:x_max+1].reshape(1, 1, h_crop, w_crop)
        resized = F.interpolate(crop, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # Normalizar masa (intensidad total)
        mass = resized.sum()
        if mass > 0:
            resized = resized * (target_mass / mass)
            
        # Pegado centrado
        sy = (H - new_h) // 2
        sx = (W - new_w) // 2
        std_images[i, 0, sy:sy+new_h, sx:sx+new_w] = resized.reshape(new_h, new_w)
        
        if (i+1) % 20000 == 0:
            print(f"Estandarizados {i+1}/{B}...")
            
    print(f"Estandarización terminada en {time.perf_counter()-t0:.2f}s")
    return std_images

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
    print(f"\n--- EXPERIMENTO V149: ROBUST STANDARDIZED MEMORY ---")
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    
    # Estandarización en CPU (Seguridad total)
    std_train = standardize_robust_cpu(train_ds.data.unsqueeze(1).float() / 255.0)
    std_test = standardize_robust_cpu(test_ds.data.unsqueeze(1).float() / 255.0)
    
    # Diagnóstico Visual
    print("Guardando muestras en results/figures/v149_robust_samples.png...")
    fig, axes = plt.subplots(2, 10, figsize=(18, 5))
    for i in range(10):
        axes[0, i].imshow(train_ds.data[i], cmap='gray')
        axes[0, i].set_title(f"Orig {train_ds.targets[i]}")
        axes[0, i].axis('off')
        axes[1, i].imshow(std_train[i, 0], cmap='magma')
        axes[1, i].set_title("Robust Std")
        axes[1, i].axis('off')
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/v149_robust_samples.png")
    
    # Extracción y Búsqueda (GPU si es posible)
    print("\nExtrayendo firmas híbridas...")
    def get_feats(data_batch):
        batch = data_batch.to(device_gpu)
        # Walsh
        flat = batch.view(batch.size(0), -1)
        padded = torch.zeros(flat.size(0), 1024).to(device_gpu)
        padded[:, :784] = flat
        w = F.normalize(fwht(padded), p=2, dim=1)
        # Islas
        isl = F.normalize(get_island_signatures(batch), p=2, dim=1)
        return torch.cat([w, isl], dim=1)

    t_search = time.perf_counter()
    bank = get_feats(std_train)
    query = get_feats(std_test)
    
    print("Ejecutando resonancia holográfica...")
    sims = torch.mm(query, bank.t())
    best = torch.argmax(sims, dim=1)
    acc = (train_ds.targets[best.cpu()] == test_ds.targets).float().mean().item()
    
    print("\n" + "="*55)
    print(f"RESULTADO MEMORIA ROBUSTA (V149)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Tiempo Búsqueda: {time.perf_counter()-t_search:.2f}s")
    print("="*55)

if __name__ == "__main__":
    run_experiment()
