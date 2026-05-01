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

# --- NORMALIZACIÓN POR CENTRO DE GRAVEDAD (CoG) ---
def center_images_batch(images, batch_size=5000):
    """
    images: (B, 1, 28, 28)
    Centra las imágenes basándose en su centro de masa para eliminar el sesgo de traslación.
    Procesado por sub-lotes para evitar picos de memoria.
    """
    B, C, H, W = images.shape
    device = images.device
    
    # Rejilla de coordenadas (0 a 27)
    y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    y_coords = y_coords.to(device).float()
    x_coords = x_coords.to(device).float()
    
    centered_results = []
    
    for i in range(0, B, batch_size):
        batch = images[i : i + batch_size].to(device)
        curr_b = batch.size(0)
        
        # Calcular masa y centro de masa para cada imagen del lote
        mass = batch.view(curr_b, -1).sum(dim=1).view(curr_b, 1, 1, 1)
        mass[mass == 0] = 1.0 # Evitar división por cero
        
        cx = (batch * x_coords).view(curr_b, -1).sum(dim=1) / mass.view(curr_b)
        cy = (batch * y_coords).view(curr_b, -1).sum(dim=1) / mass.view(curr_b)
        
        # Calcular desplazamiento para llegar al centro geométrico (13.5, 13.5)
        # El centro teórico de 28x28 es 13.5 (índice medio)
        dx = 13.5 - cx
        dy = 13.5 - cy
        
        # Matriz de transformación afín para PyTorch grid_sample
        # [ [1, 0, tx], [0, 1, ty] ]
        # tx y ty deben estar en coordenadas normalizadas [-1, 1]
        # t_norm = 2 * d_pixel / Size
        theta = torch.zeros(curr_b, 2, 3).to(device)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = 2.0 * dx / W
        theta[:, 1, 2] = 2.0 * dy / H
        
        grid = F.affine_grid(theta, batch.size(), align_corners=False)
        centered_batch = F.grid_sample(batch, grid, align_corners=False)
        centered_results.append(centered_batch.cpu()) # Guardamos en CPU para ahorrar VRAM
        
    return torch.cat(centered_results, dim=0)

def run_experiment():
    print(f"\n--- EXPERIMENTO V145: INVARIANT SPECTRAL MEMORY (CoG NORMALIZATION) ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    data_train = train_dataset.data.unsqueeze(1).float() / 255.0
    targets_train = train_dataset.targets
    
    data_test = test_dataset.data.unsqueeze(1).float() / 255.0
    targets_test = test_dataset.targets

    def evaluate_pipeline(centered=False):
        label = "CENTRADA (Invariante)" if centered else "ESTÁNDAR (Original)"
        print(f"\n>>> Evaluando Memoria {label}...")
        
        t0 = time.perf_counter()
        
        # A. Normalización
        if centered:
            print("Aplicando centrado por Centro de Masa (CoG)...")
            proc_train = center_images_batch(data_train)
            proc_test = center_images_batch(data_test)
        else:
            proc_train = data_train
            proc_test = data_test
            
        # B. Transformada Walsh
        def to_walsh_batch(images, batch_size=10000):
            results = []
            for i in range(0, images.size(0), batch_size):
                batch = images[i : i + batch_size].to(device)
                flat = batch.view(batch.size(0), -1)
                padded = torch.zeros(flat.size(0), 1024).to(device)
                padded[:, :784] = flat
                results.append(fwht(padded).cpu())
            return torch.cat(results, dim=0)
            
        print("Transformando al dominio Espectral...")
        spec_bank = to_walsh_batch(proc_train)
        spec_query = to_walsh_batch(proc_test)
        
        # C. Clasificación 1-NN (Holográfica)
        print("Ejecutando Búsqueda Holográfica 1-NN...")
        # Normalizar para Similitud Coseno
        norm_bank = F.normalize(spec_bank.to(device), p=2, dim=1)
        norm_query = F.normalize(spec_query.to(device), p=2, dim=1)
        
        # Cálculo de similitud masivo (10k x 60k)
        # Dividimos en sub-batches si es necesario, pero 10k x 60k suele caber en memoria (2.4GB)
        sims = torch.mm(norm_query, norm_bank.t())
        best_match_idx = torch.argmax(sims, dim=1)
        predictions = targets_train[best_match_idx.cpu()]
        
        acc = (predictions == targets_test).float().mean().item()
        dt = time.perf_counter() - t0
        
        print(f"Resultado: Precisión {acc*100:.2f}% | Tiempo Total: {dt:.2f}s")
        return acc

    # Ejecutar comparativa
    acc_std = evaluate_pipeline(centered=False)
    acc_inv = evaluate_pipeline(centered=True)

    print("\n" + "="*55)
    print(f"VERDICTO FINAL: NORMALIZACIÓN POR CoG")
    print(f"="*50)
    print(f"Precisión Estándar: {acc_std*100:.2f}%")
    print(f"Precisión Centrada: {acc_inv*100:.2f}%")
    print(f"Mejora Absoluta:   {acc_inv*100 - acc_std*100:+.2f}%")
    print(f"Estatus: {'ÉXITO ✅' if acc_inv > acc_std else 'SIN CAMBIO ⚠️'}")
    print("="*55)

if __name__ == "__main__":
    run_experiment()
