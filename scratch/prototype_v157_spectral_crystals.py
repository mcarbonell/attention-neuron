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

def fwht_3d(cube):
    """
    Aplica Walsh-Hadamard en las 3 dimensiones de un cubo (C, H, W, D).
    Asumimos H, W, D son potencias de 2 (ej. 32, 32, 64).
    """
    C, H, W, D = cube.shape
    # 1. Walsh en Dimensión Espacial H*W (como si fuera 2D)
    flat_spatial = cube.permute(0, 3, 1, 2).reshape(C*D, H*W)
    w_spatial = fwht(flat_spatial).reshape(C, D, H, W).permute(0, 2, 3, 1)
    
    # 2. Walsh en Dimensión de Profundidad (D) - El eje de la varianza
    flat_depth = w_spatial.reshape(C*H*W, D)
    w_3d = fwht(flat_depth).reshape(C, H, W, D)
    
    return w_3d

def run_experiment():
    print(f"\n--- EXPERIMENTO V157: SPECTRAL MEMORY CRYSTALS (3D RESONANCE) ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    
    data_test = test_ds.data.unsqueeze(1).float().to(device) / 255.0
    targets_test = test_ds.targets.to(device)

    # 2. CRECIMIENTO DE LOS 10 CRISTALES (Uno por clase)
    print("Cultivando 10 Cristales Espectrales (32x32x64)...")
    crystals = []
    
    D = 64 # Profundidad del cristal (Número de variaciones por clase)
    for c in range(10):
        # Seleccionamos D ejemplos de la clase c
        indices = (train_ds.targets == c).nonzero()[:D].squeeze()
        samples = train_ds.data[indices].float() / 255.0 # (D, 28, 28)
        
        # Padding a 32x32 para FWHT
        padded_samples = torch.zeros(D, 32, 32)
        padded_samples[:, :28, :28] = samples
        
        # Stack en un cubo (1, 32, 32, 64)
        cube = padded_samples.permute(1, 2, 0).unsqueeze(0).to(device)
        
        # Transformada 3D - El Cristal Espectral
        crystal_3d = fwht_3d(cube)
        crystals.append(crystal_3d)
        
    crystals = torch.cat(crystals, dim=0) # (10, 32, 32, 64)
    print("Cristales listos. Iniciando prueba de resonancia holográfica...")

    # 3. RESONANCIA HOLOGRÁFICA
    # Pre-procesamos el Test Set (Padding a 32x32 y Walsh 2D)
    test_padded = torch.zeros(10000, 32, 32).to(device)
    test_padded[:, :28, :28] = data_test.squeeze()
    
    # Transformamos el test al dominio de Walsh 2D
    test_flat = test_padded.reshape(10000, 1024)
    test_walsh = fwht(test_flat) # (10000, 1024)
    
    # Búsqueda de Resonancia
    # Proyectamos el vector de test sobre las "láminas" espectrales de cada cristal
    # Cada cristal de 1024x64 se puede ver como 64 filtros espectrales
    t0 = time.perf_counter()
    
    # Normalizamos cristales para similitud coseno
    # (10, 1024, 64)
    flat_crystals = crystals.reshape(10, 1024, 64)
    
    all_scores = []
    batch_size = 1000
    for i in range(0, 10000, batch_size):
        q_batch = test_walsh[i:i+batch_size] # (B, 1024)
        
        # Resonancia: Suma de las proyecciones al cuadrado (Energía de Interferencia)
        # Para cada clase c, sumamos la similitud de q con cada una de las 64 capas del cristal
        # scores[b, c] = sum_d (q[b] * crystal[c, :, d])^2
        
        # Usamos bmm para eficiencia: (B, 1, 1024) @ (10, 1024, 64) -> (10, B, 64)
        # Pero es más fácil: 
        # (B, 1024) @ (1024, 10*64) -> (B, 640) -> reshape (B, 10, 64)
        interferences = torch.mm(q_batch, flat_crystals.permute(1, 0, 2).reshape(1024, 640))
        interferences = interferences.reshape(-1, 10, 64)
        
        # La puntuación es la energía total de resonancia (Suma de cuadrados de coeficientes proyectados)
        scores = torch.sum(interferences**2, dim=2) # (B, 10)
        all_scores.append(scores)
        
    final_scores = torch.cat(all_scores, dim=0)
    predictions = torch.argmax(final_scores, dim=1)
    
    acc = (predictions == targets_test).float().mean().item()
    dt = time.perf_counter() - t0
    
    print("\n" + "="*55)
    print(f"RESULTADO CRISTALES ESPECTRALES (V157)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Objetos en Memoria: 10 Cristales (3D)")
    print(f"Densidad: {D} variaciones por cristal")
    print(f"Tiempo Resonancia: {dt:.2f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v157_spectral_crystals.json", "w") as f:
        json.dump({"accuracy": acc, "variations": D}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
