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

def run_experiment():
    print(f"\n--- EXPERIMENTO V158: HOLOGRAPHIC MULTIPLEXING (SINGLE CRYSTAL) ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    
    data_test = test_ds.data.unsqueeze(1).float().to(device) / 255.0
    targets_test = test_ds.targets.to(device)

    # 2. PREPARACIÓN DE PORTADORAS (Carriers)
    # Generamos 10 ondas de referencia ortogonales usando Walsh
    # Longitud 32: permite superponer 32 dimensiones de información
    eye = torch.eye(32).to(device)
    carriers = fwht(eye)[:10] # (10, 32) - 10 ondas ortogonales
    
    # 3. QUEMADO DEL CRISTAL MAESTRO (Burning)
    # El sustrato es una matriz de 1024 x 32 (32,768 valores)
    # Equivale a solo 32 imágenes almacenadas, pero guardaremos 5,000.
    substrate = torch.zeros(1024, 32).to(device)
    
    N_PER_CLASS = 500 # Guardamos 5000 recuerdos en total
    print(f"Quemando {N_PER_CLASS * 10} recuerdos en un solo cristal de 32,768 valores...")
    
    t_burn = time.perf_counter()
    for c in range(10):
        idx = (train_ds.targets == c).nonzero()[:N_PER_CLASS].squeeze()
        samples = train_ds.data[idx].float().to(device) / 255.0 # (N, 28, 28)
        
        # Padding a 32x32 y Walsh 2D
        padded = torch.zeros(N_PER_CLASS, 32, 32).to(device)
        padded[:, :28, :28] = samples
        flat = padded.reshape(N_PER_CLASS, 1024)
        w_2d = fwht(flat) # (N, 1024)
        
        # Multiplexación: Cada imagen se modula con la onda de su clase
        # interferencia = Suma( Imagen_i * Onda_clase )
        # Esto es un producto externo acumulado
        class_interf = torch.mm(w_2d.t(), carriers[c:c+1].expand(N_PER_CLASS, -1))
        substrate += class_interf
        
    print(f"Cristal Maestro cargado en {time.perf_counter()-t_burn:.2f}s.")

    # 4. DECODIFICACIÓN POR RESONANCIA (Decoding)
    print("Iniciando decodificación por correlación de portadoras...")
    
    # Pre-procesamos el Test Set
    test_padded = torch.zeros(10000, 32, 32).to(device)
    test_padded[:, :28, :28] = data_test.squeeze()
    test_walsh = fwht(test_padded.reshape(10000, 1024))
    
    t_dec = time.perf_counter()
    
    # Paso 1: Proyección del test sobre el sustrato (Ecos del cristal)
    # ecos = Test @ Substrate -> (10000, 32)
    ecos = torch.mm(test_walsh, substrate)
    
    # Paso 2: Filtrado por Portadoras (Correlación con las ondas de clase)
    # scores = Ecos @ Carriers.T -> (10000, 10)
    scores = torch.mm(ecos, carriers.t())
    
    predictions = torch.argmax(scores, dim=1)
    acc = (predictions == targets_test).float().mean().item()
    
    print("\n" + "="*55)
    print(f"RESULTADO CRISTAL MULTIPLEXADO (V158)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Capacidad:       5,000 imágenes")
    print(f"Tamaño Memoria:  32,768 valores (Compresión 156x)")
    print(f"Tiempo Inferencia: {time.perf_counter()-t_dec:.4f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v158_holographic_multiplexing.json", "w") as f:
        json.dump({"accuracy": acc, "stored": N_PER_CLASS*10}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
