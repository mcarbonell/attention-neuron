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

# --- EXTRACTOR DE FIRMAS DE ISLAS (MORFOLOGÍA) ---
def get_island_signatures(images):
    """
    Retorna un vector de 56 dimensiones (28H + 28V) con el conteo de islas.
    Un 'isla' es un segmento contiguo de píxeles activos en una fila o columna.
    """
    B, _, H, W = images.shape
    # Binarización suave para detectar estructura
    binary = (images > 0.1).float()
    
    # Islas horizontales (por fila)
    # Pad a la izquierda con ceros para detectar transiciones 0 -> 1 en la primera columna
    padded_h = F.pad(binary, (1, 0), value=0)
    diff_h = padded_h[:, :, :, 1:] - padded_h[:, :, :, :-1]
    islands_h = (diff_h == 1).float().sum(dim=3).squeeze(1) # Resultado: (B, 28)
    
    # Islas verticales (por columna)
    # Pad arriba con ceros para detectar transiciones 0 -> 1 en la primera fila
    padded_v = F.pad(binary, (0, 0, 1, 0), value=0)
    diff_v = padded_v[:, :, 1:, :] - padded_v[:, :, :-1, :]
    islands_v = (diff_v == 1).float().sum(dim=2).squeeze(1) # Resultado: (B, 28)
    
    return torch.cat([islands_h, islands_v], dim=1) # Vector morfológico de 56D

def run_experiment():
    print(f"\n--- EXPERIMENTO V146: HYBRID MORPHOLOGICAL-SPECTRAL MEMORY ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=60000)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=10000)
    
    data_train, targets_train = next(iter(train_loader))
    data_test, targets_test = next(iter(test_loader))
    
    targets_train = targets_train.to(device)
    targets_test = targets_test.to(device)

    # 2. EXTRACCIÓN DE CARACTERÍSTICAS
    t_start = time.perf_counter()
    
    print("Calculando firmas de Walsh (Dominio Espectral)...")
    def to_walsh(d):
        flat = d.view(d.size(0), -1)
        padded = torch.zeros(flat.size(0), 1024).to(device)
        padded[:, :784] = flat.to(device)
        return fwht(padded)
        
    walsh_train = to_walsh(data_train)
    walsh_test = to_walsh(data_test)

    print("Calculando firmas de Islas (Dominio Morfológico)...")
    # Las islas son invariantes al grosor del trazo, lo que ayuda a normalizar la 'mano'
    islands_train = get_island_signatures(data_train.to(device))
    islands_test = get_island_signatures(data_test.to(device))

    # 3. FUSIÓN HOLOGRÁFICA
    # Normalizamos los componentes por separado para que el matmul posterior 
    # sea una combinación equilibrada de ambos canales sensoriales.
    print("Inyectando ambos canales en la Memoria Híbrida (1080D)...")
    w_train_norm = F.normalize(walsh_train, p=2, dim=1)
    w_test_norm = F.normalize(walsh_test, p=2, dim=1)
    
    i_train_norm = F.normalize(islands_train, p=2, dim=1)
    i_test_norm = F.normalize(islands_test, p=2, dim=1)
    
    # Concatenamos Walsh (1024) + Islas (56) = 1080 dimensiones
    hybrid_train = torch.cat([w_train_norm, i_train_norm], dim=1)
    hybrid_test = torch.cat([w_test_norm, i_test_norm], dim=1)
    
    # 4. BÚSQUEDA ASOCIATIVA (1-NN)
    print("Ejecutando resonancia híbrida...")
    # Re-normalizamos el vector híbrido final para similitud coseno pura
    h_bank = F.normalize(hybrid_train, p=2, dim=1)
    h_query = F.normalize(hybrid_test, p=2, dim=1)
    
    # Búsqueda masiva: 10,000 queries contra 60,000 recuerdos
    sims = torch.mm(h_query, h_bank.t())
    best_match_idx = torch.argmax(sims, dim=1)
    predictions = targets_train[best_match_idx]
    
    acc = (predictions == targets_test).float().mean().item()
    total_time = time.perf_counter() - t_start
    
    print("\n" + "="*55)
    print(f"RESULTADO MEMORIA HÍBRIDA (V146)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Tiempo Total:    {total_time:.2f}s")
    print(f"Arquitectura:    Walsh (1024) + Island Signatures (56)")
    print("="*55)

    # Guardar resultados
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v146_hybrid_memory.json", "w") as f:
        json.dump({"accuracy": acc, "time": total_time, "dims": 1080}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
