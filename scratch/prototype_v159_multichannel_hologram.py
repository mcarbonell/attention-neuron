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

# --- K-MEANS RÁPIDO PARA CLANES ---
def get_clans(feats, k=6):
    """ Encuentra k arquetipos (clanes) para un conjunto de firmas """
    N, D = feats.shape
    # Inicialización aleatoria de centros
    centers = feats[torch.randperm(N)[:k]]
    for _ in range(5): # Pocas iteraciones bastan para clanes básicos
        sims = torch.mm(feats, centers.t())
        assignments = torch.argmax(sims, dim=1)
        for i in range(k):
            mask = assignments == i
            if mask.any():
                centers[i] = F.normalize(feats[mask].mean(0), p=2, dim=0)
    return centers

def run_experiment():
    print(f"\n--- EXPERIMENTO V159: MULTICHANNEL HOLOGRAPHIC CLANS ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    
    data_test = test_ds.data.unsqueeze(1).float().to(device) / 255.0
    targets_test = test_ds.targets.to(device)

    # 2. DESCUBRIMIENTO DE CLANES (6 por clase = 60 canales)
    K = 6
    print(f"Analizando morfogénesis: Descubriendo {K} clanes por clase...")
    clan_centers = [] # (10 * K, 1024)
    all_train_walsh = []
    
    t_clans = time.perf_counter()
    N_PER_CLASS = 1000
    for c in range(10):
        idx = (train_ds.targets == c).nonzero()[:N_PER_CLASS].squeeze()
        samples = train_ds.data[idx].float().to(device) / 255.0
        padded = torch.zeros(N_PER_CLASS, 32, 32).to(device)
        padded[:, :28, :28] = samples
        w_2d = F.normalize(fwht(padded.reshape(N_PER_CLASS, 1024)), p=2, dim=1)
        
        centers = get_clans(w_2d, k=K)
        clan_centers.append(centers)
        all_train_walsh.append(w_2d)
        
    clan_centers = torch.cat(clan_centers, dim=0) # (60, 1024)
    print(f"Clanes identificados en {time.perf_counter()-t_clans:.2f}s.")

    # 3. QUEMADO MULTI-CANAL (Multiplexing)
    # Sustrato de 1024 x 64 (64 portadoras de Walsh)
    substrate = torch.zeros(1024, 64).to(device)
    carriers = fwht(torch.eye(64).to(device))[:60] # 60 ondas para los 60 clanes
    
    print(f"Inyectando 10,000 recuerdos en 60 canales espectrales...")
    for c in range(10):
        w_2d = all_train_walsh[c]
        # Asignamos cada imagen de la clase c a su clan más cercano
        class_centers = clan_centers[c*K : (c+1)*K]
        sims = torch.mm(w_2d, class_centers.t())
        best_clans = torch.argmax(sims, dim=1)
        
        for k in range(K):
            clan_idx = c*K + k
            mask = best_clans == k
            if mask.any():
                # Modulamos las imágenes del clan con su onda portadora
                clan_imgs = w_2d[mask]
                class_interf = torch.mm(clan_imgs.t(), carriers[clan_idx:clan_idx+1].expand(clan_imgs.size(0), -1))
                substrate += class_interf
                
    # 4. DECODIFICACIÓN POR RESONANCIA COLECTIVA
    print("Decodificando test set mediante interferencia de clanes...")
    test_padded = torch.zeros(10000, 32, 32).to(device)
    test_padded[:, :28, :28] = data_test.squeeze()
    test_walsh = F.normalize(fwht(test_padded.reshape(10000, 1024)), p=2, dim=1)
    
    t_dec = time.perf_counter()
    
    # Paso 1: Extraemos la señal de los 60 canales
    # ecos = (10000, 64)
    ecos = torch.mm(test_walsh, substrate)
    
    # Paso 2: Proyectamos sobre las 60 portadoras
    # channel_scores = (10000, 60)
    channel_scores = torch.mm(ecos, carriers.t())
    
    # Paso 3: Resonancia Colectiva (Sumamos energía por clase)
    # Sumamos los 6 canales que pertenecen a cada clase
    class_energy = torch.zeros(10000, 10).to(device)
    for c in range(10):
        class_energy[:, c] = torch.sum(channel_scores[:, c*K : (c+1)*K]**2, dim=1)
        
    predictions = torch.argmax(class_energy, dim=1)
    acc = (predictions == targets_test).float().mean().item()
    
    print("\n" + "="*55)
    print(f"RESULTADO CRISTAL MULTI-CANAL (V159)")
    print(f"="*55)
    print(f"Precisión Final: {acc*100:.2f}%")
    print(f"Canales (Clanes): 60 (6 por clase)")
    print(f"Tamaño Memoria:   65,536 valores (Compresión 78x)")
    print(f"Tiempo Inferencia: {time.perf_counter()-t_dec:.4f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v159_multichannel_hologram.json", "w") as f:
        json.dump({"accuracy": acc, "clans": 60}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
