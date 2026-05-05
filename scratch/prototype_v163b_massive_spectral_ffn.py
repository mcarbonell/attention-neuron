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

# --- CLUSTERING MASIVO ---
def get_massive_clanes(feats, k=4096):
    """ 
    Genera un banco de memoria masivo (clanes). 
    Para k=4096, usamos un enfoque de mini-batch para no saturar la GPU.
    """
    N, D = feats.shape
    centers = feats[torch.randperm(N)[:k]]
    
    print(f"Sintonizando Manifold de {k} clanes (esto puede tardar unos segundos)...")
    for i in range(5): # Menos iteraciones pero más clanes
        # Calculamos similitudes en trozos para evitar OOM
        chunk_size = 5000
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(k, device=device)
        
        for j in range(0, N, chunk_size):
            end = min(j + chunk_size, N)
            batch_feats = feats[j:end]
            sims = torch.mm(batch_feats, centers.t())
            assignments = torch.argmax(sims, dim=1)
            
            # Usamos index_add_ para vectorizar la acumulación en los centros
            # Esto es mucho más rápido que un bucle for sobre k
            ones = torch.ones(batch_feats.size(0), device=device)
            new_centers.index_add_(0, assignments, batch_feats)
            counts.index_add_(0, assignments, ones)
        
        centers = F.normalize(new_centers / (counts.unsqueeze(1) + 1e-8), p=2, dim=1)
        print(f"  Iteración {i+1}/5 completada.")
        
    return centers

def run_massive_experiment():
    print(f"\n--- EXPERIMENTO V163b: MASSIVE SPECTRAL-FFN (LLM SCALE) ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    raw_test = test_ds.data.unsqueeze(1).float().to(device) / 255.0
    all_train_targets = train_ds.targets.to(device)

    # 2. CONSTRUCCIÓN DEL BANCO DE MEMORIA MASIVO (4096 Clanes)
    K_MASSIVE = 4096 
    sample_idx = torch.randperm(60000)[:30000] # Más datos para más clanes
    sample_data = train_ds.data[sample_idx].float().to(device) / 255.0
    padded_sample = torch.zeros(len(sample_idx), 32, 32).to(device)
    padded_sample[:, :28, :28] = sample_data
    w_sample = F.normalize(fwht(padded_sample.reshape(len(sample_idx), 1024)), p=2, dim=1)
    
    t_init = time.perf_counter()
    clanes = get_massive_clanes(w_sample, k=K_MASSIVE)
    dt_init = time.perf_counter() - t_init
    
    # Matriz de Síntesis (Down-Projection)
    print("Sintetizando Down-Projection Holográfica...")
    # Calculamos assignments en batch para el v_matrix
    chunk_size = 5000
    all_assignments = []
    for j in range(0, len(w_sample), chunk_size):
        end = min(j + chunk_size, len(w_sample))
        sims_chunk = torch.mm(w_sample[j:end], clanes.t())
        all_assignments.append(torch.argmax(sims_chunk, dim=1))
    assignments = torch.cat(all_assignments)
    
    sample_targets = all_train_targets[sample_idx]
    
    clan_labels = []
    for i in range(K_MASSIVE):
        mask = assignments == i
        if mask.any():
            targets = sample_targets[mask]
            clan_labels.append(torch.bincount(targets.cpu(), minlength=10).argmax().item())
        else:
            clan_labels.append(0)
            
    v_matrix = F.one_hot(torch.tensor(clan_labels), num_classes=10).float().to(device)

    # 3. EVALUACIÓN DE ESCALADO
    print(f"Evaluando Spectral-FFN de {K_MASSIVE} canales...")
    test_padded = torch.zeros(10000, 32, 32).to(device)
    test_padded[:, :28, :28] = raw_test.squeeze()
    test_walsh = F.normalize(fwht(test_padded.reshape(10000, 1024)), p=2, dim=1)
    
    t0 = time.perf_counter()
    
    # PASO 1: Up-Projection (Resonancia con 4096 expertos)
    h = torch.mm(test_walsh, clanes.t())
    
    # PASO 2: Activación (Sparse Attention)
    h_act = torch.pow(torch.clamp(h, min=0.0), 24)
    
    # PASO 3: Down-Projection
    output = torch.mm(h_act, v_matrix)
    
    preds = torch.argmax(output, dim=1)
    acc = (preds == test_ds.targets.to(device)).float().mean().item()
    dt_inf = time.perf_counter() - t0
    
    # PASO 4: Métrica de Fidelidad (Reconstrucción)
    best_clanes = clanes[torch.argmax(h, dim=1)]
    fidelity = (best_clanes * test_walsh).sum(1).mean().item()

    print("\n" + "="*60)
    print(f"RESULTADO MASSIVE SPECTRAL-FFN (V163b)")
    print(f"="*60)
    print(f"Precisión Final:   {acc*100:.2f}%")
    print(f"Fidelidad Espectral: {fidelity*100:.2f}% (Calidad JPEG)")
    print(f"Ancho del FFN:     {K_MASSIVE} clanes")
    print(f"Parámetros:        {K_MASSIVE * 1024 + K_MASSIVE * 10:,}")
    print(f"Latencia (10k):    {dt_inf:.4f}s")
    print(f"Tiempo Entrenamiento: {dt_init:.2f}s")
    print("="*60)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v163b_massive_spectral_ffn.json", "w") as f:
        json.dump({
            "accuracy": acc, 
            "fidelity": fidelity,
            "hidden_dim": K_MASSIVE,
            "latency": dt_inf
        }, f, indent=4)

if __name__ == "__main__":
    run_massive_experiment()
