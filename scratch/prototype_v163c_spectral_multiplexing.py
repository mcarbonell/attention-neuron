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

def run_multiplexing_experiment():
    print(f"\n--- EXPERIMENTO V163c: MULTIPLEXADO HOLOGRÁFICO (CONTEXTO COMPRIMIDO) ---")
    
    # 1. CARGA DE "TOKENS" (MNIST)
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Preparamos 100 muestras aleatorias como nuestra "biblioteca de tokens"
    sample_idx = torch.randperm(10000)[:100]
    tokens_raw = test_ds.data[sample_idx].float().to(device) / 255.0
    tokens_padded = torch.zeros(100, 32, 32).to(device)
    tokens_padded[:, :28, :28] = tokens_raw
    tokens_spec = F.normalize(fwht(tokens_padded.reshape(100, 1024)), p=2, dim=1)

    # 2. EXPERIMENTO DE CAPACIDAD
    # Vamos a meter L tokens en la MISMA memoria de 1024 floats
    sequence_lengths = [1, 2, 4, 8, 16, 32, 64]
    results = []

    print(f"{'Seq Len':>10} | {'Mem Size':>12} | {'Recall Acc':>12} | {'SNR':>10}")
    print("-" * 55)

    for L in sequence_lengths:
        successes = 0
        total_snr = 0
        trials = 50
        
        for _ in range(trials):
            # Seleccionamos L tokens al azar para la secuencia
            seq_idx = torch.randperm(100)[:L]
            sequence = tokens_spec[seq_idx]
            
            # --- FASE DE ALMACENAMIENTO (SUMA HOLOGRÁFICA) ---
            # Todos los tokens se funden en un solo vector de 1024
            hologram = sequence.sum(dim=0, keepdim=True)
            # Normalizamos para mantener la energía constante
            hologram = F.normalize(hologram, p=2, dim=1)
            
            # --- FASE DE RECUPERACIÓN ---
            # El "score" es la resonancia (dot product) entre el holograma y la biblioteca
            scores = torch.mm(hologram, tokens_spec.t())
            
            # Verificamos si los L tokens de la secuencia están en el Top-L
            top_indices = torch.topk(scores, k=L, dim=1).indices[0]
            
            # Comprobamos cuántos de los tokens originales están en el Top-L
            seq_idx_list = seq_idx.tolist()
            matched = sum([1 for idx in seq_idx_list if idx in top_indices.tolist()])
            successes += (matched / L)
            
            # Cálculo de SNR (Señal / Ruido de Interferencia)
            signal = scores[0, seq_idx].mean().item()
            noise_mask = torch.ones(100, dtype=torch.bool, device=device)
            noise_mask[seq_idx] = False
            noise = scores[0, noise_mask].abs().mean().item() # Usamos valor absoluto para el ruido
            total_snr += (signal / (noise + 1e-8))

        avg_acc = (successes / trials) * 100
        avg_snr = (total_snr / trials)
        print(f"{L:10} | {'1024 floats':>12} | {avg_acc:11.1f}% | {avg_snr:10.2f}")
        results.append({"len": L, "acc": avg_acc, "snr": avg_snr})

    print("\n" + "="*55)
    print(f"CONCLUSIÓN: CAPACIDAD HOLOGRÁFICA")
    print(f"="*55)
    print(f"Resultados preliminares para memoria asociativa.")
    print(f"Capacidad lineal teórica: DIM / log(N).")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v163c_multiplexing.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_multiplexing_experiment()
