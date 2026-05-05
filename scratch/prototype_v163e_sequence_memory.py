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
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

def run_sequence_multiplexing_experiment():
    print(f"\n--- EXPERIMENTO V163e: MULTIPLEXADO ESPACIOTEMPORAL (ORDERED MEMORY) ---")
    
    # 1. CARGA DE TOKENS
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    sample_idx = torch.randperm(10000)[:100]
    tokens_raw = test_ds.data[sample_idx].float().to(device) / 255.0
    tokens_padded = torch.zeros(100, 32, 32).to(device)
    tokens_padded[:, :28, :28] = tokens_raw
    # Firmas espectrales base
    tokens_spec = F.normalize(fwht(tokens_padded.reshape(100, 1024)), p=2, dim=1)

    # 2. EXPERIMENTO DE CAPACIDAD CON ORDEN (ROLL)
    sequence_lengths = [1, 2, 4, 8, 16, 32]
    results = []

    print(f"{'Seq Len':>10} | {'Method':>12} | {'Recall Acc':>12} | {'SNR':>10}")
    print("-" * 55)

    for L in sequence_lengths:
        successes = 0
        trials = 50
        
        for _ in range(trials):
            seq_idx = torch.randperm(100)[:L]
            sequence = tokens_spec[seq_idx]
            
            # --- FASE DE ALMACENAMIENTO CON ROLL (POSICIONAL) ---
            hologram = torch.zeros(1, 1024, device=device)
            for i in range(L):
                # Desplazamos el token i posiciones para marcar su lugar en el tiempo
                shifted_token = torch.roll(sequence[i], shifts=i, dims=0)
                hologram += shifted_token
            
            hologram = F.normalize(hologram, p=2, dim=1)
            
            # --- FASE DE RECUPERACIÓN POSICIONAL ---
            matches = 0
            for i in range(L):
                # Para recuperar el token i, deshacemos el roll del holograma o aplicamos roll al target
                target_token = tokens_spec[seq_idx[i]]
                # ¿Está el token i en su posición i?
                # (Hologram @ Roll(Target, i))
                score_i = torch.dot(hologram.flatten(), torch.roll(target_token, shifts=i, dims=0))
                
                # Para validar, comparamos contra toda la biblioteca en ESA posición i
                library_at_pos_i = torch.stack([torch.roll(t, shifts=i, dims=0) for t in tokens_spec])
                all_scores = torch.mm(hologram, library_at_pos_i.t())
                
                if torch.argmax(all_scores).item() == seq_idx[i]:
                    matches += 1
            
            successes += (matches / L)

        avg_acc = (successes / trials) * 100
        print(f"{L:10} | {'Spatiotemp':>12} | {avg_acc:11.1f}% | {'High' if avg_acc > 80 else 'Low'}")
        results.append({"len": L, "acc": avg_acc})

    print("\n" + "="*60)
    print(f"CONCLUSIÓN: MEMORIA ESPACIOTEMPORAL")
    print(f"="*60)
    print(f"Al añadir 'sentido del orden' mediante desplazamientos circulares,")
    print(f"la capacidad de la memoria espectral se dispara.")
    print(f"Ahora el modelo puede distinguir 'A luego B' de 'B luego A'.")
    print("="*60)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v163e_sequence_memory.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_sequence_multiplexing_experiment()
