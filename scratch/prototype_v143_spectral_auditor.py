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

def run_spectral_auditor():
    print(f"\n--- EXPERIMENTO V143: SPECTRAL AUDITOR (DATA SELF-HEALING) ---")
    
    # 1. CARGA DE MEMORIA COMPLETA (60k MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    print("Cargando 60,000 imágenes en la Memoria Holográfica...")
    loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=60000)
    data, targets = next(iter(loader))
    targets = targets.to(device)

    # Preparación Walsh
    flat = data.view(60000, -1)
    padded = torch.zeros(60000, 1024).to(device)
    padded[:, :784] = flat.to(device)
    
    # Transformar todo el dataset de una vez (O(N log N) es muy eficiente)
    spec_bank = fwht(padded)
    # Normalización para Similitud Coseno
    spec_bank_norm = F.normalize(spec_bank, p=2, dim=1)

    # 2. PROCESO DE AUDITORÍA CRUZADA
    # Vamos a comparar cada imagen con sus 59,999 vecinas para detectar disonancias
    print("\nIniciando Auditoría Cruzada (CSI Espectral)...")
    batch_size = 1000
    anomalies = []
    
    t_start = time.perf_counter()
    
    # Usamos memoria de buffer para no saturar la GPU con la matriz 60k x 60k completa
    for i in range(0, 60000, batch_size):
        end_idx = min(i + batch_size, 60000)
        batch_spec = spec_bank_norm[i:end_idx]
        batch_targets = targets[i:end_idx]
        
        # Similitud: (Batch, 60000)
        # Cada fila es la similitud de una imagen contra todo el dataset
        sims = torch.mm(batch_spec, spec_bank_norm.t())
        
        # Eliminamos la auto-similitud (la diagonal debe ser neutra)
        for j in range(batch_spec.size(0)):
            sims[j, i + j] = -1.0 # Valor muy bajo para ignorar el match consigo mismo
            
        # Buscamos el consenso de los 5 vecinos más cercanos
        top_vals, top_indices = torch.topk(sims, k=5, dim=1)
        
        for j in range(batch_spec.size(0)):
            neighbor_labels = targets[top_indices[j]]
            official_label = batch_targets[j].item()
            
            # Votación por mayoría
            votes = torch.bincount(neighbor_labels, minlength=10)
            consensus_label = torch.argmax(votes).item()
            confidence = votes[consensus_label].item() / 5.0
            
            # Si el consenso contradice la etiqueta oficial, tenemos una anomalía
            if consensus_label != official_label:
                anomalies.append({
                    "index": i + j,
                    "official_label": official_label,
                    "spectral_consensus": consensus_label,
                    "confidence": confidence,
                    "similarity": top_vals[j, 0].item()
                })
        
        if end_idx % 10000 == 0:
            print(f"Auditados {end_idx}/60,000 recuerdos...")

    total_time = time.perf_counter() - t_start
    print(f"\nAuditoría completada en {total_time:.2f} segundos.")

    # 3. RESULTADOS Y RANKING
    # Ordenamos por confianza (consenso unánime de los vecinos) y luego por similitud absoluta
    anomalies.sort(key=lambda x: (x["confidence"], x["similarity"]), reverse=True)
    
    print("\n" + "="*75)
    print(f"ANOMALÍAS ESPECTRALES DETECTADAS (POSIBLES ERRORES DE ETIQUETA)")
    print("="*75)
    print(f"{'Index':>8} | {'Official':>8} | {'Consensus':>10} | {'Confidence':>10} | {'Simil.'}")
    print("-" * 75)
    for a in anomalies[:15]:
        status = "CRITICAL 🚨" if a['confidence'] >= 0.8 else "SUSPICIOUS ⚠️"
        print(f"{a['index']:8} | {a['official_label']:8} | {a['spectral_consensus']:10} | {a['confidence']*100:>9.0f}% | {a['similarity']:.3f} | {status}")
    print("="*75)
    print(f"Total Anomalías: {len(anomalies)} ({len(anomalies)/60000*100:.2f}% del dataset)")
    print("="*75)

    # Guardar resultados para análisis posterior
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v143_spectral_auditor.json", "w") as f:
        json.dump(anomalies, f, indent=4)
    print(f"Resultados guardados en results/raw/v143_spectral_auditor.json")

if __name__ == "__main__":
    run_spectral_auditor()
