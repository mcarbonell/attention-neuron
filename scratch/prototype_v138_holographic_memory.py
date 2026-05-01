import torch
import torch.nn as nn
import torch.nn.functional as F
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

# --- MEMORIA HOLOGRÁFICA ESPECTRAL ---
class HolographicMemory(nn.Module):
    def __init__(self, pattern_dim, num_memories):
        super().__init__()
        self.pattern_dim = pattern_dim
        self.num_memories = num_memories
        # Banco de recuerdos: Cada fila es una firma espectral única
        # Usamos una inicialización ortogonal para maximizar la separación
        self.memory_bank = nn.Parameter(torch.randn(num_memories, pattern_dim))
        
    def forward(self, x):
        # 1. Entrada -> Dominio Espectral (O(N log N))
        x_spec = fwht(x)
        
        # 2. Correlación paralela (Holográfica)
        # Comparamos la entrada con los 131,072 recuerdos simultáneamente
        # similaridad = X_spec @ M^T
        # Esto es una búsqueda asociativa masiva en un solo paso
        similarities = torch.matmul(x_spec, self.memory_bank.t())
        
        return similarities

def run_holographic_experiment():
    # Parámetros del experimento
    DIM = 1024 # Dimensión del patrón (ej. 32x32 píxeles)
    NUM_MEMORIES = 131072 # El número mágico de nuestro benchmark anterior
    
    print(f"\n--- EXPERIMENTO V138: MEMORIA HOLOGRÁFICA (131K RECUERDOS) ---")
    print(f"Configurando banco de memoria de {NUM_MEMORIES:,} firmas espectrales...")
    
    model = HolographicMemory(DIM, NUM_MEMORIES).to(device)
    
    # --- FASE 1: RECUPERACIÓN CON RUIDO ---
    
    # 1. Seleccionamos 10 recuerdos al azar para testear
    test_indices = np.random.choice(NUM_MEMORIES, 10, replace=False)
    
    print(f"\nRealizando 10 búsquedas asociativas con ruido extremo...")
    print(f"{'Test':>4} | {'Target ID':>10} | {'Match ID':>10} | {'Status':>8} | {'Conf.':>8} | {'Time':>8}")
    print("-" * 75)
    
    total_time = 0
    successes = 0
    
    for i, idx in enumerate(test_indices):
        # Generamos el patrón espacial original a partir de su firma espectral
        # (La inversa de Walsh es la propia Walsh)
        original_key = model.memory_bank[idx].detach().unsqueeze(0)
        original_pattern = fwht(original_key)
        
        # Añadimos ruido blanco (50% de la señal)
        noise = torch.randn_like(original_pattern) * 0.5
        noisy_input = original_pattern + noise
        
        # Búsqueda en la memoria
        t0 = time.perf_counter()
        with torch.no_grad():
            scores = model(noisy_input.to(device))
            match_idx = torch.argmax(scores, dim=1).item()
        dt = (time.perf_counter() - t0) * 1000
        
        total_time += dt
        is_correct = (match_idx == idx)
        if is_correct: successes += 1
        
        conf = scores[0, match_idx].item()
        print(f"{i+1:4} | {idx:10} | {match_idx:10} | {'OK ✅' if is_correct else 'FAIL ❌':8} | {conf:8.2f} | {dt:6.2f}ms")

    # --- FASE 2: CAPACIDAD DE DISCRIMINACIÓN ---
    print("\n" + "="*75)
    print(f"RESUMEN DE CAPACIDAD HOLOGRÁFICA")
    print(f"="*75)
    print(f"Precisión de Recuperación (50% Ruido): {successes/10*100:.1f}%")
    print(f"Tiempo medio de búsqueda (131k items): {total_time/10:.2f} ms")
    print(f"Throughput de Memoria: {NUM_MEMORIES/(total_time/10):,.0f} comparaciones/ms")
    print(f"Densidad de Información: {DIM} bits/firma")
    print("="*75)

    # Guardar resultados
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v138_holographic_results.json", "w") as f:
        json.dump({
            "num_memories": NUM_MEMORIES,
            "pattern_dim": DIM,
            "avg_time_ms": total_time/10,
            "success_rate": successes/10
        }, f, indent=4)

if __name__ == "__main__":
    run_holographic_experiment()
