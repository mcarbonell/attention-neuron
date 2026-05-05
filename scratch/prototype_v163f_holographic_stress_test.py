import torch
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
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

def run_stress_test():
    print(f"\n--- EXPERIMENTO V163f: HOLOGRAPHIC MEMORY STRESS TEST ---")
    
    # Parámetros del experimento
    dimensions = [512, 1024, 2048]
    context_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
    vocab_size = 5000  # Tamaño de la biblioteca de firmas
    trials = 20        # Semillas por configuración

    all_results = {}

    for d in dimensions:
        print(f"\nProbando Dimensión D={d}...")
        dim_results = []
        
        # Generar vocabulario aleatorio (Walsh Domain signatures)
        # Usamos ruido normalizado para simular firmas espectrales
        vocab = torch.randn(vocab_size, d, device=device)
        vocab = F.normalize(vocab, p=2, dim=1)
        
        for L in context_lengths:
            accuracies = []
            snrs = []
            
            for _ in range(trials):
                # 1. Preparar secuencia: 1 Aguja (pos 0) + L-1 Ruidos
                indices = torch.randint(0, vocab_size, (L,), device=device)
                target_idx = indices[0]
                target_token = vocab[target_idx]
                
                # 2. Ingesta Holográfica
                hologram = torch.zeros(1, d, device=device)
                for i in range(L):
                    # Roll circular para marcar posición
                    # i % d para evitar que el roll sea nulo si L > d
                    shifted = torch.roll(vocab[indices[i]], shifts=i % d, dims=0)
                    hologram += shifted
                
                # Normalizamos el holograma para mantener la escala
                hologram = F.normalize(hologram, p=2, dim=1)
                
                # 3. Recuperación (Query en posición 0)
                # Al buscar pos 0, el roll es 0
                query_pos = 0
                recall_vector = hologram.clone() # Roll(H, -0)
                
                # Comparar con todo el vocabulario en ESA posición
                # Nota: En un LLM real, el gating MoE haría esto de forma eficiente
                scores = torch.mm(recall_vector, vocab.t())
                
                # 4. Métricas
                # Accuracy: ¿Es el target el top 1?
                predicted_idx = torch.argmax(scores).item()
                accuracies.append(1.0 if predicted_idx == target_idx.item() else 0.0)
                
                # SNR: (Similitud con Target) / (Media de similitud con otros)
                target_score = scores[0, target_idx].item()
                other_scores = scores[0, torch.arange(vocab_size) != target_idx]
                avg_noise = torch.mean(other_scores).item()
                std_noise = torch.std(other_scores).item()
                
                # SNR = (Signal - MeanNoise) / StdNoise (Z-score)
                current_snr = (target_score - avg_noise) / (std_noise + 1e-6)
                snrs.append(current_snr)

            avg_acc = np.mean(accuracies)
            avg_snr = np.mean(snrs)
            print(f"  L={L:5} | Acc: {avg_acc*100:6.1f}% | SNR: {avg_snr:6.2f}")
            
            dim_results.append({
                "length": L,
                "accuracy": avg_acc,
                "snr": avg_snr
            })
            
        all_results[str(d)] = dim_results

    # Guardar resultados
    os.makedirs("results/raw", exist_ok=True)
    output_path = "results/raw/v163f_holographic_stress_test.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nResultados guardados en {output_path}")

if __name__ == "__main__":
    run_stress_test()
