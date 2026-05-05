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

def run_attention_experiment():
    print(f"\n--- EXPERIMENTO V163g: HOLOGRAPHIC ATTENTION SALIENCY ---")
    
    # Parámetros fijos
    d = 1024
    vocab_size = 5000
    trials = 20
    
    # Variables a explorar
    context_lengths = [256, 1024, 4096]
    attention_weights = [1.0, 5.0, 20.0, 50.0, 100.0]  # Peso de la "Aguja"

    all_results = {}

    # Generar vocabulario base
    vocab = torch.randn(vocab_size, d, device=device)
    vocab = F.normalize(vocab, p=2, dim=1)

    for L in context_lengths:
        print(f"\nContexto L={L} tokens:")
        weight_results = []
        
        for w in attention_weights:
            accuracies = []
            snrs = []
            
            for _ in range(trials):
                # 1. Preparar secuencia
                indices = torch.randint(0, vocab_size, (L,), device=device)
                target_idx = indices[0]
                
                # 2. Ingesta Holográfica con Atención
                hologram = torch.zeros(1, d, device=device)
                for i in range(L):
                    # El target (i=0) recibe peso 'w', el resto peso 1.0
                    weight = w if i == 0 else 1.0
                    
                    shifted = torch.roll(vocab[indices[i]], shifts=i % d, dims=0)
                    hologram += weight * shifted
                
                hologram = F.normalize(hologram, p=2, dim=1)
                
                # 3. Recuperación
                recall_vector = hologram.clone()
                scores = torch.mm(recall_vector, vocab.t())
                
                # 4. Métricas
                predicted_idx = torch.argmax(scores).item()
                accuracies.append(1.0 if predicted_idx == target_idx.item() else 0.0)
                
                target_score = scores[0, target_idx].item()
                other_scores = scores[0, torch.arange(vocab_size) != target_idx]
                avg_noise = torch.mean(other_scores).item()
                std_noise = torch.std(other_scores).item()
                current_snr = (target_score - avg_noise) / (std_noise + 1e-6)
                snrs.append(current_snr)

            avg_acc = np.mean(accuracies)
            avg_snr = np.mean(snrs)
            print(f"  Peso Aguja={w:4} | Acc: {avg_acc*100:6.1f}% | SNR: {avg_snr:6.2f}")
            
            weight_results.append({
                "weight": w,
                "accuracy": avg_acc,
                "snr": avg_snr
            })
            
        all_results[str(L)] = weight_results

    # Guardar resultados
    os.makedirs("results/raw", exist_ok=True)
    output_path = "results/raw/v163g_holographic_attention.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nResultados guardados en {output_path}")

if __name__ == "__main__":
    run_attention_experiment()
