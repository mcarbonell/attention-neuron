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

def run_sentence_recall_experiment():
    print(f"\n--- EXPERIMENTO V163h: SENTENCE COMPRESSION & RECALL ---")
    
    # Parámetros
    d = 2048  # Subimos a 2048 para mayor fidelidad de frase
    vocab_size = 5000
    sentence_len = 8
    noise_len = 2000
    trials = 10
    
    # Pesos de atención (basados en v163g)
    sentence_weight = 30.0
    noise_weight = 1.0

    # Vocabulario
    vocab = torch.randn(vocab_size, d, device=device)
    vocab = F.normalize(vocab, p=2, dim=1)

    all_accuracies = []

    for t in range(trials):
        # 1. Crear Frase (índices aleatorios)
        sentence_indices = torch.randint(0, vocab_size, (sentence_len,), device=device)
        
        # 2. Ingesta Holográfica
        hologram = torch.zeros(1, d, device=device)
        
        # Insertar frase en las primeras posiciones
        for i in range(sentence_len):
            shifted = torch.roll(vocab[sentence_indices[i]], shifts=i, dims=0)
            hologram += sentence_weight * shifted
            
        # Insertar ruido en el resto de la ventana
        noise_indices = torch.randint(0, vocab_size, (noise_len,), device=device)
        for i in range(noise_len):
            # Posiciones desplazadas para no solapar directamente con la frase (opcional, pero realista)
            pos = i + sentence_len
            shifted = torch.roll(vocab[noise_indices[i]], shifts=pos % d, dims=0)
            hologram += noise_weight * shifted
            
        hologram = F.normalize(hologram, p=2, dim=1)

        # 3. Recuperación de la frase completa
        recovered_indices = []
        for i in range(sentence_len):
            # Interrogar posición i
            recall_vector = hologram.clone()
            # En la práctica, haríamos Roll(hologram, -i) y compararíamos con vocab
            # Aquí usamos el truco de Roll(vocab, i) que es matemáticamente equivalente para el dot product
            library_at_pos_i = torch.stack([torch.roll(v, shifts=i, dims=0) for v in vocab])
            scores = torch.mm(hologram, library_at_pos_i.t())
            
            predicted_idx = torch.argmax(scores).item()
            recovered_indices.append(predicted_idx)

        # Calcular fidelidad de la frase
        matches = sum([1 for a, b in zip(sentence_indices.tolist(), recovered_indices) if a == b])
        fidelity = matches / sentence_len
        all_accuracies.append(fidelity)
        
        print(f"  Trial {t+1}: Fidelity {fidelity*100:6.1f}% | Matches: {matches}/{sentence_len}")

    avg_fidelity = np.mean(all_accuracies)
    print(f"\nFIDELIDAD MEDIA DE FRASE: {avg_fidelity*100:6.1f}%")
    
    # Guardar resultados
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v163h_sentence_recall.json", "w") as f:
        json.dump({"avg_fidelity": avg_fidelity, "trials": all_accuracies}, f, indent=4)

if __name__ == "__main__":
    run_sentence_recall_experiment()
