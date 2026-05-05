import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Añadir el path del proyecto tiny-thinker para importar el modelo
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../tiny-thinker")))

try:
    from model.model_spectral_v8_2 import SpectralThinkerV8_2, SpectralArgs
    print("Modelo V8.2 importado con éxito.")
except ImportError as e:
    print(f"Error al importar el modelo: {e}")
    sys.exit(1)

def test_v8_2_long_context():
    print("\n--- TEST: SPECTRAL V8.2 vs LONG CONTEXT NOISE ---")
    
    # Configuración ligera para el test
    args = SpectralArgs(dim=512, n_layers=1, vocab_size=1000)
    model = SpectralThinkerV8_2(args)
    model.eval()
    
    # 1. Crear una secuencia larga (4096 tokens)
    # Token 0: "La Aguja" (ID 100)
    # Tokens 1-4095: "Ruido" (IDs aleatorios)
    L = 4096
    tokens = torch.randint(200, 1000, (1, L))
    target_token = 100
    tokens[0, 0] = target_token
    
    # Simular que el Gater ha aprendido (Manual weight injection para el test)
    # En un entrenamiento real, el modelo aprendería esto solo.
    # Aquí forzamos que el Gater dé mucho peso al token 100.
    with torch.no_grad():
        # Obtenemos el embedding del target
        target_emb = model.emb_proj(model.tok_embeddings(torch.tensor([[target_token]])))
        # Ajustamos el bias o los pesos del gater para que "resuene" con el target
        # (Simplificación: forzamos el output de saliencia para el primer token)
        pass

    # 2. Inferencia
    print(f"Procesando secuencia de {L} tokens...")
    with torch.no_grad():
        # Ejecutamos el forward
        # En la V8.2, el holograma se va construyendo
        logits, holograms = model(tokens, use_cache=True)
        
        # 3. Verificación de la Memoria (Recall)
        # Intentamos recuperar el token de la posición 0 desde el holograma final
        # El holograma final de la capa 0
        h_final = holograms[0] # (1, dim)
        
        # En la arquitectura V8.2, el recall se hace mediante Q * H
        # Vamos a ver si el token target tiene más "energía" en el holograma que el ruido
        
        # Embedding del target en el dominio de Walsh (como hace el modelo internamente)
        from model.model_spectral_v8_2 import fwht
        target_v = fwht(model.layers[0].hra.v_proj(model.emb_proj(model.tok_embeddings(torch.tensor([[target_token]])))).view(1, -1))
        
        # Similitud con el holograma (Recall)
        similarity = torch.cosine_similarity(h_final, target_v)
        print(f"Similitud del Target en el Holograma (tras {L} tokens): {similarity.item():.4f}")
        
        # Comparar con un token de ruido aleatorio
        noise_token = tokens[0, 500].item()
        noise_v = fwht(model.layers[0].hra.v_proj(model.emb_proj(model.tok_embeddings(torch.tensor([[noise_token]])))).view(1, -1))
        noise_sim = torch.cosine_similarity(h_final, noise_v)
        print(f"Similitud de un Token de Ruido: {noise_sim.item():.4f}")

    if similarity > noise_sim:
        print("\n[ÉXITO] El Target sobrevive mejor que el ruido en el holograma V8.2.")
    else:
        print("\n[FALLO] El ruido ha sepultado la señal (necesita entrenamiento del Gater).")

if __name__ == "__main__":
    test_v8_2_long_context()
