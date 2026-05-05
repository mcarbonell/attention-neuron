import torch
import torch.nn as nn
import time
import os
import sys

# Añadir path para importar el modelo
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../tiny-thinker")))

from model.model_spectral_v8_3_matrix_free import SpectralThinkerV8_3, SpectralArgs

def benchmark_profiles():
    print("=== SIMULADOR DE ESCALA TINYTHINKER V8.3 (MATRIX-FREE) ===")
    
    profiles = [
        {"name": "V8.3-Standard", "experts": 128, "layers": 8,  "dim": 32768, "fp16": True, "vocab": 32768, "k_emb": 128},
        {"name": "V8.3-Deep",     "experts": 128, "layers": 16, "dim": 32768, "fp16": True, "vocab": 32768, "k_emb": 128},
    ]
    
    print(f"{'Perfil':<12} | {'Expertos':<10} | {'Parámetros':<12} | {'Peso MB':<10} | {'Tok/s (CPU)':<12}")
    print("-" * 65)

    for p in profiles:
        args = SpectralArgs(
            dim=p["dim"], 
            n_layers=p["layers"], 
            vocab_size=p.get("vocab", 32768), 
            num_experts=p["experts"],
            emb_dim=p.get("k_emb", 128), # Nuevo parámetro de compresión
            top_k=16
        )
        
        model = SpectralThinkerV8_3(args).to('cpu')
        
        # Estadísticas
        total_params = sum(p.numel() for p in model.parameters())
        bytes_per_param = 2 if p.get("fp16") else 4
        weight_mb = (total_params * bytes_per_param) / (1024 * 1024)
        
        # Velocidad
        if p.get("fp16"):
            model = model.half()
        dummy_input = torch.randint(0, args.vocab_size, (1, 1))
        with torch.no_grad():
            for _ in range(5): # Calentamiento rápido
                _ = model(dummy_input)
            
            start = time.time()
            for _ in range(30): # 30 tokens para velocidad
                _ = model(dummy_input)
            total_time = time.time() - start
            tokens_per_sec = 30 / total_time
            
        print(f"{p['name']:<12} | {p['experts']:<10} | {total_params/1e6:10.1f}M | {weight_mb:8.1f} MB | {tokens_per_sec:10.2f}")

if __name__ == "__main__":
    benchmark_profiles()
