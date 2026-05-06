import torch
import torch.nn as nn
import time
import os
import sys

# Añadir path para importar el modelo de forma robusta
# El script se encuentra en scratch/, tiny-thinker está en ../../../tiny-thinker relativo a scratch/
script_dir = os.path.dirname(os.path.abspath(__file__))
tiny_thinker_path = os.path.abspath(os.path.join(script_dir, "../../../tiny-thinker"))
sys.path.append(tiny_thinker_path)

try:
    from model.model_spectral_v8_3_matrix_free import SpectralThinkerV8_3, SpectralArgs
except ImportError as e:
    print(f"Error importando el modelo: {e}")
    print(f"Path intentado: {tiny_thinker_path}")
    print("Asegúrate de que la carpeta 'tiny-thinker' existe en la raíz del proyecto.")
    sys.exit(1)

def benchmark_profiles():
    print("=== SIMULADOR DE ESCALA TINYTHINKER V8.3 (MATRIX-FREE) - V2 ===")
    print(f"Dispositivo: {'CPU'}")
    print(f"Hardware: AMD Ryzen 7 8845HS (Simulado en CPU)")
    print("-" * 80)
    
    profiles = [
        {"name": "V8.3-Micro",     "experts": 32,  "layers": 4,  "dim": 16384, "fp16": True, "k_emb": 64},
        {"name": "V8.3-Standard",  "experts": 128, "layers": 8,  "dim": 32768, "fp16": True, "k_emb": 128},
        {"name": "V8.3-Deep",      "experts": 128, "layers": 16, "dim": 32768, "fp16": True, "k_emb": 128},
        {"name": "V8.3-Wide",      "experts": 128, "layers": 8,  "dim": 65536, "fp16": True, "k_emb": 256},
        {"name": "V8.3-Ultra",     "experts": 256, "layers": 24, "dim": 32768, "fp16": True, "k_emb": 256},
    ]
    
    header = f"{'Perfil':<15} | {'Expertos':<8} | {'Dim':<8} | {'Params':<10} | {'Peso MB':<10} | {'Tok/s (CPU)':<12}"
    print(header)
    print("-" * len(header))

    for p in profiles:
        # Verificar potencia de 2 para FWHT
        if (p["dim"] & (p["dim"] - 1)) != 0:
            print(f"Error: Dim {p['dim']} no es potencia de 2.")
            continue

        args = SpectralArgs(
            dim=p["dim"], 
            n_layers=p["layers"], 
            vocab_size=32768, 
            num_experts=p["experts"],
            emb_dim=p["k_emb"],
            top_k=8
        )
        
        # Inicialización del modelo
        model = SpectralThinkerV8_3(args).to('cpu')
        
        if p.get("fp16"):
            model = model.half()
            bytes_per_param = 2
        else:
            bytes_per_param = 4
            
        total_params = sum(p.numel() for p in model.parameters())
        weight_mb = (total_params * bytes_per_param) / (1024 * 1024)
        
        # Velocidad de Inferencia
        dummy_input = torch.randint(0, args.vocab_size, (1, 1))
        
        try:
            with torch.no_grad():
                # Calentamiento rápido
                for _ in range(5):
                    _ = model(dummy_input)
                
                start = time.time()
                n_tokens = 50
                for _ in range(n_tokens):
                    _ = model(dummy_input)
                total_time = time.time() - start
                tokens_per_sec = n_tokens / total_time
        except Exception as e:
            tokens_per_sec = 0.0
            print(f"\nError en profile {p['name']}: {e}")

        print(f"{p['name']:<15} | {p['experts']:<8} | {p['dim']:<8} | {total_params/1e6:8.1f}M | {weight_mb:8.1f} MB | {tokens_per_sec:10.2f}")
        
        # Liberar memoria explícitamente
        del model
        import gc
        gc.collect()

if __name__ == "__main__":
    benchmark_profiles()
