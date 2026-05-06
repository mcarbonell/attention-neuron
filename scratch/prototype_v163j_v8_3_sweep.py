import torch
import torch.nn as nn
import time
import os
import sys
import itertools

# Añadir path para importar el modelo de forma robusta
script_dir = os.path.dirname(os.path.abspath(__file__))
tiny_thinker_path = os.path.abspath(os.path.join(script_dir, "../../../tiny-thinker"))
sys.path.append(tiny_thinker_path)

try:
    from model.model_spectral_v8_3_matrix_free import SpectralThinkerV8_3, SpectralArgs
except ImportError:
    print("Error: No se encontró el modelo tiny-thinker en la ruta especificada.")
    print(f"Ruta intentada: {tiny_thinker_path}")
    sys.exit(1)

def run_sweep():
    print("=== BARRIDO DE HIPERPARÁMETROS SPECTRAL V8.3 (MATRIX-FREE) ===")
    print("Evaluando todas las combinaciones de Expertos, Dimensión y Capas.")
    print("-" * 100)

    # Variables de barrido (60 combinaciones en total)
    expert_options = [32, 128, 256]
    dim_options    = [2048, 4096, 8192, 16384, 32768]
    layer_options  = [4, 8, 16, 24]
    
    combinations = list(itertools.product(expert_options, dim_options, layer_options))
    total_runs = len(combinations)
    
    # Cabecera de la tabla con 'Layers' (Capas) incluida
    header = f"{'Progreso':<10} | {'Exp':<5} | {'Dim':<6} | {'Lay':<4} | {'Params':<10} | {'Peso MB':<10} | {'Tok/s':<8}"
    print(header)
    print("-" * len(header))

    for i, (exp, dim, lay) in enumerate(combinations, 1):
        # Mantener k_emb constante para aislar el efecto de las otras variables
        # o escalarlo mínimamente para evitar cuellos de botella extremos en dims pequeñas
        k_emb = min(128, dim // 4) 
        
        args = SpectralArgs(
            dim=dim, 
            n_layers=lay, 
            vocab_size=32768, 
            num_experts=exp,
            emb_dim=k_emb,
            top_k=8
        )
        
        try:
            # Forzamos CPU y FP16 para el benchmark
            model = SpectralThinkerV8_3(args).to('cpu').half()
            
            total_params = sum(p.numel() for p in model.parameters())
            weight_mb = (total_params * 2) / (1024 * 1024) # 2 bytes por parámetro (FP16)
            
            # Benchmark de velocidad
            dummy_input = torch.randint(0, args.vocab_size, (1, 1))
            
            with torch.no_grad():
                # Calentamiento (3 tokens)
                for _ in range(3):
                    _ = model(dummy_input)
                
                start = time.time()
                n_toks = 30 # 30 tokens por combinación para mayor agilidad
                for _ in range(n_toks):
                    _ = model(dummy_input)
                end = time.time()
                
                tps = n_toks / (end - start)
            
            print(f"[{i:02d}/{total_runs}] {100*i/total_runs:5.1f}% | {exp:<5} | {dim:<6} | {lay:<4} | {total_params/1e6:8.2f}M | {weight_mb:8.1f} MB | {tps:8.2f}")
            
            # Limpieza agresiva de memoria
            del model
            import gc
            gc.collect()
            
        except Exception as e:
            # Capturar errores (p.ej. falta de memoria si dim fuera muy grande, aunque aquí no debería pasar)
            print(f"[{i:02d}/{total_runs}] ERROR en Exp={exp}, Dim={dim}, Lay={lay}: {str(e)[:30]}")

    print("-" * len(header))
    print("Barrido de escala completado satisfactoriamente.")

if __name__ == "__main__":
    run_sweep()
