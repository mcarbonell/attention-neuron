import torch
import torch.nn as nn
import time
import os
import sys

# Añadir path para importar el modelo de forma robusta
script_dir = os.path.dirname(os.path.abspath(__file__))
tiny_thinker_path = os.path.abspath(os.path.join(script_dir, "../../../tiny-thinker"))
sys.path.append(tiny_thinker_path)

try:
    from model.model_spectral_v8_3_matrix_free import SpectralThinkerV8_3, SpectralArgs as ArgsV83
    from model.model_spectral_v8_4_optimized import SpectralThinkerV8_4, SpectralArgs as ArgsV84
except ImportError as e:
    print(f"Error importando modelos: {e}")
    sys.exit(1)

def benchmark_comparison():
    print("=== COMPARATIVA DE VELOCIDAD: V8.3 (Baseline) vs V8.4 (Optimizado) ===")
    print("Hardware: CPU (Ryzen 7 8845HS)")
    
    # Perfil de prueba pesado para notar la diferencia en FWHT
    # 32k dim es donde v8.3 empezaba a sufrir (< 1.5 Tok/s)
    config = {"experts": 128, "layers": 16, "dim": 32768}
    
    print(f"\nConfiguración: {config['layers']} capas, {config['dim']} dim, {config['experts']} expertos")
    print("-" * 70)

    # 1. Preparar V8.3
    args3 = ArgsV83(dim=config["dim"], n_layers=config["layers"], num_experts=config["experts"])
    model3 = SpectralThinkerV8_3(args3).to('cpu').half()
    
    # 2. Preparar V8.4
    args4 = ArgsV84(dim=config["dim"], n_layers=config["layers"], num_experts=config["experts"])
    model4 = SpectralThinkerV8_4(args4).to('cpu').half()
    
    dummy_input = torch.randint(0, 32768, (1, 1))
    
    # --- BENCHMARK V8.3 ---
    print(f"Midiendo V8.3 (Baseline)...")
    try:
        with torch.no_grad():
            for _ in range(5): _ = model3(dummy_input) # Warmup
            start = time.time()
            n_tokens = 20
            for _ in range(n_tokens): _ = model3(dummy_input)
            tps3 = n_tokens / (time.time() - start)
        print(f"V8.3 Tok/s: {tps3:8.2f}")
    except Exception as e:
        print(f"Error en V8.3: {e}")
        tps3 = 0.001

    # --- BENCHMARK V8.4 ---
    print(f"\nMidiendo V8.4 (Optimizado - Spectral Residency)...")
    try:
        with torch.no_grad():
            print("Compilando kernels con torch.compile (esto puede tardar la primera vez)...")
            _ = model4(dummy_input) # Warmup + Compile
            for _ in range(4): _ = model4(dummy_input)
            
            start = time.time()
            n_tokens = 20
            for _ in range(n_tokens):
                _ = model4(dummy_input)
            tps4 = n_tokens / (time.time() - start)
        print(f"V8.4 Tok/s: {tps4:8.2f}")
    except Exception as e:
        print(f"Error en V8.4: {e}")
        tps4 = 0.001
    
    speedup = (tps4 / tps3 - 1) * 100
    print("-" * 70)
    print(f"MEJORA ESTIMADA: {speedup:+.1f}%")
    
    if tps4 > tps3:
        print("¡ÉXITO! La residencia espectral y la compilación de kernels han funcionado.")
    else:
        print("Aviso: Si la mejora es baja, puede deberse a que el coste del routing MoE")
        print("domina sobre el ahorro en FWHT en esta arquitectura específica.")

if __name__ == "__main__":
    benchmark_comparison()
