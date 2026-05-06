import torch
import torch.nn as nn
import time
import os
import sys

# Añadir path para importar el modelo
script_dir = os.path.dirname(os.path.abspath(__file__))
tiny_thinker_path = os.path.abspath(os.path.join(script_dir, "../../../tiny-thinker"))
sys.path.append(tiny_thinker_path)

try:
    from model.model_spectral_v8_4_optimized import SpectralThinkerV8_4, SpectralArgs as ArgsV84
    from model.model_spectral_v8_5_native import SpectralThinkerV8_5, SpectralArgs as ArgsV85
except ImportError as e:
    print(f"Error importando modelos: {e}")
    sys.exit(1)

def benchmark_native():
    print("=== TEST DE RENDIMIENTO V8.5: MOTOR NATIVO (C++) ===")
    print("Hardware: CPU (Ryzen 7 8845HS - 16 hilos con OpenMP)")
    
    # Configuración de alta resolución
    config = {"experts": 128, "layers": 16, "dim": 32768}
    print(f"Perfil: {config['layers']} capas, {config['dim']} dim, {config['experts']} expertos")
    print("-" * 70)

    # 1. Preparar V8.4 (Baseline Python Optimizado)
    args4 = ArgsV84(dim=config["dim"], n_layers=config["layers"], num_experts=config["experts"])
    model4 = SpectralThinkerV8_4(args4).to('cpu').half()
    
    # 2. Preparar V8.5 (Native)
    args5 = ArgsV85(dim=config["dim"], n_layers=config["layers"], num_experts=config["experts"])
    # Nota: Aquí se disparará la compilación de C++ al inicializar o al primer forward
    model5 = SpectralThinkerV8_5(args5).to('cpu').half()
    
    dummy_input = torch.randint(0, 32768, (1, 1))
    
    # --- BENCHMARK V8.4 ---
    print(f"Midiendo V8.4 (Python Baseline)...")
    with torch.no_grad():
        for _ in range(5): _ = model4(dummy_input)
        start = time.time()
        for _ in range(20): _ = model4(dummy_input)
        tps4 = 20 / (time.time() - start)
    print(f"V8.4 Tok/s: {tps4:8.2f}")
    
    # --- BENCHMARK V8.5 ---
    print(f"\nMidiendo V8.5 (C++ Native Engine)...")
    try:
        with torch.no_grad():
            # El primer forward compila el kernel si no está compilado
            print("Iniciando motor nativo...")
            _ = model5(dummy_input) # Warmup + JIT Compile
            for _ in range(4): _ = model5(dummy_input)
            
            start = time.time()
            n_tokens = 50 # Probamos con más tokens para ver estabilidad
            for _ in range(n_tokens):
                _ = model5(dummy_input)
            tps5 = n_tokens / (time.time() - start)
        print(f"V8.5 Tok/s: {tps5:8.2f}")
    except Exception as e:
        print(f"Error en V8.5: {e}")
        tps5 = 0.001
    
    speedup = (tps5 / tps4 - 1) * 100
    print("-" * 70)
    print(f"MEJORA NATIVA: {speedup:+.1f}%")
    
    if tps5 > tps4:
        print("¡INCREÍBLE! El motor C++ ha superado al baseline de Python.")
        print("Ahora la FWHT ha dejado de ser el cuello de botella principal.")
    else:
        print("Aviso: Si no hay mejora, revisa si OpenMP se ha enlazado correctamente.")

if __name__ == "__main__":
    benchmark_native()
