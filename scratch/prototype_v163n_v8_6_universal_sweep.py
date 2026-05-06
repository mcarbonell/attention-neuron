import torch
import time
import os
import sys
import json

# Añadir path para importar el modelo
script_dir = os.path.dirname(os.path.abspath(__file__))
tiny_thinker_path = os.path.abspath(os.path.join(script_dir, "../../../tiny-thinker"))
sys.path.append(tiny_thinker_path)

try:
    from model.model_spectral_v8_6_universal import SpectralThinkerV8_6, SpectralArgs, get_best_device
except ImportError as e:
    print(f"Error importando modelos: {e}")
    sys.exit(1)

def run_sweep():
    print("=== BARRIDO DE ESCALA UNIVERSAL V8.6 (CPU vs GPU) ===")
    
    # Reducimos un poco el espacio para que no sea eterno, pero mantenemos el foco en expertos
    experts_list = [32, 256, 1024]
    dim_list = [8192, 32768]
    layers_list = [8, 24]
    
    gpu_device = get_best_device()
    has_gpu = gpu_device.type != "cpu"
    
    devices = [torch.device("cpu")]
    if has_gpu:
        devices.append(gpu_device)
        print(f"Hardware detectado: Ryzen 7 (CPU) + GPU ({gpu_device})")
    else:
        print("Hardware detectado: Ryzen 7 (CPU)")

    results = []
    
    print("-" * 95)
    print(f"{'Exp':<6} | {'Dim':<8} | {'Lay':<4} | {'Params':<8} | {'Dev':<10} | {'Tok/s':<10} | {'PEI':<6}")
    print("-" * 95)

    for experts in experts_list:
        for dim in dim_list:
            for layers in layers_list:
                args = SpectralArgs(dim=dim, n_layers=layers, num_experts=experts)
                
                for device in devices:
                    try:
                        # Inicializar modelo
                        model = SpectralThinkerV8_6(args).to_device(device).half()
                        
                        # Contar parámetros reales
                        total_params = sum(p.numel() for p in model.parameters())
                        params_str = f"{total_params/1e6:.1f}M"
                        
                        dummy_input = torch.randint(0, 32768, (1, 1)).to(device)
                        
                        # Warmup
                        with torch.no_grad():
                            for _ in range(5): _ = model(dummy_input)
                            
                            start = time.time()
                            n_iter = 20 if device.type == "cpu" else 50
                            for _ in range(n_iter): 
                                _ = model(dummy_input)
                            tps = n_iter / (time.time() - start)
                            
                        # Calcular PEI (suponiendo un accuracy base de 1.0 para comparativa de escala)
                        import math
                        pei = 1.0 / math.log10(total_params + 1) * 100 # Normalizado para el benchmark
                        
                        dev_name = "GPU" if device.type != "cpu" else "CPU"
                        print(f"{experts:<6} | {dim:<8} | {layers:<4} | {params_str:<8} | {dev_name:<10} | {tps:8.2f} | {pei:.1f}")
                        
                        results.append({
                            "experts": experts,
                            "dim": dim,
                            "layers": layers,
                            "params": total_params,
                            "device": dev_name,
                            "tps": tps,
                            "pei": pei
                        })
                        
                        # Limpieza agresiva de memoria
                        del model
                        if has_gpu:
                            if device.type != "cpu":
                                # Dependiendo del backend, vaciar caché
                                try: torch.cuda.empty_cache()
                                except: pass
                        
                    except Exception as e:
                        print(f"\nError en config (Exp:{experts}, Dim:{dim}, Lay:{layers}, Dev:{device}): {e}")
                        # No salir, continuar con el siguiente

    # Asegurar directorio de resultados
    os.makedirs("results/summary", exist_ok=True)
    with open("results/summary/sweep_v8_6_universal.json", "w") as f:
        json.dump(results, f, indent=4)
    print("-" * 95)
    print(f"Barrido completado. Resultados guardados en results/summary/sweep_v8_6_universal.json")

if __name__ == "__main__":
    run_sweep()
