import torch
import time
import os
import sys

# Añadir path para importar el modelo de forma robusta
script_dir = os.path.dirname(os.path.abspath(__file__))
tiny_thinker_path = os.path.abspath(os.path.join(script_dir, "../../../tiny-thinker"))
sys.path.append(tiny_thinker_path)

try:
    from model.model_spectral_v8_6_universal import SpectralThinkerV8_6, SpectralArgs
except ImportError as e:
    print(f"Error importando modelos: {e}")
    sys.exit(1)

def benchmark_universal():
    print("=== BENCHMARK UNIVERSAL V8.6: CPU vs GPU (DirectML/CUDA) ===")
    
    # Configuración de prueba de alta resolución
    config = {"experts": 128, "layers": 16, "dim": 32768}
    args = SpectralArgs(dim=config["dim"], n_layers=config["layers"], num_experts=config["experts"])
    
    print(f"Perfil: {config['layers']} capas, {config['dim']} dim, {config['experts']} expertos")
    print("-" * 75)

    # 1. Probar en CPU (Native Pulse Engine)
    print(f"\n[Fase 1] Midiendo CPU (Motor Nativo V8.5)...")
    try:
        model_cpu = SpectralThinkerV8_6(args).to_device(torch.device("cpu")).half()
        dummy_input = torch.randint(0, 32768, (1, 1))
        
        with torch.no_grad():
            for _ in range(5): _ = model_cpu(dummy_input) # Warmup
            start = time.time()
            n_iter = 20
            for _ in range(n_iter): 
                _ = model_cpu(dummy_input)
            tps_cpu = n_iter / (time.time() - start)
        print(f"CPU Speed: {tps_cpu:8.2f} Tok/s")
    except Exception as e:
        print(f"Error en test CPU: {e}")
        tps_cpu = 0.1

    # 2. Probar en GPU (AMD Radeon 780M via DirectML o NVIDIA via CUDA)
    print(f"\n[Fase 2] Midiendo GPU (Vectorized Engine V8.6)...")
    try:
        # Detecta automáticamente DirectML en tu venv_gpu
        model_gpu = SpectralThinkerV8_6(args).to_device().half() 
        gpu_device = model_gpu._current_device
        
        if gpu_device.type == "cpu":
            print(">>> AVISO: No se detectó aceleración GPU. Asegúrate de ejecutar esto con:")
            print(">>> C:/Users/mrcm_/Local/proj/ajedrez/neural-tablebases/venv_gpu/Scripts/python.exe")
            return

        dummy_input_gpu = dummy_input.to(gpu_device)
        
        with torch.no_grad():
            print(f"Calentando GPU ({gpu_device})...")
            for _ in range(10): _ = model_gpu(dummy_input_gpu)
            
            # Test Batch 1 (Latencia pura)
            start = time.time()
            n_tokens = 50
            for _ in range(n_tokens): 
                _ = model_gpu(dummy_input_gpu)
            tps_gpu_b1 = n_tokens / (time.time() - start)
            print(f"GPU Speed (Batch 1): {tps_gpu_b1:8.2f} Tok/s")
            
            # Test Batch 16 (Rendimiento en paralelo / Throughput)
            batch_size = 16
            print(f"Midiendo Throughput (Batch {batch_size})...")
            input_b16 = torch.randint(0, 32768, (batch_size, 1)).to(gpu_device)
            start = time.time()
            n_iter_b16 = 20
            for _ in range(n_iter_b16): 
                _ = model_gpu(input_b16)
            tps_gpu_b16 = (n_iter_b16 * batch_size) / (time.time() - start)
            print(f"GPU Speed (Batch {batch_size}): {tps_gpu_b16:8.2f} Tok/s")

            # Comparativa final
            gain = (tps_gpu_b1 / tps_cpu - 1) * 100
            print("-" * 75)
            print(f"VENTAJA GPU (B1): {gain:+.1f}%")
            print(f"VENTAJA GPU (B16): {(tps_gpu_b16 / tps_cpu - 1) * 100:+.1f}%")

    except Exception as e:
        print(f"Error en el test de GPU: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    benchmark_universal()
