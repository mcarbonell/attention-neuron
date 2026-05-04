import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

ROBUST_TEXT = """
The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. It formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud. The vast majority of the system's mass is in the Sun, with most of the remaining mass contained in Jupiter. The four inner system planets—Mercury, Venus, Earth and Mars—are terrestrial planets, being primarily composed of rock and metal.

Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term AI is also used to describe a property of machines which mimic cognitive functions that humans associate with the human mind.

The Industrial Revolution was a period of global economic transition towards more efficient and stable manufacturing processes that succeeded the Agricultural Revolution. This transition included going from hand production methods to machines, new chemical manufacturing and iron production processes, the increasing use of steam power and water power, the development of machine tools and the rise of the mechanized factory system.
"""

def get_walsh_matrix(N, dtype):
    if N == 1: return torch.tensor([[1.0]], dtype=dtype)
    H_prev = get_walsh_matrix(N // 2, dtype)
    top = torch.cat([H_prev, H_prev], dim=1)
    bottom = torch.cat([H_prev, -H_prev], dim=1)
    return torch.cat([top, bottom], dim=0)

# Caché para evitar recálculos costosos
WALSH_CACHE = {}

def get_walsh_matrix_sequency_cached(N, dtype):
    if (N, dtype) in WALSH_CACHE:
        return WALSH_CACHE[(N, dtype)]
    
    print(f"  [MATH] Generando matriz Walsh {N}x{N} (Vectorizado)...")
    H = get_walsh_matrix(N, dtype)
    
    # Vectorización: Contar cruces de signo para todas las filas a la vez
    # Multiplicamos elementos adyacentes; si el producto es negativo, hay un cruce.
    crossings = (H[:, :-1] * H[:, 1:] < 0).sum(dim=1)
    
    # Ordenar por número de cruces (sequency order)
    indices = torch.argsort(crossings)
    H_seq = H[indices]
    
    WALSH_CACHE[(N, dtype)] = H_seq
    return H_seq

def apply_spectral_pruning_fp16(model, keep_ratio):
    dtype = torch.float16
    print("Iniciando poda espectral optimizada...")
    for name, module in model.named_modules():
        if any(x in name for x in ["c_attn", "c_fc", "c_proj"]):
            if hasattr(module, "weight"):
                W_orig = module.weight.data.t().clone().to(dtype)
                orig_std = W_orig.std()
                h, w = W_orig.shape
                
                N = 2**int(np.ceil(np.log2(max(h, w))))
                # Reutilizar matriz de la caché
                H = get_walsh_matrix_sequency_cached(N, dtype).to(device)
                
                W_padded = torch.zeros((N, N), device=device, dtype=dtype)
                W_padded[:h, :w] = W_orig
                
                # Transformar
                spectrum = torch.matmul(H, torch.matmul(W_padded, H.t()))
                
                # Top-K Pruning
                flat_spectrum = spectrum.flatten()
                k = int(flat_spectrum.numel() * keep_ratio)
                values, _ = torch.topk(torch.abs(flat_spectrum), k)
                threshold = values[-1]
                mask = torch.abs(spectrum) >= threshold
                spectrum = spectrum * mask
                
                # Inversa
                W_rec = (torch.matmul(H.t(), torch.matmul(spectrum, H)) / (N * N))[:h, :w]
                
                # Rescalar
                if W_rec.std() > 0:
                    W_rec = W_rec * (orig_std / W_rec.std())
                
                module.weight.data = W_rec.t().contiguous().to(module.weight.dtype)
                print(f"  [OK] {name}")

def evaluate_robust_ppl(model, tokenizer, text, device):
    model.eval()
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        ppl = torch.exp(loss)
    return loss.item(), ppl.item()

def main():
    model_name = "gpt2"
    keep_ratio = 0.70
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"\n--- Experimento v238: SPECTRAL FP16 COMPRESSION (Ultimate) ---")
    
    # 1. Baseline FP32
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    _, ppl32 = evaluate_robust_ppl(model, tokenizer, ROBUST_TEXT, device)
    print(f"Baseline FP32: {ppl32:.4f}")
    
    # 2. Paso a FP16 + Poda Espectral
    print(f"Casting a FP16 + Poda Espectral (Ratio {keep_ratio:.2f})...")
    model = model.half()
    apply_spectral_pruning_fp16(model, keep_ratio)
    
    _, ppl_final = evaluate_robust_ppl(model, tokenizer, ROBUST_TEXT, device)
    
    print("\n--- RESULTADOS FINALES ---")
    print(f"Configuración      | Perplexity | Tamaño Est.")
    print(f"Original (FP32)    | {ppl32:<10.4f} | 324 MB")
    print(f"Spectral FP16 (0.7)| {ppl_final:<10.4f} | 113 MB")
    
    print(f"\nCompresión Total: {324/113.4:.2f}x")
    print(f"Delta PPL Total:  {ppl_final - ppl32:+.4f}")

if __name__ == "__main__":
    main()
