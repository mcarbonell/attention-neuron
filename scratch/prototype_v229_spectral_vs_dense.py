import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time
import os

# Configuración de Hardware (según reglas del USER)
# Nota: Si se requiere GPU con DirectML, el USER debe ejecutar con el venv correspondiente.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

def bit_reverse(n, bits):
    res = 0
    for i in range(bits):
        res <<= 1
        res |= (n & 1)
        n >>= 1
    return res

def gray_to_binary(n):
    mask = n >> 1
    while mask != 0:
        n = n ^ mask
        mask = mask >> 1
    return n

def get_sequency_indices(N):
    """Calcula el mapeo de orden Natural a orden de Secuencialidad."""
    num_bits = int(np.log2(N))
    indices = []
    for i in range(N):
        # El orden de secuencialidad en Walsh-Hadamard se obtiene 
        # mediante la inversión de bits del código Gray de i
        rev_i = bit_reverse(i, num_bits)
        seq_i = gray_to_binary(rev_i)
        indices.append(seq_i)
    return torch.tensor(indices)

def fwht(x, sequency_order=True):
    """
    Fast Walsh-Hadamard Transform con opción de orden de Secuencialidad.
    """
    orig_shape = x.shape
    N = x.shape[-1]
    x = x.clone()
    h = 1
    while h < N:
        x = x.view(-1, N // (h * 2), h, 2)
        a, b = x[..., 0], x[..., 1]
        x = torch.stack([a + b, a - b], dim=-1)
        h *= 2
    
    x = x.view(orig_shape) / np.sqrt(N)
    
    if sequency_order:
        # Reordenar de Natural a Sequency
        indices = get_sequency_indices(N).to(x.device)
        # Aplicamos el reordenamiento en la última dimensión
        x = torch.index_select(x, -1, indices)
    return x

def ifwht(x, sequency_order=True):
    """Inversa de la FWHT (es auto-inversa, solo hay que deshacer el orden)."""
    if sequency_order:
        N = x.shape[-1]
        indices = get_sequency_indices(N).to(x.device)
        # Creamos el mapeo inverso
        inv_indices = torch.zeros_like(indices)
        inv_indices[indices] = torch.arange(N, device=x.device)
        x = torch.index_select(x, -1, inv_indices)
    
    # La FWHT es su propia inversa (con el escalado que ya tiene fwht)
    return fwht(x, sequency_order=False) 

def fwht_2d(W, sequency_order=True):
    """Aplica FWHT 2D con orden de secuencialidad opcional."""
    W = fwht(W, sequency_order=sequency_order)
    W = fwht(W.t(), sequency_order=sequency_order).t()
    return W

def ifwht_2d(W, sequency_order=True):
    """Aplica FWHT 2D inversa."""
    W = ifwht(W.t(), sequency_order=sequency_order).t()
    W = ifwht(W, sequency_order=sequency_order)
    return W

def uniform_quantization(W, bits=4):
    """Cuantización uniforme simple de N bits (Round-to-Nearest)."""
    if bits is None: return W
    levels = 2**bits
    
    w_min, w_max = W.min(), W.max()
    scale = (w_max - w_min) / (levels - 1)
    
    # Cuantizar
    W_q = torch.round((W - w_min) / scale)
    # Reconstruir
    W_rec = W_q * scale + w_min
    return W_rec

def evaluate_model(model, tokenizer, text, device):
    """Calcula la pérdida (loss) y perplejidad en un texto dado."""
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return loss.item(), perplexity.item()

def hierarchical_spectral_quantization(W_spec, low_bits=8, high_bits=2, ratio=0.1):
    """
    Cuantización jerárquica: las frecuencias bajas (esquina superior izquierda) 
    tienen más bits que las altas.
    """
    N = W_spec.shape[0]
    W_rec = torch.zeros_like(W_spec)
    
    # Definimos la zona de "baja frecuencia" (el core de la señal)
    core_size = int(N * ratio)
    
    # Zona Core (Alta precisión)
    core = W_spec[:core_size, :core_size]
    W_rec[:core_size, :core_size] = uniform_quantization(core, bits=low_bits)
    
    # Zona Resto (Baja precisión)
    # Esto es una simplificación, lo ideal sería por bloques, pero para el test sirve
    mask = torch.ones_like(W_spec, dtype=torch.bool)
    mask[:core_size, :core_size] = False
    
    W_rec[mask] = uniform_quantization(W_spec[mask], bits=high_bits)
    
    return W_rec

def analyze_outliers(W_orig, W_rec):
    """Analiza cuánto han cambiado los pesos con mayor magnitud."""
    threshold = 3 * W_orig.std()
    mask = torch.abs(W_orig) > threshold
    outliers_orig = W_orig[mask]
    outliers_rec = W_rec[mask]
    
    mse_global = torch.mean((W_orig - W_rec)**2).item()
    
    if len(outliers_orig) == 0:
        return mse_global, 0.0, 0.0, 0.0, 0.0
    
    mse_outliers = torch.mean((outliers_orig - outliers_rec)**2).item()
    strength_orig = torch.mean(torch.abs(outliers_orig)).item()
    strength_rec = torch.mean(torch.abs(outliers_rec)).item()
    
    # Proporción de outliers preservados (dentro de un 10% de su valor)
    preserved = torch.sum(torch.abs(outliers_orig - outliers_rec) < 0.1 * torch.abs(outliers_orig)).item()
    preservation_rate = preserved / len(outliers_orig)
    
    return mse_global, mse_outliers, strength_orig, strength_rec, preservation_rate

def main():
    model_name = "gpt2"
    print(f"\n--- Iniciando Experimento v229: Hierarchical Spectral vs Spatial ---")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eval_text = (
        "The Walsh–Hadamard transform is an example of a generalized class of Fourier transforms. "
        "It performs an orthogonal, symmetric, self-inverse, linear operation on 2n real numbers. "
        "In quantum information theory, the Hadamard transform is used to create superposition states. "
        "The transformation is often used in data compression and digital signal processing."
    )
    
    # 1. Baseline FP32
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    loss_0, ppl_0 = evaluate_model(model, tokenizer, eval_text, device)
    print(f"\n[1. BASELINE FP32]")
    print(f"  Loss: {loss_0:.4f} | Perplexity: {ppl_0:.4f}")
    
    target_layer = model.transformer.h[6].mlp.c_fc
    W_orig = target_layer.weight.data.t().clone()
    h, w = W_orig.shape
    
    # 2. Experimento Espacial (INT4 tradicional)
    print(f"\n[2. CUANTIZACIÓN ESPACIAL (4-bit RTN)]")
    W_spatial_q = uniform_quantization(W_orig, bits=4)
    mse_g_s, mse_o_s, str_o_s, str_r_s, pres_s = analyze_outliers(W_orig, W_spatial_q)
    
    target_layer.weight.data = W_spatial_q.t().contiguous()
    loss_s, ppl_s = evaluate_model(model, tokenizer, eval_text, device)
    
    print(f"  MSE Global:   {mse_g_s:.8f}")
    print(f"  MSE Outliers: {mse_o_s:.8f}")
    print(f"  Preservation Rate (>90% accuracy): {pres_s*100:.1f}%")
    print(f"  Model Perplexity: {ppl_s:.4f} (Delta: {ppl_s - ppl_0:+.4f})")
    
    # 3. Experimento Espectral JERÁRQUICO (Tu enfoque optimizado)
    print(f"\n[3. CUANTIZACIÓN ESPECTRAL JERÁRQUICA (8-bit Core / 4-bit Detalle)]")
    # Nota: El promedio de bits será cercano a 4.8 bits
    
    N = 2**int(np.ceil(np.log2(max(h, w))))
    W_padded = torch.zeros((N, N), device=device)
    W_padded[:h, :w] = W_orig
    
    W_spec = fwht_2d(W_padded, sequency_order=True)
    # Aplicamos jerarquía: 20% del espectro a 8 bits, resto a 4 bits
    W_spec_q = hierarchical_spectral_quantization(W_spec, low_bits=8, high_bits=4, ratio=0.2)
    
    W_spectral_rec_full = ifwht_2d(W_spec_q, sequency_order=True)
    W_spectral_rec = W_spectral_rec_full[:h, :w]
    
    mse_g_e, mse_o_e, str_o_e, str_r_e, pres_e = analyze_outliers(W_orig, W_spectral_rec)
    
    target_layer.weight.data = W_spectral_rec.t().contiguous()
    loss_e, ppl_e = evaluate_model(model, tokenizer, eval_text, device)
    
    print(f"  MSE Global:   {mse_g_e:.8f}")
    print(f"  MSE Outliers: {mse_o_e:.8f}")
    print(f"  Preservation Rate (>90% accuracy): {pres_e*100:.1f}%")
    print(f"  Model Perplexity: {ppl_e:.4f} (Delta: {ppl_e - ppl_0:+.4f})")
    
    # 4. Generación final
    prompt = "The secret of intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_out = model.generate(**inputs, max_new_tokens=15, do_sample=False)
    print(f"\n[GENERACIÓN FINAL - MODELO ESPECTRAL JERÁRQUICO]")
    print(f"  Salida: {tokenizer.decode(gen_out[0], skip_special_tokens=True)}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
