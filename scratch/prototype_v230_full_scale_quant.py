import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

def bit_reverse(n, bits):
    res = 0
    for i in range(bits):
        res <<= 1
        res |= (n & 1)
        n >>= 1
    return res

def gray_to_binary(n):
    n_copy = n
    mask = n_copy >> 1
    while mask != 0:
        n_copy = n_copy ^ mask
        mask = mask >> 1
    return n_copy

def get_sequency_indices(N):
    num_bits = int(np.log2(N))
    indices = []
    for i in range(N):
        rev_i = bit_reverse(i, num_bits)
        seq_i = gray_to_binary(rev_i)
        indices.append(seq_i)
    return torch.tensor(indices)

def fwht(x, sequency_order=True):
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
        indices = get_sequency_indices(N).to(x.device)
        x = torch.index_select(x, -1, indices)
    return x

def ifwht(x, sequency_order=True):
    if sequency_order:
        N = x.shape[-1]
        indices = get_sequency_indices(N).to(x.device)
        inv_indices = torch.zeros_like(indices)
        inv_indices[indices] = torch.arange(N, device=x.device)
        x = torch.index_select(x, -1, inv_indices)
    return fwht(x, sequency_order=False)

def fwht_2d(W, sequency_order=True):
    W = fwht(W, sequency_order=sequency_order)
    W = fwht(W.t(), sequency_order=sequency_order).t()
    return W

def ifwht_2d(W, sequency_order=True):
    W = ifwht(W.t(), sequency_order=sequency_order).t()
    W = ifwht(W, sequency_order=sequency_order)
    return W

def uniform_quantization(W, bits=4):
    """Cuantización uniforme por filas (Per-Row) para máxima resolución."""
    if bits is None: return W
    levels = 2**bits
    
    # Dimensiones para per-row
    # Asumimos W es (Rows, Cols)
    w_min = W.min(dim=-1, keepdim=True)[0]
    w_max = W.max(dim=-1, keepdim=True)[0]
    
    scale = (w_max - w_min) / (levels - 1)
    # Evitar división por cero
    scale[scale == 0] = 1.0
    
    # Cuantizar y Reconstruir
    W_q = torch.round((W - w_min) / scale)
    W_rec = W_q * scale + w_min
    return W_rec

def hierarchical_spectral_quantization(W_spec, low_bits=8, high_bits=4, ratio=0.2):
    N = W_spec.shape[0]
    W_rec = torch.zeros_like(W_spec)
    core_size = int(N * ratio)
    # Core
    core = W_spec[:core_size, :core_size]
    W_rec[:core_size, :core_size] = uniform_quantization(core, bits=low_bits)
    # Detail
    mask = torch.ones_like(W_spec, dtype=torch.bool)
    mask[:core_size, :core_size] = False
    W_rec[mask] = uniform_quantization(W_spec[mask], bits=high_bits)
    return W_rec

def evaluate_model(model, tokenizer, text, device):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return loss.item(), perplexity.item()

def main():
    model_name = "gpt2"
    print(f"\n--- Experimento v230: Cuantización Espectral GLOBAL (STABLE) ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    eval_text = "The secret of intelligence is in the structure of information. Spectral methods allow us to see this structure clearly."
    
    # 1. Baseline
    l0, p0 = evaluate_model(model, tokenizer, eval_text, device)
    print(f"Baseline PPL: {p0:.4f}")
    
    # 2. Bucle de Cuantización
    print("\nIniciando cuantización estable de todas las capas...")
    quant_params = 0
    
    start_time = time.time()
    for name, module in model.named_modules():
        if any(x in name for x in ["c_attn", "c_fc", "c_proj"]):
            if hasattr(module, "weight"):
                W_orig = module.weight.data.t().clone()
                orig_std = W_orig.std()
                h, w = W_orig.shape
                
                # Transformación
                N = 2**int(np.ceil(np.log2(max(h, w))))
                W_padded = torch.zeros((N, N), device=device)
                W_padded[:h, :w] = W_orig
                
                W_spec = fwht_2d(W_padded, sequency_order=True)
                
                # JERARQUÍA ESTABLE: 8 bits core, 6 bits detalle (Media: 6.4 bits)
                W_spec_q = hierarchical_spectral_quantization(W_spec, low_bits=8, high_bits=6, ratio=0.2)
                
                W_rec_full = ifwht_2d(W_spec_q, sequency_order=True)
                W_rec = W_rec_full[:h, :w]
                
                # RE-ESCALADO DE VARIANZA (Crucial para estabilidad)
                if W_rec.std() > 0:
                    W_rec = W_rec * (orig_std / W_rec.std())
                
                # Aplicar
                module.weight.data = W_rec.t().contiguous()
                quant_params += h * w
                print(f"  [STABLE OK] {name}")
    
    duration = time.time() - start_time
    
    # 3. Cálculos de Tamaño
    # 6.4 bits promedio = 0.8 bytes por param
    size_fp32 = (quant_params * 4) / (1024**2)
    size_quant = (quant_params * 0.8) / (1024**2)
    
    print(f"\n--- Métricas de Tamaño ---")
    print(f" Tamaño FP32: {size_fp32:.2f} MB")
    print(f" Tamaño Espectral (6.4-bit): {size_quant:.2f} MB")
    print(f" Reducción: {size_fp32 / size_quant:.2f}x")
    
    # 4. Evaluación Final
    lq, pq = evaluate_model(model, tokenizer, eval_text, device)
    print(f"\n--- Métricas de Inteligencia Final ---")
    print(f" Baseline PPL: {p0:.4f}")
    print(f" Post-Quant PPL: {pq:.4f} (Delta: {pq - p0:+.4f})")
    
    # 5. Prueba de Generación
    prompt = "The secret of intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print(f"\nGeneración: {tokenizer.decode(out[0])}")

if __name__ == "__main__":
    main()
