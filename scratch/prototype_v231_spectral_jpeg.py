import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_dct_matrix(N, device):
    """Genera la matriz de transformación DCT-II."""
    n = torch.arange(N, device=device).view(1, -1)
    k = torch.arange(N, device=device).view(-1, 1)
    M = torch.cos(np.pi * k * (2 * n + 1) / (2 * N))
    M[0, :] *= 1 / np.sqrt(2)
    M *= np.sqrt(2 / N)
    return M

def dct2(W):
    """DCT 2D nativa en PyTorch."""
    h, w = W.shape
    M_h = get_dct_matrix(h, W.device)
    M_w = get_dct_matrix(w, W.device)
    return M_h @ W @ M_w.t()

def idct2(W_dct):
    """IDCT 2D nativa en PyTorch (M es ortogonal, M_inv = M.t())."""
    h, w = W_dct.shape
    M_h = get_dct_matrix(h, W_dct.device)
    M_w = get_dct_matrix(w, W_dct.device)
    return M_h.t() @ W_dct @ M_w

def compress_spectral_jpeg(W, keep_ratio=0.25):
    """
    Comprime la matriz manteniendo solo el 'keep_ratio' de coeficientes DCT.
    Usa una máscara de frecuencias bajas (esquina superior izquierda).
    """
    h, w = W.shape
    W_dct = dct2(W)
    
    # Creamos una máscara para mantener solo la esquina superior izquierda
    mask = torch.zeros_like(W_dct)
    h_limit = int(h * np.sqrt(keep_ratio))
    w_limit = int(w * np.sqrt(keep_ratio))
    
    mask[:h_limit, :w_limit] = 1.0
    
    W_dct_compressed = W_dct * mask
    
    # Reconstruir
    W_rec = idct2(W_dct_compressed)
    
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
    keep_ratio = 0.75 # Subimos al 75% (Compresión ligera para buscar estabilidad)
    
    print(f"\n--- Experimento v231: GPT-2 Spectral JPEG (High Quality) ---")
    print(f" Ratio de coeficientes a mantener: {keep_ratio*100:.1f}%")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    eval_text = "The secret of intelligence is in the structure of information."
    
    # Check de Sanidad: ¿Es la DCT reversible?
    W_test = model.transformer.h[0].mlp.c_fc.weight.data.t()[:64, :64]
    W_rec_test = idct2(dct2(W_test))
    diff = torch.abs(W_test - W_rec_test).max().item()
    print(f"Sanity Check (DCT Reversibility Error): {diff:.2e}")
    
    # 1. Baseline
    l0, p0 = evaluate_model(model, tokenizer, eval_text, device)
    print(f"Baseline PPL: {p0:.4f}")
    
    # 2. Bucle de Compresión
    print("\nComprimiendo capas (DCT-2D)...")
    total_weights = 0
    start_time = time.time()
    for name, module in model.named_modules():
        if any(x in name for x in ["c_attn", "c_fc", "c_proj"]):
            if hasattr(module, "weight"):
                W_orig = module.weight.data.t().clone()
                orig_std = W_orig.std()
                
                # Comprimir 
                W_rec = compress_spectral_jpeg(W_orig, keep_ratio=keep_ratio)
                
                # Rescalar
                if W_rec.std() > 0:
                    W_rec = W_rec * (orig_std / W_rec.std())
                
                module.weight.data = W_rec.t().contiguous()
                total_weights += W_orig.numel()
    
    # 3. Métricas
    size_fp32 = (total_weights * 4) / (1024**2)
    size_jpeg = (total_weights * keep_ratio * 4) / (1024**2)
    print(f"\n--- Métricas ---")
    print(f" Tamaño FP32: {size_fp32:.2f} MB")
    print(f" Tamaño JPEG: {size_jpeg:.2f} MB")
    
    # 4. Evaluación
    lq, pq = evaluate_model(model, tokenizer, eval_text, device)
    print(f"Post-JPEG PPL: {pq:.4f} (Delta: {pq - p0:+.4f})")
    
    # 5. Generación
    prompt = "The secret of intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print(f"\nGeneración: {tokenizer.decode(out[0])}")

if __name__ == "__main__":
    main()
