import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_walsh_matrix(N):
    if N == 1: return torch.tensor([[1.0]])
    H_prev = get_walsh_matrix(N // 2)
    top = torch.cat([H_prev, H_prev], dim=1)
    bottom = torch.cat([H_prev, -H_prev], dim=1)
    return torch.cat([top, bottom], dim=0)

def get_walsh_matrix_sequency(N):
    H = get_walsh_matrix(N)
    crossings = []
    for i in range(N):
        row = H[i]
        num_crossings = (row[:-1] * row[1:] < 0).sum().item()
        crossings.append((num_crossings, i))
    crossings.sort()
    indices = [idx for _, idx in crossings]
    return H[indices]

def walsh_2d_transform(image, H):
    return torch.matmul(H, torch.matmul(image, H.t()))

def iwalsh_2d_transform(coeffs, H):
    N = H.shape[0]
    return torch.matmul(H.T, torch.matmul(coeffs, H)) / (N * N)

def compress_spectral_topk(W, keep_ratio=0.25):
    """
    Poda Espectral Adaptativa: Mantiene el Top-K de coeficientes por magnitud.
    """
    h, w = W.shape
    N = 2**int(np.ceil(np.log2(max(h, w))))
    
    W_padded = torch.zeros((N, N), device=device)
    W_padded[:h, :w] = W
    
    H = get_walsh_matrix_sequency(N).to(device)
    
    # 1. Transformar
    spectrum = walsh_2d_transform(W_padded, H)
    
    # 2. Top-K Pruning por magnitud
    flat_spectrum = spectrum.flatten()
    k = int(flat_spectrum.numel() * keep_ratio)
    
    # Buscamos el umbral del top-k
    values, _ = torch.topk(torch.abs(flat_spectrum), k)
    threshold = values[-1]
    
    # Aplicar máscara
    mask = torch.abs(spectrum) >= threshold
    compressed_spectrum = spectrum * mask
    
    # 3. Inversa
    W_rec_full = iwalsh_2d_transform(compressed_spectrum, H)
    W_rec = W_rec_full[:h, :w]
    
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
    keep_ratio = 0.25 # Compresión 4x (Mantenemos solo el 25% de los datos)
    
    print(f"\n--- Experimento v233: GPT-2 SPECTRAL TOP-K PRUNING ---")
    print(f" Ratio de coeficientes mantenidos: {keep_ratio*100:.1f}%")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    eval_text = "The secret of intelligence is in the structure of information."
    
    # 1. Baseline
    l0, p0 = evaluate_model(model, tokenizer, eval_text, device)
    print(f"Baseline PPL: {p0:.4f}")
    
    # 2. Bucle de Compresión
    print("\nComprimiendo capas con Spectral Top-K...")
    total_weights = 0
    start_time = time.time()
    for name, module in model.named_modules():
        if any(x in name for x in ["c_attn", "c_fc", "c_proj"]):
            if hasattr(module, "weight"):
                W_orig = module.weight.data.t().clone()
                orig_std = W_orig.std()
                
                # Comprimir 
                W_rec = compress_spectral_topk(W_orig, keep_ratio=keep_ratio)
                
                # Rescalar varianza
                if W_rec.std() > 0:
                    W_rec = W_rec * (orig_std / W_rec.std())
                
                module.weight.data = W_rec.t().contiguous()
                total_weights += W_orig.numel()
                print(f"  [TOP-K OK] {name}")
    
    duration = time.time() - start_time
    print(f"\nCompresión completada en {duration:.2f}s")
    
    # 3. Métricas
    size_fp32 = (total_weights * 4) / (1024**2)
    size_sparse = (total_weights * keep_ratio * 4) / (1024**2)
    print(f"\n--- Métricas ---")
    print(f" Tamaño Original: {size_fp32:.2f} MB")
    print(f" Tamaño Comprimido (Top-K): {size_sparse:.2f} MB")
    
    # 4. Evaluación Final
    lq, pq = evaluate_model(model, tokenizer, eval_text, device)
    print(f"Post-TopK PPL: {pq:.4f} (Delta: {pq - p0:+.4f})")
    
    # 5. Prueba de Generación
    prompt = "The secret of intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print(f"\nGeneración: {tokenizer.decode(out[0])}")

if __name__ == "__main__":
    main()
