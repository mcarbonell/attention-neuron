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
    return torch.matmul(H.t(), torch.matmul(coeffs, H)) / (N * N)

def apply_spectral_pruning(model, keep_ratio):
    """Aplica la poda Top-K a todas las capas pesadas del modelo."""
    for name, module in model.named_modules():
        if any(x in name for x in ["c_attn", "c_fc", "c_proj"]):
            if hasattr(module, "weight"):
                W_orig = module.weight.data.t().clone()
                orig_std = W_orig.std()
                h, w = W_orig.shape
                
                N = 2**int(np.ceil(np.log2(max(h, w))))
                W_padded = torch.zeros((N, N), device=device)
                W_padded[:h, :w] = W_orig
                H = get_walsh_matrix_sequency(N).to(device)
                
                # Transformar
                spectrum = walsh_2d_transform(W_padded, H)
                
                # Top-K
                if keep_ratio < 1.0:
                    flat_spectrum = spectrum.flatten()
                    k = int(flat_spectrum.numel() * keep_ratio)
                    values, _ = torch.topk(torch.abs(flat_spectrum), k)
                    threshold = values[-1]
                    mask = torch.abs(spectrum) >= threshold
                    spectrum = spectrum * mask
                
                # Inversa
                W_rec = iwalsh_2d_transform(spectrum, H)[:h, :w]
                
                # Rescalar varianza
                if W_rec.std() > 0:
                    W_rec = W_rec * (orig_std / W_rec.std())
                
                module.weight.data = W_rec.t().contiguous()

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
    ratios = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    results = []
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eval_text = "The secret of intelligence is in the structure of information."
    
    print(f"--- Experimento v235: BARRIDO DE COMPRESIÓN ESPECTRAL ---")
    
    for r in ratios:
        # Cargamos el modelo limpio en cada iteración
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        if r == 1.0:
            l, p = evaluate_model(model, tokenizer, eval_text, device)
            print(f"Ratio 1.00 (Baseline) | PPL: {p:.4f}")
            results.append((r, p))
            continue
            
        print(f"Aplicando Ratio {r:.2f}...", end=" ", flush=True)
        start = time.time()
        apply_spectral_pruning(model, r)
        l, p = evaluate_model(model, tokenizer, eval_text, device)
        end = time.time()
        
        print(f"PPL: {p:.4f} | Tiempo: {end-start:.1f}s")
        results.append((r, p))
    
    print("\n--- RESUMEN DEL BARRIDO ---")
    print(f"{'Ratio':<10} | {'PPL':<10} | {'Delta':<10}")
    base_ppl = results[0][1]
    for r, p in results:
        print(f"{r:<10.2f} | {p:<10.4f} | {p - base_ppl:<+10.4f}")

if __name__ == "__main__":
    main()
