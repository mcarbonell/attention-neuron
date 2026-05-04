import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

# Textos de referencia para evaluación robusta (Diversos dominios)
ROBUST_TEXT = """
The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. It formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud. The vast majority of the system's mass is in the Sun, with most of the remaining mass contained in Jupiter. The four inner system planets—Mercury, Venus, Earth and Mars—are terrestrial planets, being primarily composed of rock and metal.

Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term AI is also used to describe a property of machines which mimic cognitive functions that humans associate with the human mind.

The Industrial Revolution was a period of global economic transition towards more efficient and stable manufacturing processes that succeeded the Agricultural Revolution. This transition included going from hand production methods to machines, new chemical manufacturing and iron production processes, the increasing use of steam power and water power, the development of machine tools and the rise of the mechanized factory system.
"""

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
                spectrum = walsh_2d_transform(W_padded, H)
                
                if keep_ratio < 1.0:
                    flat_spectrum = spectrum.flatten()
                    k = int(flat_spectrum.numel() * keep_ratio)
                    values, _ = torch.topk(torch.abs(flat_spectrum), k)
                    threshold = values[-1]
                    mask = torch.abs(spectrum) >= threshold
                    spectrum = spectrum * mask
                
                W_rec = iwalsh_2d_transform(spectrum, H)[:h, :w]
                if W_rec.std() > 0:
                    W_rec = W_rec * (orig_std / W_rec.std())
                module.weight.data = W_rec.t().contiguous()

def evaluate_robust_ppl(model, tokenizer, text, device):
    model.eval()
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    
    # Evaluar en una sola pasada si el texto no es gigante
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        ppl = torch.exp(loss)
    return loss.item(), ppl.item()

def main():
    model_name = "gpt2"
    ratios = [1.0, 0.7, 0.6, 0.5] # Foco en la zona crítica
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"\n--- Experimento v236: EVALUACIÓN ROBUSTA (DIVERSOS DOMINIOS) ---")
    print(f" Tokens de evaluación: {len(tokenizer.encode(ROBUST_TEXT))}")
    
    results = []
    for r in ratios:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        if r < 1.0:
            print(f"Poda Espectral (Ratio {r:.2f})...", end=" ", flush=True)
            apply_spectral_pruning(model, r)
        else:
            print(f"Baseline (Ratio 1.00)...", end=" ", flush=True)
            
        loss, ppl = evaluate_robust_ppl(model, tokenizer, ROBUST_TEXT, device)
        print(f"PPL Robusta: {ppl:.4f}")
        results.append((r, ppl))
    
    print("\n--- COMPARATIVA FINAL (ESTADÍSTICAMENTE ROBUSTA) ---")
    base_ppl = results[0][1]
    for r, p in results:
        delta = p - base_ppl
        status = "🟢" if delta < 10 else "🔴"
        print(f"Ratio {r:.2f} | PPL {p:10.4f} | Delta {delta:+10.4f} {status}")

if __name__ == "__main__":
    main()
