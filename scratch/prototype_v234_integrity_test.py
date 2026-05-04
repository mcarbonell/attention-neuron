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

def compress_spectral_topk(W, keep_ratio=1.0):
    """
    Poda Espectral Adaptativa: Mantiene el Top-K de coeficientes por magnitud.
    Si keep_ratio=1.0, debería ser una reconstrucción perfecta.
    """
    h, w = W.shape
    N = 2**int(np.ceil(np.log2(max(h, w))))
    
    W_padded = torch.zeros((N, N), device=device)
    W_padded[:h, :w] = W
    
    H = get_walsh_matrix_sequency(N).to(device)
    
    # 1. Transformar
    spectrum = walsh_2d_transform(W_padded, H)
    
    # 2. Top-K Pruning
    if keep_ratio < 1.0:
        flat_spectrum = spectrum.flatten()
        k = int(flat_spectrum.numel() * keep_ratio)
        values, _ = torch.topk(torch.abs(flat_spectrum), k)
        threshold = values[-1]
        mask = torch.abs(spectrum) >= threshold
        spectrum = spectrum * mask
    
    # 3. Inversa
    W_rec_full = iwalsh_2d_transform(spectrum, H)
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
    keep_ratio = 1.0 # TEST DE INTEGRIDAD (100% COEFICIENTES)
    
    print(f"\n--- Experimento v234: TEST DE INTEGRIDAD SPECTRAL WALSH ---")
    print(f" Ratio: {keep_ratio*100:.1f}% (Debería ser Lossless)")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    eval_text = "The secret of intelligence is in the structure of information."
    
    # 1. Baseline
    l0, p0 = evaluate_model(model, tokenizer, eval_text, device)
    print(f"Baseline PPL: {p0:.4f}")
    
    # 2. Bucle de Verificación por Capa
    print("\nVerificando capas una a una...")
    for name, module in model.named_modules():
        if any(x in name for x in ["c_attn", "c_fc", "c_proj"]):
            if hasattr(module, "weight"):
                W_orig = module.weight.data.t().clone()
                
                # Transformación + Reconstrucción
                W_rec = compress_spectral_topk(W_orig, keep_ratio=keep_ratio)
                
                # Cálculo de Error
                mse = torch.mean((W_orig - W_rec)**2).item()
                max_err = torch.max(torch.abs(W_orig - W_rec)).item()
                
                # Aplicar al modelo
                module.weight.data = W_rec.t().contiguous()
                
                print(f"  {name:40} | MSE: {mse:.2e} | MaxErr: {max_err:.2e}")
    
    # 3. Evaluación Final
    lq, pq = evaluate_model(model, tokenizer, eval_text, device)
    print(f"\n--- Resultado Final ---")
    print(f" Baseline PPL: {p0:.4f}")
    print(f" Post-Test PPL: {pq:.4f} (Delta: {pq - p0:+.4f})")
    
    if abs(pq - p0) < 1e-3:
        print("\n✅ TEST SUPERADO: La transformación es matemáticamente íntegra.")
    else:
        print("\n❌ TEST FALLIDO: Hay una pérdida de información inesperada.")

if __name__ == "__main__":
    main()
