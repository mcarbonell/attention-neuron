import torch
import torch.nn as nn
import torch.nn.functional as F
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

def compress_smooth_walsh(W, target_ratio=0.25):
    """
    Aplica Smooth Walsh: Walsh -> Sub-espectro -> Inversa Mini -> Upscale Bilineal.
    """
    h, w = W.shape
    
    # Buscamos potencias de 2 para Walsh
    N = 2**int(np.ceil(np.log2(max(h, w))))
    K = 2**int(np.ceil(np.log2(max(h, w) * np.sqrt(target_ratio))))
    
    # Padding a N
    W_padded = torch.zeros((N, N), device=device)
    W_padded[:h, :w] = W
    
    # Matrices de Walsh
    H_N = get_walsh_matrix_sequency(N).to(device)
    H_K = get_walsh_matrix_sequency(K).to(device)
    
    # 1. Transformada completa
    spectrum = walsh_2d_transform(W_padded, H_N)
    
    # 2. Quedarnos con el mini-espectro (Bajas frecuencias)
    mini_spectrum = spectrum[:K, :K]
    
    # 3. Inversa en baja resolución
    # El factor de escala (N/K)**2 compensa la diferencia de dimensiones en la energía
    img_mini = iwalsh_2d_transform(mini_spectrum, H_K) #/ ((N/K)**2)
    
    # 4. Upscale Bilineal a la resolución original
    img_mini_tensor = img_mini.unsqueeze(0).unsqueeze(0)
    img_smooth = F.interpolate(img_mini_tensor, size=(h, w), mode='bilinear', align_corners=False)
    W_rec = img_smooth.squeeze(0).squeeze(0)
    
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
    target_ratio = 0.25 # Compresión 4x
    
    print(f"\n--- Experimento v232: GPT-2 SMOOTH WALSH COMPRESSION ---")
    print(f" Ratio de compresión objetivo: {1/target_ratio:.1f}x")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    eval_text = "The secret of intelligence is in the structure of information."
    
    # 1. Baseline
    l0, p0 = evaluate_model(model, tokenizer, eval_text, device)
    print(f"Baseline PPL: {p0:.4f}")
    
    # 2. Bucle de Compresión Smooth Walsh
    print("\nComprimiendo capas con Smooth Walsh...")
    total_weights = 0
    start_time = time.time()
    for name, module in model.named_modules():
        if any(x in name for x in ["c_attn", "c_fc", "c_proj"]):
            if hasattr(module, "weight"):
                W_orig = module.weight.data.t().clone()
                orig_std = W_orig.std()
                
                # Comprimir 
                W_rec = compress_smooth_walsh(W_orig, target_ratio=target_ratio)
                
                # Rescalar varianza
                if W_rec.std() > 0:
                    W_rec = W_rec * (orig_std / W_rec.std())
                
                module.weight.data = W_rec.t().contiguous()
                total_weights += W_orig.numel()
                print(f"  [SMOOTH OK] {name}")
    
    duration = time.time() - start_time
    print(f"\nCompresión completada en {duration:.2f}s")
    
    # 3. Métricas
    size_fp32 = (total_weights * 4) / (1024**2)
    size_smooth = (total_weights * target_ratio * 4) / (1024**2)
    print(f"\n--- Métricas ---")
    print(f" Tamaño Original: {size_fp32:.2f} MB")
    print(f" Tamaño Smooth Walsh: {size_smooth:.2f} MB")
    
    # 4. Evaluación
    lq, pq = evaluate_model(model, tokenizer, eval_text, device)
    print(f"Post-Smooth PPL: {pq:.4f} (Delta: {pq - p0:+.4f})")
    
    # 5. Generación
    prompt = "The secret of intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    print(f"\nGeneración: {tokenizer.decode(out[0])}")

if __name__ == "__main__":
    main()
