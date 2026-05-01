import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# --- TRANSFORMADAS ---

def walsh_transform(x):
    # Walsh-Hadamard Transform (recursive implementation for any power of 2)
    n = x.size(-1)
    if n == 1:
        return x
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    a = walsh_transform(x_even + x_odd)
    b = walsh_transform(x_even - x_odd)
    return torch.cat([a, b], dim=-1)

def dct_transform(x):
    # Discrete Cosine Transform (type II)
    n = x.size(-1)
    k = torch.arange(n, device=x.device).float()
    i = torch.arange(n, device=x.device).float()
    cos_mat = torch.cos(np.pi / n * (i.unsqueeze(1) + 0.5) * k.unsqueeze(0))
    return torch.matmul(x, cos_mat)

# --- EXPERIMENTO ---

def run_recursive_compression_test():
    print(">>> V196: RECURSIVE SPECTRAL COMPRESSION TEST")
    
    # 1. Señal de prueba (64 puntos, combinación de frecuencias y ruido)
    n = 64
    t = torch.linspace(0, 1, n)
    x = torch.sin(2 * np.pi * 3 * t) + 0.5 * torch.sin(2 * np.pi * 10 * t) + 0.2 * torch.randn(n)
    
    # --- CASO A: COMPRESIÓN DIRECTA (Top 8) ---
    c_direct = walsh_transform(x)
    # Mantener solo los 8 con más energía
    val, idx = torch.topk(torch.abs(c_direct), 8)
    mask = torch.zeros_like(c_direct)
    mask[idx] = 1.0
    c_direct_trunc = c_direct * mask
    x_rec_direct = walsh_transform(c_direct_trunc) / n # Inversa es la misma / N
    mse_direct = torch.mean((x - x_rec_direct)**2).item()
    
    # --- CASO B: COMPRESIÓN RECURSIVA (Top 16 -> Top 8) ---
    # Paso 1: Walsh y truncar a 16
    c1 = walsh_transform(x)
    val1, idx1 = torch.topk(torch.abs(c1), 16)
    mask1 = torch.zeros_like(c1)
    mask1[idx1] = 1.0
    c1_trunc = c1 * mask1
    
    # Paso 2: Transformar los coeficientes truncados (usando DCT para variar)
    c2 = dct_transform(c1_trunc)
    val2, idx2 = torch.topk(torch.abs(c2), 8) # Truncar a los 8 finales
    mask2 = torch.zeros_like(c2)
    mask2[idx2] = 1.0
    c2_trunc = c2 * mask2
    
    # Reconstrucción
    # Inversa DCT (es la transpuesta para DCT ortogonal, aquí aproximamos)
    # Para este test rápido, invertimos el proceso
    c1_rec = torch.matmul(c2_trunc, dct_transform(torch.eye(n)).T)
    x_rec_recursive = walsh_transform(c1_rec) / n
    mse_recursive = torch.mean((x - x_rec_recursive)**2).item()
    
    print(f"\nResultados (N={n}, Final K=8):")
    print(f"  MSE Directo (Walsh K=8):    {mse_direct:.6f}")
    print(f"  MSE Recursivo (W16 -> D8): {mse_recursive:.6f}")
    
    if mse_recursive < mse_direct:
        print("\n¡SORPRESA! La compresión recursiva ha mejorado la reconstrucción.")
    else:
        print("\nComo se esperaba, la doble compresión suele perder más información.")

if __name__ == "__main__":
    run_recursive_compression_test()
