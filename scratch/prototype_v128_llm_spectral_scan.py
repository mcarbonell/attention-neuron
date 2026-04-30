import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
import matplotlib.pyplot as plt
import numpy as np
import time

def get_walsh_matrix_sequency(N):
    def get_walsh(n):
        if n == 1: return torch.tensor([[1.0]])
        h_prev = get_walsh(n // 2)
        return torch.cat([torch.cat([h_prev, h_prev], dim=1),
                          torch.cat([h_prev, -h_prev], dim=1)], dim=0)
    H = get_walsh(N)
    crossings = [( (H[i, :-1] * H[i, 1:] < 0).sum().item(), i) for i in range(N)]
    crossings.sort()
    return H[[idx for _, idx in crossings]]

def analyze_matrix(W, name="Layer"):
    W = W.detach() # Ensure no grad
    original_shape = W.shape
    h, w = W.shape
    N = 2**int(np.ceil(np.log2(max(h, w))))
    
    W_padded = torch.zeros((N, N), device=W.device)
    W_padded[:h, :w] = W
    
    print(f"\n--- Analyzing {name} (Shape: {h}x{w}) ---")
    
    # --- 1. WALSH ANALYSIS ---
    H = get_walsh_matrix_sequency(N).to(W.device)
    W_walsh = torch.matmul(H, torch.matmul(W_padded, H.t())) / N
    
    # --- 2. DCT ANALYSIS (using FFT magnitude as proxy) ---
    def dct_proxy(x):
        # rfft2 is a good proxy for frequency content
        return torch.fft.rfft2(x, norm='ortho').abs()
    
    W_spec = dct_proxy(W_padded)
    
    def get_comp_ratios(coeffs, title):
        energy_total = torch.sum(coeffs**2)
        sorted_c, _ = torch.sort(coeffs.flatten().abs(), descending=True)
        cumulative = torch.cumsum(sorted_c**2, dim=0) / energy_total
        
        ratios = []
        for t in [0.5, 0.9, 0.99]:
            try:
                idx = torch.where(cumulative >= t)[0][0].item()
                ratios.append((idx + 1) / coeffs.numel())
            except:
                ratios.append(1.0)
        print(f"  [{title}] 50%: {ratios[0]*100:.2f}%, 90%: {ratios[1]*100:.2f}%, 99%: {ratios[2]*100:.2f}%")
        return ratios

    r_walsh = get_comp_ratios(W_walsh, "Walsh")
    r_spec = get_comp_ratios(W_spec, "FFT-Mag")

    # Visualization
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(W[:128, :128].cpu().numpy(), cmap='RdBu')
    ax[0].set_title(f"Original {name}")
    
    ax[1].imshow(torch.log10(W_walsh.abs() + 1e-6).cpu().numpy(), cmap='magma')
    ax[1].set_title("Walsh Spectrum")
    
    ax[2].imshow(torch.log10(W_spec + 1e-6).cpu().numpy(), cmap='viridis')
    ax[2].set_title("FFT Magnitude Spectrum")
    
    plt.tight_layout()
    plt.savefig(f"docs/spectral_analysis_{name}.png")
    return r_walsh, r_spec

def main():
    model_name = "openai-community/gpt2"
    print(f"Loading {model_name} weights to CPU...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Layer 0 (Input-ish)
        analyze_matrix(model.transformer.h[0].attn.c_attn.weight.t(), name="GPT2_L0_Attn")
        
        # Layer 6 (Middle)
        analyze_matrix(model.transformer.h[6].mlp.c_fc.weight.t(), name="GPT2_L6_MLP")
        
        # Layer 11 (Output-ish)
        analyze_matrix(model.transformer.h[11].mlp.c_proj.weight.t(), name="GPT2_L11_MLP_Out")

    except Exception as e:
        print(f"Error loading model: {e}")
        # Simulation of a structured matrix (Low-rank + Low-frequency bias)
        print("Falling back to simulated Structured Weight Matrix...")
        N = 1024
        H = get_walsh_matrix_sequency(N)
        freqs = torch.linspace(1, N, N).view(1, -1) * torch.linspace(1, N, N).view(-1, 1)
        spectrum = 1.0 / (freqs**0.5)
        W_sim_walsh = torch.randn(N, N) * spectrum
        W_sim = torch.matmul(H.t(), torch.matmul(W_sim_walsh, H)) / N
        analyze_matrix(W_sim, name="Simulated_Structured_Layer")

if __name__ == "__main__":
    main()
