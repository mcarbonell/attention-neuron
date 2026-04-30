import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
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

def spectral_quantize_matrix(W, bits=1):
    """
    Quantizes a matrix in the Walsh spectral domain.
    """
    W_orig = W.detach()
    orig_std = W_orig.std()
    h, w = W_orig.shape
    N = 2**int(np.ceil(np.log2(max(h, w))))
    
    W_padded = torch.zeros((N, N), device=W_orig.device)
    W_padded[:h, :w] = W_orig
    
    H = get_walsh_matrix_sequency(N).to(W_orig.device)
    W_walsh = torch.matmul(H, torch.matmul(W_padded, H.t())) / N
    
    if bits == 1:
        # 1-bit: Keep only the sign and scale by the mean absolute value
        scale = W_walsh.abs().mean()
        W_walsh_q = torch.sign(W_walsh) * scale
    elif bits == 2:
        # 2-bit: 4 levels. We can use simple uniform quantization of the range
        # Or better: use the standard deviation
        std = W_walsh.std()
        # Levels: -1.5*std, -0.5*std, 0.5*std, 1.5*std (approx)
        # We can also use quantiles for better distribution
        levels = torch.tensor([-1.5, -0.5, 0.5, 1.5], device=W_orig.device) * std
        
        # Simple rounding to nearest level
        W_walsh_q = torch.zeros_like(W_walsh)
        # Find nearest
        dist = (W_walsh.unsqueeze(-1) - levels).abs()
        indices = torch.argmin(dist, dim=-1)
        W_walsh_q = levels[indices]
    else:
        W_walsh_q = W_walsh # No quantization
        
    # Inverse Walsh
    W_rec_padded = torch.matmul(H.t(), torch.matmul(W_walsh_q, H)) / N
    W_rec = W_rec_padded[:h, :w]
    
    # Rescale variance to match original power
    new_std = W_rec.std()
    if new_std > 1e-9:
        W_rec = W_rec * (orig_std / new_std)
        
    return W_rec

@torch.no_grad()
def apply_spectral_quantization(model, bits=1):
    print(f"\nApplying Spectral Quantization ({bits}-bit Walsh)...")
    count = 0
    for name, module in model.named_modules():
        if "c_attn" in name or "c_fc" in name or "c_proj" in name:
            if hasattr(module, "weight"):
                W = module.weight.t() 
                W_q = spectral_quantize_matrix(W, bits=bits)
                module.weight.copy_(W_q.t())
                count += 1
    print(f"  Quantized {count} weight matrices.")

def generate_text(model, tokenizer, prompt="The capital of France is", max_len=20):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_len, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Modes to test
    modes = [None, 2, 1] # Baseline, 2-bit, 1-bit
    
    for m in modes:
        if m is None:
            print(f"\n--- Testing Baseline (FP32) ---")
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            print(f"\n--- Testing {m}-bit Spectral Quantization ---")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.eval()
            
            start_time = time.time()
            apply_spectral_quantization(model, bits=m)
            print(f"  Quantization took {time.time() - start_time:.2f}s")
        
        # Test Generation
        prompts = [
            "The capital of France is",
            "The first man on the moon was",
            "Artificial intelligence is a field of"
        ]
        
        print("  Generating samples:")
        for p in prompts:
            gen = generate_text(model, tokenizer, p)
            print(f"    Q: {p} | A: {gen.replace(p, '').strip()}")
            
if __name__ == "__main__":
    main()
