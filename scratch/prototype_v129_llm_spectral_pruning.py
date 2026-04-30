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

def spectral_prune_matrix(W, ratio=0.25):
    """
    Prunes a matrix in the Walsh spectral domain, keeping only the top-ratio coefficients.
    """
    W_orig = W.detach()
    h, w = W_orig.shape
    N = 2**int(np.ceil(np.log2(max(h, w))))
    
    # Pad to power of 2
    W_padded = torch.zeros((N, N), device=W_orig.device)
    W_padded[:h, :w] = W_orig
    
    # Walsh matrix
    H = get_walsh_matrix_sequency(N).to(W_orig.device)
    
    # Forward Walsh
    W_walsh = torch.matmul(H, torch.matmul(W_padded, H.t())) / N
    
    # Pruning: Keep only top-ratio coefficients by magnitude
    flat = W_walsh.flatten().abs()
    num_keep = int(N * N * ratio)
    if num_keep < N*N:
        threshold = torch.topk(flat, num_keep).values[-1]
        mask = (W_walsh.abs() >= threshold).float()
        W_walsh = W_walsh * mask
    
    # Inverse Walsh
    W_rec_padded = torch.matmul(H.t(), torch.matmul(W_walsh, H)) / N
    
    # Unpad
    W_rec = W_rec_padded[:h, :w]
    return W_rec

@torch.no_grad()
def apply_spectral_pruning(model, ratio=0.25):
    print(f"\nApplying Spectral Pruning (Ratio={ratio*100:.1f}%)...")
    count = 0
    for name, module in model.named_modules():
        # Target Linear layers in Attention and MLP
        # GPT2 uses Conv1D which acts like Linear but transposed weights
        if "c_attn" in name or "c_fc" in name or "c_proj" in name:
            if hasattr(module, "weight"):
                # GPT2 weights are [In, Out]
                W = module.weight.t() 
                W_pruned = spectral_prune_matrix(W, ratio=ratio)
                module.weight.copy_(W_pruned.t())
                count += 1
    print(f"  Pruned {count} weight matrices.")

def generate_text(model, tokenizer, prompt="The capital of France is", max_len=20):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_len, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ratios to test
    ratios = [1.0, 0.5, 0.25, 0.1, 0.05]
    
    results = []
    
    # We load the model fresh for each ratio to avoid compounding errors
    for r in ratios:
        print(f"\n--- Testing Compression Ratio: {r*100:.1f}% ---")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        
        if r < 1.0:
            start_time = time.time()
            apply_spectral_pruning(model, ratio=r)
            print(f"  Pruning took {time.time() - start_time:.2f}s")
        
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
