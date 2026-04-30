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

def block_spectral_prune_matrix(W, ratio=0.25, block_size=16):
    """
    Prunes a matrix in the Walsh spectral domain and RESCALES variance to match original.
    """
    W_orig = W.detach()
    orig_std = W_orig.std()
    h, w = W_orig.shape
    
    ph = (block_size - h % block_size) % block_size
    pw = (block_size - w % block_size) % block_size
    W_padded = torch.zeros((h + ph, w + pw), device=W_orig.device)
    W_padded[:h, :w] = W_orig
    
    nh, nw = W_padded.shape
    H = get_walsh_matrix_sequency(block_size).to(W_orig.device)
    
    blocks = W_padded.view(nh // block_size, block_size, nw // block_size, block_size)
    blocks = blocks.permute(0, 2, 1, 3) 
    
    W_walsh_blocks = torch.matmul(H, torch.matmul(blocks, H.t())) / block_size
    
    # --- LOCAL PRUNING ---
    bh, bw, _, _ = W_walsh_blocks.shape
    num_keep = max(1, int(block_size * block_size * ratio))
    
    flat_blocks = W_walsh_blocks.reshape(-1, block_size * block_size)
    abs_vals = flat_blocks.abs()
    
    topk_vals = torch.topk(abs_vals, num_keep, dim=1).values
    thresholds = topk_vals[:, -1].view(-1, 1)
    
    mask = (abs_vals >= thresholds).float()
    mask[:, 0] = 1.0 
    
    W_walsh_blocks = (flat_blocks * mask).view(bh, bw, block_size, block_size)
    
    # Inverse Walsh
    W_rec_blocks = torch.matmul(H.t(), torch.matmul(W_walsh_blocks, H)) / block_size
    W_rec_padded = W_rec_blocks.permute(0, 2, 1, 3).reshape(nh, nw)
    W_rec = W_rec_padded[:h, :w]
    
    # --- VARIANCE RESCALING ---
    # Crucial step: ensure the compressed weights have the same "power" as the original
    new_std = W_rec.std()
    if new_std > 1e-9:
        W_rec = W_rec * (orig_std / new_std)
        
    return W_rec

@torch.no_grad()
def apply_block_spectral_pruning(model, ratio=0.25, block_size=16):
    print(f"\nApplying Block Spectral Pruning (Ratio={ratio*100:.1f}%, BlockSize={block_size})...")
    count = 0
    for name, module in model.named_modules():
        if "c_attn" in name or "c_fc" in name or "c_proj" in name:
            if hasattr(module, "weight"):
                W = module.weight.t() 
                W_pruned = block_spectral_prune_matrix(W, ratio=ratio, block_size=block_size)
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
    
    # Ratios to test (focusing on the failure points of V129)
    ratios = [0.25, 0.15, 0.1]
    
    for r in ratios:
        print(f"\n--- Testing Block-Based Compression Ratio: {r*100:.1f}% ---")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        
        start_time = time.time()
        apply_block_spectral_pruning(model, ratio=r, block_size=16)
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
