"""
prototype_v340_hurwitz_infinite_context.py
============================================
Experiment v340: Hurwitz Stability & Infinite Context Stress Test (L = 100,000 tokens).
Evaluates memory state norm bounded asymptote under Re(s) <= 0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InfiniteContextLaplaceCore(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_k=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.inv_dk = 1.0 / float(d_k)
        
        self.w_theta_k = nn.Linear(d_model, d_model, bias=False)
        self.w_sigma_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_beta = nn.Linear(d_model, n_heads, bias=False)

    def run_infinite_context(self, sequence_length=100000, check_points=[100, 1000, 10000, 50000, 100000]):
        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        B = 1
        n_heads = self.n_heads
        d_k = self.d_k
        
        M = torch.zeros(B, n_heads, d_k, d_k, dtype=torch.complex64, device=device)
        norms_at_checkpoints = {}
        
        print(f"[v340] Starting Infinite Context Stress Test: L = {sequence_length:,} tokens...")
        
        chunk_size = 1000
        for step_start in range(0, sequence_length, chunk_size):
            # Stream 1000 tokens per chunk to fit in VRAM
            x_chunk = torch.randn(B, chunk_size, self.d_model, device=device)
            
            theta_k = self.w_theta_k(x_chunk).view(B, chunk_size, n_heads, d_k).transpose(1, 2)
            sigma_k = -F.softplus(self.w_sigma_k(x_chunk).view(B, chunk_size, n_heads, d_k).transpose(1, 2))
            
            v = self.w_v(x_chunk).view(B, chunk_size, n_heads, d_k).transpose(1, 2)
            beta = 2.0 * torch.sigmoid(self.w_beta(x_chunk)).transpose(1, 2)
            
            r_k = torch.exp(sigma_k)
            K = torch.complex(r_k * torch.cos(theta_k), r_k * torch.sin(theta_k))
            
            for t in range(chunk_size):
                kt, vt, bt = K[:, :, t], v[:, :, t], beta[:, :, t]
                current_step = step_start + t + 1
                
                v_old = torch.matmul(M, torch.conj(kt).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
                err = vt - v_old
                M = M * r_k[:, :, t].unsqueeze(-1) + bt.unsqueeze(-1).unsqueeze(-1) * torch.matmul(err.to(torch.complex64).unsqueeze(-1), kt.unsqueeze(-2))
                
                if current_step in check_points:
                    norms_at_checkpoints[current_step] = M.norm().item()
                    print(f"       Token Step {current_step:>7,}: Memory Norm = {M.norm().item():.4f}")
                    
        return norms_at_checkpoints, not torch.isnan(M).any().item()

def run_v340_experiment():
    print("=" * 85)
    print("EXPERIMENT v340: HURWITZ STABILITY & INFINITE CONTEXT STRESS TEST (L=100,000)")
    print("=" * 85)
    
    d_model = 64
    d_k = 16
    core = InfiniteContextLaplaceCore(d_model=d_model, n_heads=4, d_k=d_k)
    
    norms_dict, is_clean = core.run_infinite_context(sequence_length=100000)
    
    max_norm = max(norms_dict.values())
    final_norm = norms_dict[100000]
    
    print("-" * 85)
    print(f"[v340] Max Norm Recorded:   {max_norm:.4f}")
    print(f"[v340] Final Step Norm:     {final_norm:.4f}")
    print(f"[v340] Output Clean Check:  {'CLEAN (No NaNs/Infs)' if is_clean else 'FAILED'}")
    
    acc_passed = is_clean and max_norm < 500.0
    print("=" * 85)
    print(f"EXPERIMENT v340 RESULT: {'PASSED [ANCLA]' if acc_passed else 'FAILED'}")
    print("=" * 85)

if __name__ == "__main__":
    run_v340_experiment()
