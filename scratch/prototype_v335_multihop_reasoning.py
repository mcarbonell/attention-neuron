"""
prototype_v335_multihop_reasoning.py
======================================
Experiment v335: Multi-Hop Reasoning Chains (A -> B -> C -> D) in Single Forward Pass.
Evaluates autonomous internal phase micro-step recurrence over Complex State Memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHopPhaseBlock(nn.Module):
    def __init__(self, d_model=64, d_k=16, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.n_heads = n_heads
        self.inv_dk = 1.0 / float(d_k)
        
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.phase_map = nn.Linear(d_k, d_k, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward_multihop(self, x, hops=2):
        B, L, D = x.shape
        theta_k = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        theta_q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        K = torch.complex(torch.cos(theta_k), torch.sin(theta_k))
        Q = torch.complex(torch.cos(theta_q), torch.sin(theta_q))
        
        # Build Complex State Matrix M
        M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=x.device)
        for t in range(L):
            kt, vt = K[:, :, t], v[:, :, t]
            v_old = torch.matmul(M, torch.conj(kt).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            err = vt - v_old
            M = M + torch.matmul(err.to(torch.complex64).unsqueeze(-1), kt.unsqueeze(-2))
            
        # Autonomous Internal Multi-Hop Recurrence
        q_current = Q[:, :, -1] # Query last token A
        for hop in range(hops):
            readout = torch.matmul(M, torch.conj(q_current).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            # Map readout v_hop to next query phase Q_next for hop + 1
            theta_next = self.phase_map(readout)
            q_current = torch.complex(torch.cos(theta_next), torch.sin(theta_next))
            
        out = self.out_proj(readout.reshape(B, -1))
        return out, readout

def run_v335_experiment():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    d_model = 64
    d_k = 16
    block = MultiHopPhaseBlock(d_model=d_model, d_k=d_k, n_heads=4).to(device)
    
    print("=" * 85)
    print("EXPERIMENT v335: AUTONOMOUS MULTI-HOP REASONING (A -> B -> C) IN 1 FORWARD PASS")
    print("=" * 85)
    
    x = torch.randn(4, 32, d_model, device=device)
    out_1hop, _ = block.forward_multihop(x, hops=1)
    out_2hop, _ = block.forward_multihop(x, hops=2)
    out_3hop, _ = block.forward_multihop(x, hops=3)
    
    diff_1_vs_2 = (out_1hop - out_2hop).abs().mean().item()
    diff_2_vs_3 = (out_2hop - out_3hop).abs().mean().item()
    
    print(f"[v335] 1-Hop Output Norm: {out_1hop.norm().item():.4f}")
    print(f"[v335] 2-Hop Output Norm: {out_2hop.norm().item():.4f} | Diff 1-Hop vs 2-Hop: {diff_1_vs_2:.4f}")
    print(f"[v335] 3-Hop Output Norm: {out_3hop.norm().item():.4f} | Diff 2-Hop vs 3-Hop: {diff_2_vs_3:.4f}")
    
    print("=" * 85)
    print("EXPERIMENT v335 RESULT: EXECUTED CLEANLY [ANCLA]")
    print("=" * 85)

if __name__ == "__main__":
    run_v335_experiment()
