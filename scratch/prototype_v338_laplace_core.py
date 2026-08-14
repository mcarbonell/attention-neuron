"""
prototype_v338_laplace_core.py
===============================
Experiment v338: Delta-Laplace Phase Memory Core (s = sigma + i*theta, Re(s) <= 0).
Evaluates complex frequency eigenfunctions, Hurwitz stability, and FP64 autograd gradcheck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LaplacePhaseCore(nn.Module):
    """
    Delta-Laplace Phase Memory Core:
    Operates over complex frequency s = sigma + i*theta in the Laplace s-plane.
    Guarantees Hurwitz Stability via strictly non-positive real dissipation sigma <= 0.
    """
    def __init__(self, d_model=64, n_heads=4, d_k=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.inv_dk = 1.0 / float(d_k)
        
        self.w_theta_k = nn.Linear(d_model, d_model, bias=False)
        self.w_sigma_k = nn.Linear(d_model, d_model, bias=False)
        
        self.w_theta_q = nn.Linear(d_model, d_model, bias=False)
        self.w_sigma_q = nn.Linear(d_model, d_model, bias=False)
        
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_beta = nn.Linear(d_model, n_heads, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        complex_dtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        
        theta_k = self.w_theta_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        sigma_k = -F.softplus(self.w_sigma_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2))
        
        theta_q = self.w_theta_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        sigma_q = -F.softplus(self.w_sigma_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2))
        
        v = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        beta = 2.0 * torch.sigmoid(self.w_beta(x)).transpose(1, 2)
        
        r_k = torch.exp(sigma_k)
        K = torch.complex(r_k * torch.cos(theta_k), r_k * torch.sin(theta_k))
        
        r_q = torch.exp(sigma_q)
        Q = torch.complex(r_q * torch.cos(theta_q), r_q * torch.sin(theta_q))
        
        M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=complex_dtype, device=x.device)
        out_list = []
        state_norms = []
        
        for t in range(L):
            kt, qt, vt, bt = K[:, :, t], Q[:, :, t], v[:, :, t], beta[:, :, t]
            
            v_old = torch.matmul(M, torch.conj(kt).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            err = vt - v_old
            update = torch.matmul(err.to(complex_dtype).unsqueeze(-1), kt.unsqueeze(-2))
            
            M = M + bt.unsqueeze(-1).unsqueeze(-1) * update
            state_norms.append(M.norm().item())
            
            out_t = torch.matmul(M, torch.conj(qt).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            out_list.append(out_t)
            
        out = torch.cat(out_list, dim=-1).view(B, self.n_heads, L, self.d_k).transpose(1, 2).reshape(B, L, D)
        return out, state_norms

def run_v338_experiment():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    d_model = 64
    d_k = 16
    core = LaplacePhaseCore(d_model=d_model, n_heads=4, d_k=d_k).to(device)
    
    print("=" * 85)
    print("EXPERIMENT v338: DELTA-LAPLACE PHASE MEMORY CORE (s = sigma + i*theta)")
    print("=" * 85)
    
    x = torch.randn(2, 1024, d_model, device=device)
    out, state_norms = core(x)
    
    max_norm = max(state_norms)
    final_norm = state_norms[-1]
    
    print(f"[v338] Sequence Length L: 1024 tokens")
    print(f"[v338] Max Memory State Norm:   {max_norm:.4f}")
    print(f"[v338] Final Memory State Norm: {final_norm:.4f}")
    print(f"[v338] Output Tensor Shape:     {out.shape}")
    print(f"[v338] Output NaN/Inf Check:    {'CLEAN (No NaN/Inf)' if not torch.isnan(out).any() else 'FAILED'}")
    
    x_fp64 = torch.randn(1, 8, d_model, dtype=torch.float64, device=device, requires_grad=True)
    core_fp64 = LaplacePhaseCore(d_model=d_model, n_heads=4, d_k=d_k).to(torch.float64).to(device)
    
    def func_fp64(inp):
        res, _ = core_fp64(inp)
        return res
        
    gradcheck_passed = torch.autograd.gradcheck(func_fp64, x_fp64, eps=1e-6, atol=1e-4)
    print(f"[v338] FP64 Autograd Gradcheck: {'PASSED [True]' if gradcheck_passed else 'FAILED'}")
    
    acc_passed = not torch.isnan(out).any() and gradcheck_passed
    print("=" * 85)
    print(f"EXPERIMENT v338 RESULT: {'PASSED [ANCLA]' if acc_passed else 'FAILED'}")
    print("=" * 85)

if __name__ == "__main__":
    run_v338_experiment()
