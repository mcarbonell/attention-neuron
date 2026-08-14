"""
prototype_v339_time_scale_invariance.py
=========================================
Experiment v339: Time-Scale Invariance Test under Laplace Eigenfunctions e^{st}.
Evaluates continuous Zero-Order Hold (ZOH) discretization M_t = exp(sigma * dt) * M_{t-1} + beta * dt * update.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContinuousZOHLaplaceBlock(nn.Module):
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

    def forward(self, x, time_scale=1.0):
        B, L, D = x.shape
        dt = 1.0 / float(time_scale)
        
        theta_k = (self.w_theta_k(x) * dt).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        sigma_k = (-F.softplus(self.w_sigma_k(x)) * dt).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        v = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        beta = (2.0 * torch.sigmoid(self.w_beta(x)) * dt).transpose(1, 2)
        
        r_k = torch.exp(sigma_k)
        K = torch.complex(r_k * torch.cos(theta_k), r_k * torch.sin(theta_k))
        
        M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=x.device)
        for t in range(L):
            kt, vt, bt = K[:, :, t], v[:, :, t], beta[:, :, t]
            
            # Continuous decay exp(sigma * dt)
            v_old = torch.matmul(M, torch.conj(kt).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            err = vt - v_old
            M = M * r_k[:, :, t].unsqueeze(-1) + bt.unsqueeze(-1).unsqueeze(-1) * torch.matmul(err.to(torch.complex64).unsqueeze(-1), kt.unsqueeze(-2))
            
        readout = torch.matmul(M, torch.conj(K[:, :, -1]).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
        return readout

def run_v339_experiment():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    d_model = 64
    d_k = 16
    block = ContinuousZOHLaplaceBlock(d_model=d_model, n_heads=4, d_k=d_k).to(device)
    
    print("=" * 85)
    print("EXPERIMENT v339: CONTINUOUS ZOH TIME-SCALE INVARIANCE TEST (s = sigma + i*theta)")
    print("=" * 85)
    
    x_1x = torch.randn(1, 64, d_model, device=device)
    out_1x = block(x_1x, time_scale=1.0)
    
    x_2x = F.interpolate(x_1x.transpose(1, 2), size=128, mode='linear', align_corners=True).transpose(1, 2)
    out_2x = block(x_2x, time_scale=2.0)
    
    x_4x = F.interpolate(x_1x.transpose(1, 2), size=256, mode='linear', align_corners=True).transpose(1, 2)
    out_4x = block(x_4x, time_scale=4.0)
    
    cos_sim_1x_2x = F.cosine_similarity(out_1x.view(1, -1), out_2x.view(1, -1)).item()
    cos_sim_1x_4x = F.cosine_similarity(out_1x.view(1, -1), out_4x.view(1, -1)).item()
    
    mse_1x_2x = F.mse_loss(out_1x, out_2x).item()
    mse_1x_4x = F.mse_loss(out_1x, out_4x).item()
    
    print(f"[v339] Baseline 1x Output Shape: {out_1x.shape}")
    print(f"[v339] 1x vs 2x Time Scale Cosine Similarity: {cos_sim_1x_2x:.4f} (MSE: {mse_1x_2x:.6f})")
    print(f"[v339] 1x vs 4x Time Scale Cosine Similarity: {cos_sim_1x_4x:.4f} (MSE: {mse_1x_4x:.6f})")
    
    acc_passed = cos_sim_1x_2x > 0.90 and cos_sim_1x_4x > 0.85
    print("=" * 85)
    print(f"EXPERIMENT v339 RESULT: {'PASSED [ANCLA]' if acc_passed else 'FAILED'}")
    print("=" * 85)

if __name__ == "__main__":
    run_v339_experiment()
