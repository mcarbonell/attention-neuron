"""
attention_neuron/layers/delta_phase.py
======================================
Complex Delta Phase Holographic Layers & Recurrent Memory Blocks.

Features:
- ComplexDeltaPhaseHolographicBlock: O(N) linear recurrent layer with complex U(1) unit-circle phase projections.
- RealDeltaNetVanillaBlock: Standard real-valued DeltaNet layer (Square state memory).
- RealDeltaNetRectangularBlock: Real-valued DeltaNet layer matching rectangular key/val state memory.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShortCausalConv1D(nn.Module):
    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size-1, groups=d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        B, L, D = x.shape
        return x + self.act(self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2))

class FFN(nn.Module):
    def __init__(self, d_model, expand=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.SiLU(),
            nn.Linear(d_model * expand, d_model)
        )

    def forward(self, x):
        return self.net(x)

class ComplexDeltaPhaseHolographicBlock(nn.Module):
    """
    Complex Delta Phase Holographic Layer.
    Key and Query projections are constrained to the complex U(1) unit circle:
      K = polar(1, theta_k), Q = polar(1, theta_q)
    
    State matrix M: (B, H, d_k, d_k) complex64 -> 2 * d_k^2 floats/head.
    Key dimension = 2 * d_k, Value dimension = d_k.
    """
    def __init__(self, d_model, n_heads=2, d_k=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        
        self.theta_k_proj = nn.Linear(d_model, n_heads * d_k)
        self.theta_q_proj = nn.Linear(d_model, n_heads * d_k)
        self.val_proj = nn.Linear(d_model, n_heads * d_k)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(n_heads * d_k, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x
        normed = self.norm1(x)
        conv_x = self.causal_conv(normed)
        B, L, D = conv_x.shape
        
        theta_k = self.theta_k_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        theta_q = self.theta_q_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        v = self.val_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        beta = torch.sigmoid(self.beta_proj(conv_x)).view(B, L, self.n_heads, 1, 1)
        
        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)
        
        M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=conv_x.device)
        out_retrieved = []
        inv_dk = 1.0 / float(self.d_k)
        
        for t in range(L):
            k_t = K[:, t]
            q_t = Q[:, t]
            v_t = v[:, t]
            beta_t = beta[:, t]
            
            k_conj = torch.conj(k_t)
            q_conj = torch.conj(q_t)
            
            v_old = torch.matmul(M, k_conj.unsqueeze(-1)).squeeze(-1).real * inv_dk
            err = v_t - v_old
            
            update = err.to(torch.complex64).unsqueeze(-1) * k_t.unsqueeze(-2)
            M = M + beta_t * update
            
            ret = torch.matmul(M, q_conj.unsqueeze(-1)).squeeze(-1).real * inv_dk
            out_retrieved.append(ret)
            
        retrieved = torch.stack(out_retrieved, dim=1).view(B, L, self.n_heads * self.d_k)
        out = res + self.out_proj(retrieved)
        return out + self.ffn(self.norm2(out))

class RealDeltaNetVanillaBlock(nn.Module):
    """
    Standard Real-Valued DeltaNet Layer (Square State Memory).
    State matrix M: (B, H, d_k_real, d_k_real) float32 -> d_k_real^2 floats/head.
    """
    def __init__(self, d_model, n_heads=2, d_k_real=90):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k_real
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        
        self.k_proj = nn.Linear(d_model, n_heads * self.d_k)
        self.q_proj = nn.Linear(d_model, n_heads * self.d_k)
        self.val_proj = nn.Linear(d_model, n_heads * self.d_k)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(n_heads * self.d_k, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x
        normed = self.norm1(x)
        conv_x = self.causal_conv(normed)
        B, L, D = conv_x.shape
        
        k_raw = self.k_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        q_raw = self.q_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        v = self.val_proj(conv_x).view(B, L, self.n_heads, self.d_k)
        beta = torch.sigmoid(self.beta_proj(conv_x)).view(B, L, self.n_heads, 1, 1)
        
        K = F.normalize(k_raw, p=2, dim=-1)
        Q = F.normalize(q_raw, p=2, dim=-1)
        
        M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.float32, device=conv_x.device)
        out_retrieved = []
        
        for t in range(L):
            k_t = K[:, t]
            q_t = Q[:, t]
            v_t = v[:, t]
            beta_t = beta[:, t]
            
            v_old = torch.matmul(M, k_t.unsqueeze(-1)).squeeze(-1)
            err = v_t - v_old
            
            update = err.unsqueeze(-1) * k_t.unsqueeze(-2)
            M = M + beta_t * update
            
            ret = torch.matmul(M, q_t.unsqueeze(-1)).squeeze(-1)
            out_retrieved.append(ret)
            
        retrieved = torch.stack(out_retrieved, dim=1).view(B, L, self.n_heads * self.d_k)
        out = res + self.out_proj(retrieved)
        return out + self.ffn(self.norm2(out))

class RealDeltaNetRectangularBlock(nn.Module):
    """
    Real-Valued Rectangular DeltaNet Layer.
    Key dimension = 2 * d_k, Value dimension = d_k.
    State matrix M: (B, H, d_val, d_key) float32 -> 2 * d_k^2 floats/head.
    Exact Iso-Floats and Iso-Rectangular control for Complex Delta Phase.
    """
    def __init__(self, d_model, n_heads=2, d_k=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = 2 * d_k
        self.d_val = d_k
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        
        self.k_proj = nn.Linear(d_model, n_heads * self.d_key)
        self.q_proj = nn.Linear(d_model, n_heads * self.d_key)
        self.val_proj = nn.Linear(d_model, n_heads * self.d_val)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(n_heads * self.d_val, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x
        normed = self.norm1(x)
        conv_x = self.causal_conv(normed)
        B, L, D = conv_x.shape
        
        k_raw = self.k_proj(conv_x).view(B, L, self.n_heads, self.d_key)
        q_raw = self.q_proj(conv_x).view(B, L, self.n_heads, self.d_key)
        v = self.val_proj(conv_x).view(B, L, self.n_heads, self.d_val)
        beta = torch.sigmoid(self.beta_proj(conv_x)).view(B, L, self.n_heads, 1, 1)
        
        K = F.normalize(k_raw, p=2, dim=-1)
        Q = F.normalize(q_raw, p=2, dim=-1)
        
        M = torch.zeros(B, self.n_heads, self.d_val, self.d_key, dtype=torch.float32, device=conv_x.device)
        out_retrieved = []
        
        for t in range(L):
            k_t = K[:, t]
            q_t = Q[:, t]
            v_t = v[:, t]
            beta_t = beta[:, t]
            
            v_old = torch.matmul(M, k_t.unsqueeze(-1)).squeeze(-1)
            err = v_t - v_old
            
            update = err.unsqueeze(-1) * k_t.unsqueeze(-2)
            M = M + beta_t * update
            
            ret = torch.matmul(M, q_t.unsqueeze(-1)).squeeze(-1)
            out_retrieved.append(ret)
            
        retrieved = torch.stack(out_retrieved, dim=1).view(B, L, self.n_heads * self.d_val)
        out = res + self.out_proj(retrieved)
        return out + self.ffn(self.norm2(out))
