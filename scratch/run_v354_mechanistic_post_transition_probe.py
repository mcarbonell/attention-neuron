"""
run_v354_mechanistic_post_transition_probe.py
=============================================
Post-Transition Mechanistic Angle Probe (Step 3000) for Z_7 and Z_9.

Tasks:
------
1. Extract trained phi_t angles at step 3000 (post-transition convergence).
2. For Z_7: Compute circular mean, circular std, and linear fit R^2 across coprimes m in {1..6}.
3. For Z_9: Check if phi_t angles organize into a 3x3 grid (Z_3 x Z_3) or 9 uniform angles!
"""

import time, sys, math, platform, torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()

def ts():
    elapsed = time.time() - T0
    h = int(elapsed // 3600)
    m = int(elapsed % 3600 // 60)
    s = elapsed % 60
    return f"[{h:02d}:{m:02d}:{s:05.2f}]"

def generate_zk_batch(batch_size=64, seq_len=64, modulo=7, device='cpu'):
    elements = torch.randint(0, modulo, (batch_size, seq_len), device=device)
    target_cumsum = torch.cumsum(elements, dim=1) % modulo
    return elements, target_cumsum

class ShortCausalConv1D(nn.Module):
    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size-1, groups=d_model)
        self.act = nn.SiLU()
    def forward(self, x):
        B, L, D = x.shape
        return x + self.act(self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2))

class ComplexBetaDeltaPhaseBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_k=16):
        super().__init__()
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_k
        self.inv_dk = 1.0 / float(d_k)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.theta_k_proj = nn.Linear(d_model, n_heads * d_k)
        self.theta_q_proj = nn.Linear(d_model, n_heads * d_k)
        self.val_proj = nn.Linear(d_model, n_heads * d_k)
        self.phi_beta_proj = nn.Linear(d_model, n_heads)
        
        self.out_proj = nn.Linear(n_heads * d_k, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2*d_model), nn.SiLU(), nn.Linear(2*d_model, d_model))

    def forward(self, x, return_phi=False):
        res = x; nx = self.conv(self.norm1(x))
        B, L, D = nx.shape
        
        theta_k = self.theta_k_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        theta_q = self.theta_q_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.val_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        phi_beta = self.phi_beta_proj(nx).transpose(1, 2)
        
        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)
        beta_complex = 1.0 + torch.polar(torch.ones_like(phi_beta), phi_beta)
        
        M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=x.device)
        out_list = []
        
        for t in range(L):
            kt, qt, vt, b_t = K[:, :, t], Q[:, :, t], v[:, :, t], beta_complex[:, :, t]
            v_old = torch.matmul(M_state, torch.conj(kt).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            err = vt - v_old
            err_c = err.to(torch.complex64)
            update_term = b_t.unsqueeze(-1).unsqueeze(-1) * torch.matmul(err_c.unsqueeze(-1), kt.unsqueeze(-2))
            M_state = M_state + update_term
            out_t = torch.matmul(M_state, torch.conj(qt).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            out_list.append(out_t)
            
        out_concat = torch.stack(out_list, dim=2).transpose(1, 2).reshape(B, L, self.n_heads * self.d_k)
        out = res + self.out_proj(out_concat)
        out = out + self.ffn(self.norm2(out))
        if return_phi:
            return out, phi_beta
        return out

class DeepModelLM(nn.Module):
    def __init__(self, block_cls, modulo=7, d_model=64, n_layers=4, n_heads=4, d_k=16):
        super().__init__()
        self.embed = nn.Embedding(modulo, d_model)
        self.pos_embed = nn.Embedding(1024, d_model)
        self.layers = nn.ModuleList([block_cls(d_model=d_model, n_heads=n_heads, d_k=d_k) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, modulo)

    def forward(self, x, return_phi=False):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        phi_list = []
        for layer in self.layers:
            if return_phi and isinstance(layer, ComplexBetaDeltaPhaseBlock):
                h, phi = layer(h, return_phi=True)
                phi_list.append(phi)
            else:
                h = layer(h)
        logits = self.head(h)
        if return_phi:
            return logits, phi_list
        return logits

def train_and_probe(modulo=7, steps=3000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{ts()} --- Training Complex Beta on Z_{modulo} for {steps} steps (Post-Transition Probe) ---", flush=True)
    
    model = DeepModelLM(ComplexBetaDeltaPhaseBlock, modulo=modulo, d_model=64, n_layers=4, n_heads=4, d_k=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    for step in range(1, steps + 1):
        model.train()
        x_b, y_b = generate_zk_batch(batch_size=64, seq_len=64, modulo=modulo, device=device)
        logits = model(x_b)
        loss = F.cross_entropy(logits.view(-1, modulo), y_b.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 1000 == 0 or step == steps:
            preds = logits.argmax(dim=-1)
            acc = (preds == y_b).float().mean().item() * 100.0
            print(f"{ts()}  [Step {step:>4}/{steps}] Loss: {loss.item():.4f} | Acc: {acc:.2f}%", flush=True)
            
    model.eval()
    with torch.no_grad():
        x_ev, y_ev = generate_zk_batch(batch_size=200, seq_len=64, modulo=modulo, device=device)
        logits_ev, phi_list = model(x_ev, return_phi=True)
        preds_ev = logits_ev.argmax(dim=-1)
        final_acc = (preds_ev == y_ev).float().mean().item() * 100.0
        print(f"{ts()}  ==> Final Z_{modulo} Accuracy at Step {steps}: {final_acc:.2f}%\n", flush=True)
        
        # Extract phi_t from last layer: shape [B, n_heads, L]
        phi_last = phi_list[-1].permute(0, 2, 1).reshape(-1, 4) # [B*L, 4]
        tokens_flat = x_ev.view(-1)
        
        print(f"=== POST-TRANSITION PHI PROBE FOR Z_{modulo} (STEP {steps}) ===", flush=True)
        for h in range(4):
            print(f"\n--- Layer 4 Head {h+1} Phase Angle Analysis ---", flush=True)
            means = []
            stds = []
            for digit in range(modulo):
                mask = (tokens_flat == digit)
                phi_d = (phi_last[mask, h] % (2.0 * math.pi)).cpu().numpy()
                sin_mean = np.mean(np.sin(phi_d))
                cos_mean = np.mean(np.cos(phi_d))
                circ_mean = np.arctan2(sin_mean, cos_mean) % (2.0 * math.pi)
                R = np.sqrt(sin_mean**2 + cos_mean**2)
                circ_std = np.sqrt(-2.0 * np.log(max(R, 1e-6)))
                
                means.append(circ_mean)
                stds.append(circ_std)
                print(f"  Digit {digit}: CircMean = {circ_mean:.4f} rad ({np.degrees(circ_mean):.1f}deg) | CircStd = {circ_std:.4f}", flush=True)
                
            # Fit phi_j = 2*pi*m*j / modulo + c for coprimes m
            best_r2 = -1.0
            best_m = None
            digits_arr = np.arange(modulo)
            for m in range(1, modulo):
                theory = (2.0 * math.pi * m * digits_arr / modulo) % (2.0 * math.pi)
                # Circular correlation / R2
                res = (np.array(means) - theory + math.pi) % (2.0 * math.pi) - math.pi
                r2 = 1.0 - (np.var(res) / (np.var(means) + 1e-6))
                if r2 > best_r2:
                    best_r2 = r2
                    best_m = m
            print(f"  --> Best Linear Fourier Generator Fit: m = {best_m} with R^2 = {best_r2:.4f}", flush=True)

def main():
    print("===============================================================================================", flush=True)
    print("          V354 POST-TRANSITION MECHANISTIC ANGLE PROBE (STEP 3000)                             ", flush=True)
    print("===============================================================================================", flush=True)
    train_and_probe(modulo=7, steps=3000)
    train_and_probe(modulo=9, steps=3000)

if __name__ == "__main__":
    main()
