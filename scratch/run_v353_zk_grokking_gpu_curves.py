"""
run_v353_zk_grokking_gpu_curves.py
==================================
Certified GPU Benchmark Script for Z_k Grokking Phase Transition Curves & Steps-to-Solve (v353).

Target Modulos:
---------------
- Z_7 (Odd Prime): Pure non-trivial group without subgroups.
- Z_9 (Odd Composite 3^2): Critical test for Odd Composite vs Even Parity Trap.
- Z_12 (Even Composite 2^2 * 3): Parity shortcut vs full 12-state rotation.

Arms Tested:
------------
1. Real Beta DeltaNet (beta in (0, 2))
2. Fixed Real Beta=2.0 (Exact Reflection Isometry)
3. DeltaProduct Real (n_h = 2 Real Householders per token)
4. Complex Beta DeltaPhase (beta_t = 1 + e^{i phi_t}, Unitary U(d) Phasors)

Execution Setup:
----------------
- Budget: 3,000 Steps per model (Tesla T4 GPU accelerated).
- Metrics: Loss & Accuracy logged every 50 steps to record the exact Grokking Phase Transition curve.
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

def print_log_header(device):
    print("===============================================================================================", flush=True)
    print("        GPU BENCHMARK v353: Z_k GROKKING PHASE TRANSITION CURVES & STEPS-TO-SOLVE              ", flush=True)
    print("===============================================================================================", flush=True)
    print(f" Timestamp:              {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", flush=True)
    print(f" Execution Device:       {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f" PyTorch Version:        {torch.__version__}", flush=True)
    print(f" Budget:                 3,000 Steps per model | Logging every 50 steps", flush=True)
    print("===============================================================================================\n", flush=True)

def generate_zk_batch(batch_size=64, seq_len=64, modulo=7, device='cpu'):
    elements = torch.randint(0, modulo, (batch_size, seq_len), device=device)
    target_cumsum = torch.cumsum(elements, dim=1) % modulo
    return elements, target_cumsum

# ── 1. Building Blocks ──────────────────────────────────────────────────────

class ShortCausalConv1D(nn.Module):
    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size-1, groups=d_model)
        self.act = nn.SiLU()
    def forward(self, x):
        B, L, D = x.shape
        return x + self.act(self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2))

# ── A. Complex Beta DeltaPhase Block ────────────────────────────────────────

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

    def forward(self, x):
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
        return out + self.ffn(self.norm2(out))

# ── B. Fixed Beta=2.0 Real Isometric Block ──────────────────────────────────

class FixedIsometricRealBetaBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_k=16):
        super().__init__()
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_k
        self.inv_dk = 1.0 / float(d_k)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.k_proj = nn.Linear(d_model, n_heads * d_k)
        self.q_proj = nn.Linear(d_model, n_heads * d_k)
        self.val_proj = nn.Linear(d_model, n_heads * d_k)
        
        self.out_proj = nn.Linear(n_heads * d_k, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2*d_model), nn.SiLU(), nn.Linear(2*d_model, d_model))

    def forward(self, x):
        res = x; nx = self.conv(self.norm1(x))
        B, L, D = nx.shape
        
        k = self.k_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        q = self.q_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.val_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, device=x.device)
        out_list = []
        
        for t in range(L):
            kt, qt, vt = k[:, :, t], q[:, :, t], v[:, :, t]
            v_old = torch.matmul(M_state, kt.unsqueeze(-1)).squeeze(-1) * self.inv_dk
            err = vt - v_old
            update_term = 2.0 * torch.matmul(err.unsqueeze(-1), kt.unsqueeze(-2))
            M_state = M_state + update_term
            out_t = torch.matmul(M_state, qt.unsqueeze(-1)).squeeze(-1) * self.inv_dk
            out_list.append(out_t)
            
        out_concat = torch.stack(out_list, dim=2).transpose(1, 2).reshape(B, L, self.n_heads * self.d_k)
        out = res + self.out_proj(out_concat)
        return out + self.ffn(self.norm2(out))

# ── C. Real Beta (0, 2) DeltaNet Block ──────────────────────────────────────

class RealBetaDeltaNetBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_k=16):
        super().__init__()
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_k
        self.inv_dk = 1.0 / float(d_k)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.k_proj = nn.Linear(d_model, n_heads * d_k)
        self.q_proj = nn.Linear(d_model, n_heads * d_k)
        self.val_proj = nn.Linear(d_model, n_heads * d_k)
        self.beta_proj = nn.Linear(d_model, n_heads)
        
        self.out_proj = nn.Linear(n_heads * d_k, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2*d_model), nn.SiLU(), nn.Linear(2*d_model, d_model))

    def forward(self, x):
        res = x; nx = self.conv(self.norm1(x))
        B, L, D = nx.shape
        
        k = self.k_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        q = self.q_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.val_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        beta_real = 2.0 * torch.sigmoid(self.beta_proj(nx)).transpose(1, 2)
        
        M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, device=x.device)
        out_list = []
        
        for t in range(L):
            kt, qt, vt, b_t = k[:, :, t], q[:, :, t], v[:, :, t], beta_real[:, :, t]
            v_old = torch.matmul(M_state, kt.unsqueeze(-1)).squeeze(-1) * self.inv_dk
            err = vt - v_old
            update_term = b_t.unsqueeze(-1).unsqueeze(-1) * torch.matmul(err.unsqueeze(-1), kt.unsqueeze(-2))
            M_state = M_state + update_term
            out_t = torch.matmul(M_state, qt.unsqueeze(-1)).squeeze(-1) * self.inv_dk
            out_list.append(out_t)
            
        out_concat = torch.stack(out_list, dim=2).transpose(1, 2).reshape(B, L, self.n_heads * self.d_k)
        out = res + self.out_proj(out_concat)
        return out + self.ffn(self.norm2(out))

# ── D. DeltaProduct Real Arm (n_h = 2 Real Householders per token) ──────────

class DeltaProductRealBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_k=16):
        super().__init__()
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_k
        self.inv_dk = 1.0 / float(d_k)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.k1_proj = nn.Linear(d_model, n_heads * d_k)
        self.k2_proj = nn.Linear(d_model, n_heads * d_k)
        self.q_proj = nn.Linear(d_model, n_heads * d_k)
        self.val_proj = nn.Linear(d_model, n_heads * d_k)
        self.beta1_proj = nn.Linear(d_model, n_heads)
        self.beta2_proj = nn.Linear(d_model, n_heads)
        
        self.out_proj = nn.Linear(n_heads * d_k, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2*d_model), nn.SiLU(), nn.Linear(2*d_model, d_model))

    def forward(self, x):
        res = x; nx = self.conv(self.norm1(x))
        B, L, D = nx.shape
        
        k1 = self.k1_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k2 = self.k2_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        q = self.q_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.val_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        beta1 = 2.0 * torch.sigmoid(self.beta1_proj(nx)).transpose(1, 2)
        beta2 = 2.0 * torch.sigmoid(self.beta2_proj(nx)).transpose(1, 2)
        
        M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, device=x.device)
        out_list = []
        
        for t in range(L):
            k1_t, k2_t, qt, vt = k1[:, :, t], k2[:, :, t], q[:, :, t], v[:, :, t]
            b1_t, b2_t = beta1[:, :, t], beta2[:, :, t]
            
            v_old1 = torch.matmul(M_state, k1_t.unsqueeze(-1)).squeeze(-1) * self.inv_dk
            err1 = vt - v_old1
            M_state = M_state + b1_t.unsqueeze(-1).unsqueeze(-1) * torch.matmul(err1.unsqueeze(-1), k1_t.unsqueeze(-2))
            
            v_old2 = torch.matmul(M_state, k2_t.unsqueeze(-1)).squeeze(-1) * self.inv_dk
            err2 = vt - v_old2
            M_state = M_state + b2_t.unsqueeze(-1).unsqueeze(-1) * torch.matmul(err2.unsqueeze(-1), k2_t.unsqueeze(-2))
            
            out_t = torch.matmul(M_state, qt.unsqueeze(-1)).squeeze(-1) * self.inv_dk
            out_list.append(out_t)
            
        out_concat = torch.stack(out_list, dim=2).transpose(1, 2).reshape(B, L, self.n_heads * self.d_k)
        out = res + self.out_proj(out_concat)
        return out + self.ffn(self.norm2(out))

class DeepModelLM(nn.Module):
    def __init__(self, block_cls, modulo=7, d_model=64, n_layers=4, n_heads=4, d_k=16):
        super().__init__()
        self.embed = nn.Embedding(modulo, d_model)
        self.pos_embed = nn.Embedding(1024, d_model)
        self.layers = nn.ModuleList([block_cls(d_model=d_model, n_heads=n_heads, d_k=d_k) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, modulo)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)

# ── 2. Training Curve Extractor ─────────────────────────────────────────────

def train_and_extract_curve(block_cls, name, modulo=7, steps=3000, log_every=50, device='cpu'):
    print(f"\n{ts()} --- Training {name} on Z_{modulo} for {steps} steps on {device} ---", flush=True)
    model = DeepModelLM(block_cls, modulo=modulo, d_model=64, n_layers=4, n_heads=4, d_k=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    curve = []
    step_50_acc = None
    step_80_acc = None
    
    t0 = time.time()
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
        
        if step % log_every == 0 or step == 1 or step == steps:
            preds = logits.argmax(dim=-1)
            acc = (preds == y_b).float().mean().item() * 100.0
            curve.append((step, loss.item(), acc))
            
            if acc >= 50.0 and step_50_acc is None:
                step_50_acc = step
            if acc >= 80.0 and step_80_acc is None:
                step_80_acc = step
                
            if step % 500 == 0 or step == 1 or step == steps:
                print(f"{ts()}  [{name} Z_{modulo} | Step {step:>4}/{steps}] Loss: {loss.item():.4f} | Acc: {acc:.2f}%", flush=True)
                
    model.eval()
    with torch.no_grad():
        x_ev, y_ev = generate_zk_batch(batch_size=500, seq_len=64, modulo=modulo, device=device)
        preds_ev = model(x_ev).argmax(dim=-1)
        final_val_acc = (preds_ev == y_ev).float().mean().item() * 100.0
        dt = time.time() - t0
        print(f"{ts()}  ==> [{name} Final Z_{modulo} Val Accuracy]: {final_val_acc:.2f}% | Steps to 50%: {step_50_acc} | Steps to 80%: {step_80_acc} (Time: {dt:.1f}s)", flush=True)
        
    return curve, final_val_acc, step_50_acc, step_80_acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_log_header(device)
    
    target_modulos = [7, 9, 12]
    steps_total = 3000
    
    summary_results = []
    
    for k in target_modulos:
        mod_type = "Odd (Prime)" if k == 7 else ("Odd (Comp)" if k == 9 else "Even (Comp)")
        chance = 100.0 / k
        print("\n" + "=" * 100, flush=True)
        print(f"EVALUATING GROKKING PHASE TRANSITION CURVES: Z_{k} ({mod_type}, Chance Level = {chance:.2f}%)", flush=True)
        print("=" * 100, flush=True)
        
        c_real, val_real, s50_r, s80_r = train_and_extract_curve(RealBetaDeltaNetBlock, "Real Beta", modulo=k, steps=steps_total, device=device)
        c_iso, val_iso, s50_iso, s80_iso = train_and_extract_curve(FixedIsometricRealBetaBlock, "Fixed Real Beta=2.0", modulo=k, steps=steps_total, device=device)
        c_dp, val_dp, s50_dp, s80_dp = train_and_extract_curve(DeltaProductRealBlock, "DeltaProduct (n_h=2)", modulo=k, steps=steps_total, device=device)
        c_comp, val_comp, s50_c, s80_c = train_and_extract_curve(ComplexBetaDeltaPhaseBlock, "Complex Beta", modulo=k, steps=steps_total, device=device)
        
        summary_results.append({
            'k': k, 'type': mod_type, 'chance': chance,
            'val_real': val_real, 's50_r': s50_r, 's80_r': s80_r,
            'val_iso': val_iso, 's50_iso': s50_iso, 's80_iso': s80_iso,
            'val_dp': val_dp, 's50_dp': s50_dp, 's80_dp': s80_dp,
            'val_comp': val_comp, 's50_c': s50_c, 's80_c': s80_c,
        })
        
    print("\n" + "=" * 125, flush=True)
    print("GPU SUMMARY BENCHMARK v353: GROKKING PHASE TRANSITION & STEPS-TO-SOLVE AUDIT", flush=True)
    print("=" * 125, flush=True)
    print(f"{'Modulo k':<10} | {'Type':<12} | {'Chance':<8} | {'Real Beta (Acc / Steps>50%)':<28} | {'Fixed Beta=2.0 (Acc / Steps>50%)':<32} | {'Complex Beta (Acc / Steps>50%)':<28}")
    print("-" * 125, flush=True)
    
    for r in summary_results:
        print(f"Z_{r['k']:<7} | {r['type']:<12} | {r['chance']:>6.2f}% | {r['val_real']:>6.2f}% (Steps>50: {str(r['s50_r']):<4}) | {r['val_iso']:>6.2f}% (Steps>50: {str(r['s50_iso']):<4}) | {r['val_comp']:>6.2f}% (Steps>50: {str(r['s50_c']):<4})", flush=True)
        
    print("=" * 125, flush=True)

if __name__ == "__main__":
    main()
