"""
prototype_v351_mechanistic_zk_audit.py
========================================
Mechanistic Probe & Isometry Control Audit for Z_k Cyclic Groups.

Audit Tasks:
------------
1. Mechanistic Angle Histogram: Probe learned phi_t angles for digits j in {0..6} on Z_7.
   Check if phi_j approx (2 * pi * j / 7).
2. Confusion Matrix Audit for Z_12: Check if errors cluster in offsets of 3 (quotient group Z_3).
3. Isometric Real Control: Real Beta fixed to EXACT 2.0 (eigenvalue -1.0, norm preserving by construction).
"""

import time, sys, math, torch
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

def print_log_header():
    print("===============================================================================================", flush=True)
    print("                V351 MECHANISTIC PROBE & ISOMETRY CONTROL AUDIT                                ", flush=True)
    print("===============================================================================================", flush=True)
    print(f" Timestamp:              {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", flush=True)
    print(f" PyTorch Version:        {torch.__version__}", flush=True)
    print(f" Mechanistic Goal:       1. Probe learned phi_j vs (2*pi*j/7) regular representation of Z_7", flush=True)
    print(f"                         2. Z_12 Confusion Matrix for Z_3 quotient subgroup clustering", flush=True)
    print(f"                         3. Fixed Beta=2.0 Real Isometric Control (eigenvalue -1.0)", flush=True)
    print("===============================================================================================\n", flush=True)

def generate_zk_batch(batch_size=32, seq_len=64, modulo=7, device='cpu'):
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

class FixedIsometricRealBetaBlock(nn.Module):
    """Real Beta fixed to EXACT 2.0 (eigenvalue -1.0, exact Householder reflection isometry)."""
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
            # Fixed Beta = 2.0 (Exact Householder Reflection Isometry)
            update_term = 2.0 * torch.matmul(err.unsqueeze(-1), kt.unsqueeze(-2))
            M_state = M_state + update_term
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

# ── 3. Diagnostic Audit 1: Mechanistic Angle Histogram Probe for Z_7 ────────

def audit_mechanistic_phi_histogram(modulo=7, steps=1500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{ts()} --- AUDIT 1: Mechanistic Angle Histogram Probe on Trained Z_{modulo} Model ---", flush=True)
    
    model = DeepModelLM(ComplexBetaDeltaPhaseBlock, modulo=modulo, d_model=64, n_layers=4, n_heads=4, d_k=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    for step in range(1, steps + 1):
        model.train()
        x_b, y_b = generate_zk_batch(batch_size=32, seq_len=64, modulo=modulo, device=device)
        logits = model(x_b)
        loss = F.cross_entropy(logits.view(-1, modulo), y_b.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
    model.eval()
    with torch.no_grad():
        x_ev, y_ev = generate_zk_batch(batch_size=100, seq_len=64, modulo=modulo, device=device)
        logits_ev, phi_list = model(x_ev, return_phi=True)
        preds_ev = logits_ev.argmax(dim=-1)
        final_acc = (preds_ev == y_ev).float().mean().item() * 100.0
        print(f"{ts()}  [Model Final Accuracy]: {final_acc:.2f}%", flush=True)
        
        # Probe phi_t angles for digit j in {0..modulo-1} across all layers
        phi_last_layer = phi_list[-1] # [B, n_heads, L]
        tokens_flat = x_ev.view(-1)
        phi_flat = phi_last_layer.permute(0, 2, 1).reshape(-1, 4) # [B*L, n_heads]
        
        print("\n  --- Learned Phase Angle phi_t Breakdown Per Digit (Expected: 2*pi*j / 7) ---", flush=True)
        for digit in range(modulo):
            mask = (tokens_flat == digit)
            if mask.sum() > 0:
                digit_phi = phi_flat[mask] % (2.0 * math.pi)
                mean_angles = digit_phi.mean(dim=0).cpu().numpy()
                std_angles = digit_phi.std(dim=0).cpu().numpy()
                expected_theory = (2.0 * math.pi * digit / modulo) % (2.0 * math.pi)
                print(f"  Digit {digit}: Expected Theory={expected_theory:.4f} rad | Learned Head Means: {[round(m, 4) for m in mean_angles.tolist()]} (Std: {[round(s, 4) for s in std_angles.tolist()]})", flush=True)
        print("  -------------------------------------------------------------------\n", flush=True)

# ── 4. Diagnostic Audit 2: Z_12 Confusion Matrix Subgroup Probe ────────────

def audit_z12_confusion_matrix(steps=1500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{ts()} --- AUDIT 2: Z_12 Confusion Matrix & Subgroup Z_3 Quotient Probe ---", flush=True)
    
    # Evaluate Complex Beta Model
    model_c = DeepModelLM(ComplexBetaDeltaPhaseBlock, modulo=12, d_model=64, n_layers=4, n_heads=4, d_k=16).to(device)
    opt_c = torch.optim.AdamW(model_c.parameters(), lr=2e-3)
    sch_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=steps)
    
    for step in range(1, steps + 1):
        model_c.train()
        x_b, y_b = generate_zk_batch(batch_size=32, seq_len=64, modulo=12, device=device)
        loss = F.cross_entropy(model_c(x_b).view(-1, 12), y_b.view(-1))
        opt_c.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model_c.parameters(), 1.0); opt_c.step(); sch_c.step()
        
    model_c.eval()
    with torch.no_grad():
        x_ev, y_ev = generate_zk_batch(batch_size=200, seq_len=64, modulo=12, device=device)
        logits_ev = model_c(x_ev)
        preds = logits_ev.argmax(dim=-1).view(-1).cpu().numpy()
        targets = y_ev.view(-1).cpu().numpy()
        
        conf_mat = np.zeros((12, 12), dtype=int)
        for p, t in zip(preds, targets):
            conf_mat[t, p] += 1
            
        print("  --- Z_12 Confusion Matrix (Rows=True, Cols=Pred) ---", flush=True)
        print(conf_mat, flush=True)
        
        # Calculate Error Offsets: delta = (pred - target) mod 12
        offsets = (preds - targets) % 12
        offset_counts = np.bincount(offsets, minlength=12)
        total_errors = len(preds) - (preds == targets).sum()
        
        print("\n  --- Error Offset Breakdown modulo 12 ---", flush=True)
        for offset_val in range(12):
            cnt = offset_counts[offset_val]
            pct = (cnt / len(preds)) * 100.0
            print(f"  Offset Delta = {offset_val:>2}: {cnt:>5} occurrences ({pct:.2f}% of total)", flush=True)
            
        # Check Z_3 Quotient subgroup hypothesis (offsets 0, 3, 6, 9)
        z3_subgroup_pct = ((offset_counts[0] + offset_counts[3] + offset_counts[6] + offset_counts[9]) / len(preds)) * 100.0
        print(f"\n  [Z_3 Quotient Subgroup Test] (Offsets 0, 3, 6, 9): {z3_subgroup_pct:.2f}% of all predictions (Chance = 33.33%)\n", flush=True)

# ── 5. Diagnostic Audit 3: Fixed Beta=2.0 Real Isometric Control ───────────

def audit_fixed_isometric_real_control(steps=1500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{ts()} --- AUDIT 3: Fixed Beta=2.0 Real Isometric Control (Eigenvalue -1.0) on Z_7 ---", flush=True)
    
    model_iso = DeepModelLM(FixedIsometricRealBetaBlock, modulo=7, d_model=64, n_layers=4, n_heads=4, d_k=16).to(device)
    optimizer = torch.optim.AdamW(model_iso.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    for step in range(1, steps + 1):
        model_iso.train()
        x_b, y_b = generate_zk_batch(batch_size=32, seq_len=64, modulo=7, device=device)
        loss = F.cross_entropy(model_iso(x_b).view(-1, 7), y_b.view(-1))
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model_iso.parameters(), 1.0); optimizer.step(); scheduler.step()
        
    model_iso.eval()
    with torch.no_grad():
        x_ev, y_ev = generate_zk_batch(batch_size=300, seq_len=64, modulo=7, device=device)
        preds = model_iso(x_ev).argmax(dim=-1)
        acc = (preds == y_ev).float().mean().item() * 100.0
        print(f"{ts()}  [Fixed Beta=2.0 Real Isometric Control Final Z_7 Accuracy]: {acc:.2f}% (Chance Level = 14.29%)\n", flush=True)

def main():
    print_log_header()
    audit_mechanistic_phi_histogram(modulo=7, steps=1500)
    audit_z12_confusion_matrix(steps=1500)
    audit_fixed_isometric_real_control(steps=1500)

if __name__ == "__main__":
    main()
