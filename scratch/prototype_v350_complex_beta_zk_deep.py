"""
prototype_v350_complex_beta_zk_deep.py
======================================
Deep 4-Layer Architecture Benchmark for Z_k Cyclic Group Arithmetic (Z_7 and Z_12).
Evaluates Real Beta DeltaNet vs Complex Beta DeltaPhase (beta_t = 1 + e^{i phi_t}).

Metadata:
---------
- Harness: Modular Arithmetic Sequence Task (seq_len=64, Z_5, Z_7, Z_12)
- Architecture: 4-Layer Residual Model with ShortCausalConv1D (k=4) & FFN
- Optimization: AdamW (lr=2e-3, CosineAnnealingLR, steps=1500, batch_size=32)
"""

import time, sys, math, platform, torch
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
    print("                V350 DEEP 4-LAYER COMPLEX BETA Z_k CYCLIC GROUP BENCHMARK                      ", flush=True)
    print("===============================================================================================", flush=True)
    print(f" Timestamp:              {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", flush=True)
    print(f" Platform / Python:      {platform.platform()} | Python {sys.version.split()[0]}", flush=True)
    print(f" PyTorch Version:        {torch.__version__}", flush=True)
    print(f" Target Hypothesis:      Z_k Native Cyclic Group Representation via Complex Beta_t = 1 + e^{{i phi_t}}", flush=True)
    print(f" Architecture:          4-Layer Residual Model with ShortCausalConv1D (kernel=4) & FFN", flush=True)
    print(f" Optimization:          AdamW (lr=2e-3, CosineAnnealingLR, steps=1500, batch_size=32)", flush=True)
    print("===============================================================================================\n", flush=True)

# ── 1. Z_k Modular Addition Sequence Generator ─────────────────────────────

def generate_zk_modular_batch(batch_size=32, seq_len=64, modulo=7, device='cpu'):
    elements = torch.randint(0, modulo, (batch_size, seq_len), device=device)
    target_cumsum = torch.cumsum(elements, dim=1) % modulo
    return elements, target_cumsum

# ── 2. Building Blocks ──────────────────────────────────────────────────────

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
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.inv_dk = 1.0 / float(d_k)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.theta_k_proj = nn.Linear(d_model, n_heads * d_k)
        self.theta_q_proj = nn.Linear(d_model, n_heads * d_k)
        self.val_proj = nn.Linear(d_model, n_heads * d_k)
        self.phi_beta_proj = nn.Linear(d_model, n_heads)
        
        self.out_proj = nn.Linear(n_heads * d_k, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model)
        )

    def forward(self, x):
        res = x
        nx = self.conv(self.norm1(x))
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
            kt = K[:, :, t]
            qt = Q[:, :, t]
            vt = v[:, :, t]
            b_t = beta_complex[:, :, t]
            
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

class RealBetaDeltaNetBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_k=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.inv_dk = 1.0 / float(d_k)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.k_proj = nn.Linear(d_model, n_heads * d_k)
        self.q_proj = nn.Linear(d_model, n_heads * d_k)
        self.val_proj = nn.Linear(d_model, n_heads * d_k)
        self.beta_proj = nn.Linear(d_model, n_heads)
        
        self.out_proj = nn.Linear(n_heads * d_k, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model)
        )

    def forward(self, x):
        res = x
        nx = self.conv(self.norm1(x))
        B, L, D = nx.shape
        
        k = self.k_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        q = self.q_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.val_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        beta_real = 2.0 * torch.sigmoid(self.beta_proj(nx)).transpose(1, 2)
        
        M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, device=x.device)
        out_list = []
        
        for t in range(L):
            kt = k[:, :, t]
            qt = q[:, :, t]
            vt = v[:, :, t]
            b_t = beta_real[:, :, t]
            
            v_old = torch.matmul(M_state, kt.unsqueeze(-1)).squeeze(-1) * self.inv_dk
            err = vt - v_old
            
            update_term = b_t.unsqueeze(-1).unsqueeze(-1) * torch.matmul(err.unsqueeze(-1), kt.unsqueeze(-2))
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

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)

def evaluate_zk_addition(block_cls, name, modulo=7, steps=1500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{ts()} --- Evaluating Deep 4-Layer {name} on Z_{modulo} Modular Addition (steps={steps}) ---", flush=True)
    
    model = DeepModelLM(block_cls, modulo=modulo, d_model=64, n_layers=4, n_heads=4, d_k=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        x_b, y_b = generate_zk_modular_batch(batch_size=32, seq_len=64, modulo=modulo, device=device)
        
        logits = model(x_b)
        loss = F.cross_entropy(logits.view(-1, modulo), y_b.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if step % 300 == 0 or step == 1 or step == steps:
            preds = logits.argmax(dim=-1)
            acc = (preds == y_b).float().mean().item() * 100.0
            print(f"{ts()}  [Step {step:>4}/{steps}] Loss: {loss.item():.4f} | Z_{modulo} Acc: {acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)
            
    model.eval()
    with torch.no_grad():
        x_ev, y_ev = generate_zk_modular_batch(batch_size=300, seq_len=64, modulo=modulo, device=device)
        logits_ev = model(x_ev)
        preds_ev = logits_ev.argmax(dim=-1)
        final_acc = (preds_ev == y_ev).float().mean().item() * 100.0
        dt = time.time() - t0
        print(f"{ts()}  [{name} Final Z_{modulo} Val Accuracy]: {final_acc:.2f}% (Time: {dt:.2f}s)\n", flush=True)
    return final_acc

def main():
    print_log_header()
    
    for modulo in [7, 12]:
        print(f"\n" + "=" * 80)
        print(f"EVALUATING MODULAR GROUP Z_{modulo} (Chance Level: {100.0/modulo:.2f}%)")
        print("=" * 80)
        acc_real = evaluate_zk_addition(RealBetaDeltaNetBlock, name="Real Beta DeltaNet (Real Eigenvalues Z_2)", modulo=modulo)
        acc_complex = evaluate_zk_addition(ComplexBetaDeltaPhaseBlock, name="Complex Beta DeltaPhase (Complex Eigenvalues Z_k)", modulo=modulo)
        
        print(f"Summary Z_{modulo}: Real Beta = {acc_real:.2f}% | Complex Beta = {acc_complex:.2f}% (Gap: {acc_complex - acc_real:+.2f}%)")
        print("=" * 80)

if __name__ == "__main__":
    main()
