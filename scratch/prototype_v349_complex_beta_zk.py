"""
prototype_v349_complex_beta_zk.py
==================================
Complex Beta-Gated Householder Reflection Core (beta_t = 1 + e^{i phi_t})
Unlocking Native Z_k Cyclic Group Arithmetic & Modular Reasoning.

Theory:
-------
Real Householder beta=2 yields eigenvalue -1 (Z_2 parity, counting mod 2).
Complex Generalized Householder:
    H_t = I - (1 + e^{i phi_t}) * (k_t k_t^*) / ||k_t||^2
Yields eigenvalue -e^{i phi_t} on the unit circle S^1 with arbitrary phase angle phi_t.
This unlocks native Z_k cyclic group counting modulo k in a single step per token.
"""

import time, sys, math, torch
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
    print("                      V349 COMPLEX BETA Z_k CYCLIC GROUP BENCHMARK                            ", flush=True)
    print("===============================================================================================", flush=True)
    print(f" Timestamp:              {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", flush=True)
    print(f" PyTorch Version:        {torch.__version__}", flush=True)
    print(f" Target Hypothesis:      Z_k Native Cyclic Group Counting via Complex Beta_t = 1 + e^{{i phi_t}}", flush=True)
    print(f" Householder Eigenvalue: -e^{{i phi_t}} on S^1 (Unit Circle, |autovalor| = 1.0)", flush=True)
    print("===============================================================================================\n", flush=True)

# ── 1. Z_k Modular Addition Sequence Generator ─────────────────────────────

def generate_zk_modular_batch(batch_size=32, seq_len=64, modulo=7, device='cpu'):
    """
    Generates sequence of modular additions in Z_k:
    Input: [a_1, a_2, a_3, ..., a_L] where a_i in {0, 1, ..., k-1}
    Target: cumulative sum modulo k: (sum_{i=1}^L a_i) mod k
    """
    elements = torch.randint(0, modulo, (batch_size, seq_len), device=device)
    target_cumsum = torch.cumsum(elements, dim=1) % modulo
    
    # Vocab mapping: 0..k-1 for digits, k for equal marker
    inputs = elements
    targets = target_cumsum
    return inputs, targets

# ── 2. Complex Beta Householder Core (beta_t = 1 + e^{i phi_t}) ─────────────

class ComplexBetaDeltaPhaseBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=2, d_k=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.inv_dk = 1.0 / float(d_k)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.theta_k_proj = nn.Linear(d_model, n_heads * d_k)
        self.theta_q_proj = nn.Linear(d_model, n_heads * d_k)
        self.val_proj = nn.Linear(d_model, n_heads * d_k)
        
        # Complex Beta Phase Angle phi_t (1 + e^{i phi_t})
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
        nx = self.norm1(x)
        B, L, D = nx.shape
        
        theta_k = self.theta_k_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        theta_q = self.theta_q_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.val_proj(nx).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        phi_beta = self.phi_beta_proj(nx).transpose(1, 2) # [B, n_heads, L]
        
        # Complex Phasors K_t, Q_t on S^1
        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)
        
        # Complex Beta_t = 1 + e^{i phi_t}
        beta_complex = 1.0 + torch.polar(torch.ones_like(phi_beta), phi_beta) # [B, n_heads, L]
        
        M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=x.device)
        out_list = []
        
        for t in range(L):
            kt = K[:, :, t]       # [B, n_heads, d_k]
            qt = Q[:, :, t]       # [B, n_heads, d_k]
            vt = v[:, :, t]       # [B, n_heads, d_k]
            b_t = beta_complex[:, :, t] # [B, n_heads]
            
            # v_old = 1/d_k * Re(M_state * K_t^*)
            v_old = torch.matmul(M_state, torch.conj(kt).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            err = vt - v_old
            
            # Generalized Householder Update with Complex Beta: M_t = M_{t-1} + b_t * (err x K_t)
            err_c = err.to(torch.complex64)
            update_term = b_t.unsqueeze(-1).unsqueeze(-1) * torch.matmul(err_c.unsqueeze(-1), kt.unsqueeze(-2))
            M_state = M_state + update_term
            
            # Readout with Query Q_t
            out_t = torch.matmul(M_state, torch.conj(qt).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            out_list.append(out_t)
            
        out_concat = torch.stack(out_list, dim=2).transpose(1, 2).reshape(B, L, self.n_heads * self.d_k)
        out = res + self.out_proj(out_concat)
        return out + self.ffn(self.norm2(out))

class RealBetaDeltaNetBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=2, d_k=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.inv_dk = 1.0 / float(d_k)
        
        self.norm1 = nn.LayerNorm(d_model)
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
        nx = self.norm1(x)
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

class ModelLM(nn.Module):
    def __init__(self, block_cls, modulo=7, d_model=64, n_heads=2, d_k=16):
        super().__init__()
        self.embed = nn.Embedding(modulo, d_model)
        self.block1 = block_cls(d_model=d_model, n_heads=n_heads, d_k=d_k)
        self.block2 = block_cls(d_model=d_model, n_heads=n_heads, d_k=d_k)
        self.head = nn.Linear(d_model, modulo)

    def forward(self, x):
        h = self.embed(x)
        h = self.block1(h)
        h = self.block2(h)
        return self.head(h)

def evaluate_zk_addition(block_cls, name, modulo=7, steps=600):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{ts()} --- Evaluating {name} on Z_{modulo} Modular Addition ---", flush=True)
    
    model = ModelLM(block_cls, modulo=modulo, d_model=64, n_heads=2, d_k=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    
    for step in range(1, steps + 1):
        model.train()
        x_b, y_b = generate_zk_modular_batch(batch_size=32, seq_len=64, modulo=modulo, device=device)
        
        logits = model(x_b)
        loss = F.cross_entropy(logits.view(-1, modulo), y_b.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if step % 150 == 0 or step == 1 or step == steps:
            preds = logits.argmax(dim=-1)
            acc = (preds == y_b).float().mean().item() * 100.0
            print(f"{ts()}  [Step {step:>4}/{steps}] Loss: {loss.item():.4f} | Z_{modulo} Acc: {acc:.2f}%", flush=True)
            
    model.eval()
    with torch.no_grad():
        x_ev, y_ev = generate_zk_modular_batch(batch_size=200, seq_len=64, modulo=modulo, device=device)
        logits_ev = model(x_ev)
        preds_ev = logits_ev.argmax(dim=-1)
        final_acc = (preds_ev == y_ev).float().mean().item() * 100.0
        print(f"{ts()}  [{name} Final Z_{modulo} Val Accuracy]: {final_acc:.2f}%\n", flush=True)
    return final_acc

def main():
    print_log_header()
    
    modulo = 7
    acc_real = evaluate_zk_addition(RealBetaDeltaNetBlock, name="Real Beta DeltaNet (Real Eigenvalue 1-beta in Z_2)", modulo=modulo)
    acc_complex = evaluate_zk_addition(ComplexBetaDeltaPhaseBlock, name="Complex Beta DeltaPhase (Complex Eigenvalue -e^{i phi} in Z_k)", modulo=modulo)
    
    print("=" * 95, flush=True)
    print(f"Z_{modulo} MODULAR ARITHMETIC BENCHMARK RESULTS SUMMARY", flush=True)
    print("=" * 95, flush=True)
    print(f"Chance Level Baseline: 1/{modulo} = {100.0/modulo:.2f}%")
    print(f"1. Real Beta DeltaNet   (Eigenvalues in Z_2): {acc_real:.2f}%")
    print(f"2. Complex Beta DeltaPhase (Eigenvalues in Z_k): {acc_complex:.2f}% (Gap: {acc_complex - acc_real:+.2f}%)")
    print("=" * 95, flush=True)

if __name__ == "__main__":
    main()
