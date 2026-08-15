"""
colab_5seed_zk_robustness_benchmark.py
======================================
Official 5-Seed GPU Benchmark for Z_k Grokking & Robustness.
Evaluates 5 Seeds x 4 Arms x 3 Groups (60 Total Runs) on GPU (Tesla T4).

Target Groups:
--------------
- Z_7 (Odd Prime)
- Z_9 (Odd Composite 3^2)
- Z_12 (Even Composite 2^2 * 3)

Arms Tested:
------------
1. Real Beta DeltaNet (beta in (0, 2))
2. Fixed Real Beta=2.0 (Exact Reflection Isometry)
3. DeltaProduct Real (n_h = 2 Real Householders per token)
4. Complex Beta DeltaPhase (beta_t = 1 + e^{i phi_t}, Unitary U(d) Phasors)

Metrics Extracted:
------------------
- Median and IQR (Interquartile Range: 25th - 75th percentile) for Steps-to-50% and Steps-to-80%.
- Failure Count (Number of seeds that NEVER reached 50% or 80% accuracy).
- Configuration Hash locking (locking batch_size=64, seq_len=64, d_model=64, lr=2e-3, steps=3000).
"""

import time, sys, math, hashlib, platform, torch
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

def get_config_hash(modulo, steps, batch_size, seq_len, d_model, lr):
    cfg_str = f"mod={modulo}_steps={steps}_bs={batch_size}_len={seq_len}_dim={d_model}_lr={lr}"
    return hashlib.md5(cfg_str.encode('utf-8')).hexdigest()[:8]

def print_log_header(device):
    print("===============================================================================================", flush=True)
    print("        OFFICIAL 5-SEED GPU BENCHMARK: Z_k ROBUSTNESS & STEPS-TO-SOLVE (MEDIAN & IQR)         ", flush=True)
    print("===============================================================================================", flush=True)
    print(f" Timestamp:              {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", flush=True)
    print(f" Execution Device:       {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f" PyTorch Version:        {torch.__version__}", flush=True)
    print(f" Protocol:               5 Seeds x 4 Arms x 3 Groups (60 Runs Total)", flush=True)
    print(f" Metrics:                Median & IQR of Steps-to-50% / Steps-to-80% & Failure Rate", flush=True)
    print("===============================================================================================\n", flush=True)

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

def run_single_seed(block_cls, arm_name, modulo, seed, steps=3000, device='cpu'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = DeepModelLM(block_cls, modulo=modulo, d_model=64, n_layers=4, n_heads=4, d_k=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    step_50 = None
    step_80 = None
    
    t0_seed = time.time()
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
        
        if step % 50 == 0 or step == 1 or step == steps:
            preds = logits.argmax(dim=-1)
            acc = (preds == y_b).float().mean().item() * 100.0
            if acc >= 50.0 and step_50 is None:
                step_50 = step
            if acc >= 80.0 and step_80 is None:
                step_80 = step
                
            if step % 500 == 0 or step == 1 or step == steps:
                dt_seed = time.time() - t0_seed
                print(f"    {ts()} [{arm_name} Z_{modulo} | Seed {seed} | Step {step:>4}/{steps}] Loss: {loss.item():.4f} | Acc: {acc:.2f}% ({dt_seed:.1f}s)", flush=True)
                
    model.eval()
    with torch.no_grad():
        x_ev, y_ev = generate_zk_batch(batch_size=500, seq_len=64, modulo=modulo, device=device)
        preds_ev = model(x_ev).argmax(dim=-1)
        final_acc = (preds_ev == y_ev).float().mean().item() * 100.0
        
    return final_acc, step_50, step_80

def compute_stats(values, inf_val=99999):
    """Computes median and IQR (25th - 75th percentile). Unreached steps replaced by inf_val for ranking."""
    valid_vals = [v if v is not None else inf_val for v in values]
    med = np.median(valid_vals)
    q25 = np.percentile(valid_vals, 25)
    q75 = np.percentile(valid_vals, 75)
    failures = sum(1 for v in values if v is None)
    return med, q25, q75, failures

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_log_header(device)
    
    seeds = [42, 43, 44, 45, 46]
    modulos = [7, 9, 12]
    arms = [
        ("Real Beta (0,2)", RealBetaDeltaNetBlock),
        ("Fixed Real Beta=2.0", FixedIsometricRealBetaBlock),
        ("DeltaProduct (n_h=2)", DeltaProductRealBlock),
        ("Complex Beta (S^1)", ComplexBetaDeltaPhaseBlock)
    ]
    steps_total = 3000
    
    print(f"Locking Config Hash: {get_config_hash(7, steps_total, 64, 64, 64, 2e-3)}", flush=True)
    print("=" * 115, flush=True)
    
    full_report = {}
    
    for k in modulos:
        mod_type = "Odd (Prime)" if k == 7 else ("Odd (Comp)" if k == 9 else "Even (Comp)")
        print(f"\n" + "=" * 100)
        print(f"RUNNING 5 SEEDS FOR MODULO Z_{k} ({mod_type}, Chance Level = {100.0/k:.2f}%)")
        print("=" * 100, flush=True)
        
        full_report[k] = {}
        
        for arm_name, block_cls in arms:
            accs, s50s, s80s = [], [], []
            t0_arm = time.time()
            print(f"{ts()} Running Arm: {arm_name} across 5 seeds...", flush=True)
            
            for seed in seeds:
                acc, s50, s80 = run_single_seed(block_cls, arm_name, k, seed, steps=steps_total, device=device)
                accs.append(acc)
                s50s.append(s50)
                s80s.append(s80)
                print(f"  [Seed {seed}] Acc: {acc:.2f}% | Steps>50%: {s50} | Steps>80%: {s80}", flush=True)
                
            med_acc = np.median(accs)
            iqr_acc = (np.percentile(accs, 75) - np.percentile(accs, 25))
            
            med_s50, q25_s50, q75_s50, fail_s50 = compute_stats(s50s)
            med_s80, q25_s80, q75_s80, fail_s80 = compute_stats(s80s)
            dt = time.time() - t0_arm
            
            full_report[k][arm_name] = {
                'med_acc': med_acc, 'iqr_acc': iqr_acc, 'accs': accs,
                'med_s50': med_s50, 'iqr_s50': (q25_s50, q75_s50), 'fail_s50': fail_s50,
                'med_s80': med_s80, 'iqr_s80': (q25_s80, q75_s80), 'fail_s80': fail_s80,
                'time': dt
            }
            
    print("\n" + "=" * 135, flush=True)
    print("FINAL 5-SEED CERTIFIED BENCHMARK REPORT: MEDIAN & IQR STEPS-TO-SOLVE", flush=True)
    print("=" * 135, flush=True)
    
    for k in modulos:
        chance = 100.0 / k
        print(f"\n--- MODULO Z_{k} (Chance Level = {chance:.2f}%) ---")
        print(f"{'Arm Name':<26} | {'Val Acc (Med / IQR)':<22} | {'Steps>50% (Med [Q25-Q75] Fail)':<35} | {'Steps>80% (Med [Q25-Q75] Fail)':<35}")
        print("-" * 135)
        for arm_name, _ in arms:
            rep = full_report[k][arm_name]
            acc_str = f"{rep['med_acc']:.2f}% (IQR {rep['iqr_acc']:.2f})"
            s50_str = f"{rep['med_s50']:.0f} [{rep['iqr_s50'][0]:.0f}-{rep['iqr_s50'][1]:.0f}] (Fail: {rep['fail_s50']}/5)"
            s80_str = f"{rep['med_s80']:.0f} [{rep['iqr_s80'][0]:.0f}-{rep['iqr_s80'][1]:.0f}] (Fail: {rep['fail_s80']}/5)"
            print(f"{arm_name:<26} | {acc_str:<22} | {s50_str:<35} | {s80_str:<35}")
        print("-" * 135)
        
if __name__ == "__main__":
    main()
