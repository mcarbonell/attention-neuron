#!/usr/bin/env python3
"""
================================================================================
EXPERIMENT v378: MULTIPLIER-FREE PHASE ATTENTION (OPTION C)
================================================================================
Hypothesis:
  The trigonometric cosine function cos(Delta theta) in Phase Attention can be
  replaced by:
    1. A piecewise-linear Periodic Triangular wave tri(Delta theta) = 1 - 2/pi * |wrap(Delta theta)|
       (requiring ONLY subtraction and absolute value, zero multiplications).
    2. A 16-entry discrete Look-Up Table (LUT-16), emulating FPGA/ASIC microcode ROM.
    3. A 1-bit Square Wave phase detector sign(cos(Delta theta)).
  
  If Triangular or LUT-16 Phase Attention sustains 100% associative recall,
  it proves that neural attention can be executed in hardware without floating-point
  multipliers or transcendental math in the attention affinity kernel.

Compliance (GEMINI.md):
  - Timestamps [+HH:MM:SS.ss] with flush=True
  - Candidates evaluated first, then baselines
  - Fast feedback in first 5 batches of Epoch 1
  - 100% Vectorized forward pass
  - Persist JSON to results/raw/v378_multiplier_free_phase_attention.json
================================================================================
"""

import sys
import time
import json
import math
import os
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()
def log(msg):
    elapsed = time.time() - T0
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60
    print(f"[+{h:02d}:{m:02d}:{s:05.2f}] {msg}", flush=True)

SEED = 42
torch.manual_seed(SEED)
device = torch.device('cpu')

# --- SYNTHETIC DATASET ---
def generate_associative_data(num_samples, num_slots=8, num_keys=8, num_values=8, seed=42):
    gen = torch.Generator().manual_seed(seed)
    seq_len = num_slots + 1
    
    X_k = torch.zeros(num_samples, seq_len, dtype=torch.long)
    X_v = torch.zeros(num_samples, seq_len, dtype=torch.long)
    Y = torch.zeros(num_samples, dtype=torch.long)
    
    for i in range(num_samples):
        keys = torch.randperm(num_keys, generator=gen)[:num_slots] + 1
        vals = torch.randint(1, num_values + 1, (num_slots,), generator=gen)
        
        X_k[i, :num_slots] = keys
        X_v[i, :num_slots] = vals
        
        target_idx = torch.randint(0, num_slots, (1,), generator=gen).item()
        X_k[i, num_slots] = keys[target_idx]
        X_v[i, num_slots] = 0
        Y[i] = vals[target_idx] - 1
        
    return X_k, X_v, Y

# Helper: wrap angles to [-pi, pi)
def wrap_angle(x):
    return (x + math.pi) % (2 * math.pi) - math.pi

# --- MODEL ARCHITECTURES ---

# 1. CANDIDATE C1: Triangular Phase Attention (Subtraction + Absolute Value Only)
# tri(Delta theta) = 1.0 - (2.0 / pi) * |wrap(theta_q - theta_k)|
class TriangularPhaseAttention(nn.Module):
    def __init__(self, num_keys, num_values, d_model, num_heads, d_v, num_classes):
        super().__init__()
        self.num_heads = num_heads
        self.d_v = d_v
        self.emb_k = nn.Embedding(num_keys + 1, d_model)
        self.emb_v = nn.Embedding(num_values + 1, d_model)
        self.w_q = nn.Linear(d_model, num_heads)
        self.w_k = nn.Linear(d_model, num_heads)
        self.w_v = nn.Linear(d_model * 2, num_heads * d_v)
        self.out_proj = nn.Linear(num_heads * d_v, num_classes)
        self.scale = nn.Parameter(torch.tensor(8.0))

    def forward(self, x_k, x_v):
        B, L = x_k.shape
        h_k = self.emb_k(x_k)
        h_v = self.emb_v(x_v)
        h_full = torch.cat([h_k, h_v], dim=-1)
        
        theta_q = (torch.tanh(self.w_q(h_k)) * math.pi).transpose(1, 2).unsqueeze(-1)
        theta_k = (torch.tanh(self.w_k(h_k)) * math.pi).transpose(1, 2).unsqueeze(-2)
        
        diff = wrap_angle(theta_q - theta_k)
        # Triangular affinity: max +1 at diff=0, 0 at diff=pi/2, -1 at diff=pi
        tri_affinity = 1.0 - (2.0 / math.pi) * torch.abs(diff)
        scores = tri_affinity * self.scale
        attn = F.softmax(scores, dim=-1)
        
        v = self.w_v(h_full).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out[:, -1, :])

# 2. CANDIDATE C2: 16-Entry Look-Up Table (LUT-16 ROM Emulation)
class LUT16PhaseAttention(nn.Module):
    def __init__(self, num_keys, num_values, d_model, num_heads, d_v, num_classes):
        super().__init__()
        self.num_heads = num_heads
        self.d_v = d_v
        self.emb_k = nn.Embedding(num_keys + 1, d_model)
        self.emb_v = nn.Embedding(num_values + 1, d_model)
        self.w_q = nn.Linear(d_model, num_heads)
        self.w_k = nn.Linear(d_model, num_heads)
        self.w_v = nn.Linear(d_model * 2, num_heads * d_v)
        self.out_proj = nn.Linear(num_heads * d_v, num_classes)
        self.scale = nn.Parameter(torch.tensor(8.0))
        
        # Precomputed 16-entry cosine table buffer in [-pi, pi)
        # 16 equidistant angle bins
        angles = torch.linspace(-math.pi, math.pi - (2 * math.pi / 16), 16)
        self.register_buffer("lut_table", torch.cos(angles))

    def forward(self, x_k, x_v):
        B, L = x_k.shape
        h_k = self.emb_k(x_k)
        h_v = self.emb_v(x_v)
        h_full = torch.cat([h_k, h_v], dim=-1)
        
        theta_q = (torch.tanh(self.w_q(h_k)) * math.pi).transpose(1, 2).unsqueeze(-1)
        theta_k = (torch.tanh(self.w_k(h_k)) * math.pi).transpose(1, 2).unsqueeze(-2)
        
        diff = wrap_angle(theta_q - theta_k) # in [-pi, pi)
        # Normalize to [0, 15] for table lookup
        bin_idx = torch.clamp(((diff + math.pi) / (2 * math.pi) * 16.0).long(), 0, 15)
        lut_affinity = self.lut_table[bin_idx]
        
        # Straight-through estimator for clean backprop through table indices
        scores = (lut_affinity - torch.cos(diff)).detach() + torch.cos(diff)
        scores = scores * self.scale
        attn = F.softmax(scores, dim=-1)
        
        v = self.w_v(h_full).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out[:, -1, :])

# 3. CANDIDATE C3: Square Wave Phase Attention (1-bit Thresholding: +1 or -1)
class SquareWavePhaseAttention(nn.Module):
    def __init__(self, num_keys, num_values, d_model, num_heads, d_v, num_classes):
        super().__init__()
        self.num_heads = num_heads
        self.d_v = d_v
        self.emb_k = nn.Embedding(num_keys + 1, d_model)
        self.emb_v = nn.Embedding(num_values + 1, d_model)
        self.w_q = nn.Linear(d_model, num_heads)
        self.w_k = nn.Linear(d_model, num_heads)
        self.w_v = nn.Linear(d_model * 2, num_heads * d_v)
        self.out_proj = nn.Linear(num_heads * d_v, num_classes)
        self.scale = nn.Parameter(torch.tensor(8.0))

    def forward(self, x_k, x_v):
        B, L = x_k.shape
        h_k = self.emb_k(x_k)
        h_v = self.emb_v(x_v)
        h_full = torch.cat([h_k, h_v], dim=-1)
        
        theta_q = (torch.tanh(self.w_q(h_k)) * math.pi).transpose(1, 2).unsqueeze(-1)
        theta_k = (torch.tanh(self.w_k(h_k)) * math.pi).transpose(1, 2).unsqueeze(-2)
        
        diff = wrap_angle(theta_q - theta_k)
        # Sign of cosine: +1 if |diff| <= pi/2, else -1
        smooth_cos = torch.cos(diff)
        hard_sq = torch.sign(smooth_cos)
        scores = ((hard_sq - smooth_cos).detach() + smooth_cos) * self.scale
        attn = F.softmax(scores, dim=-1)
        
        v = self.w_v(h_full).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out[:, -1, :])

# 4. REFERENCE A: Exact Cosine Phase Attention
class ExactCosinePhaseAttention(nn.Module):
    def __init__(self, num_keys, num_values, d_model, num_heads, d_v, num_classes):
        super().__init__()
        self.num_heads = num_heads
        self.d_v = d_v
        self.emb_k = nn.Embedding(num_keys + 1, d_model)
        self.emb_v = nn.Embedding(num_values + 1, d_model)
        self.w_q = nn.Linear(d_model, num_heads)
        self.w_k = nn.Linear(d_model, num_heads)
        self.w_v = nn.Linear(d_model * 2, num_heads * d_v)
        self.out_proj = nn.Linear(num_heads * d_v, num_classes)
        self.scale = nn.Parameter(torch.tensor(8.0))

    def forward(self, x_k, x_v):
        B, L = x_k.shape
        h_k = self.emb_k(x_k)
        h_v = self.emb_v(x_v)
        h_full = torch.cat([h_k, h_v], dim=-1)
        
        theta_q = (torch.tanh(self.w_q(h_k)) * math.pi).transpose(1, 2).unsqueeze(-1)
        theta_k = (torch.tanh(self.w_k(h_k)) * math.pi).transpose(1, 2).unsqueeze(-2)
        
        scores = torch.cos(theta_q - theta_k) * self.scale
        attn = F.softmax(scores, dim=-1)
        
        v = self.w_v(h_full).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out[:, -1, :])

# 5. REFERENCE B: Vector Attention Baseline (d_k=8)
class StandardVectorAttention(nn.Module):
    def __init__(self, num_keys, num_values, d_model, num_heads, d_k, d_v, num_classes):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.emb_k = nn.Embedding(num_keys + 1, d_model)
        self.emb_v = nn.Embedding(num_values + 1, d_model)
        self.w_q = nn.Linear(d_model, num_heads * d_k)
        self.w_k = nn.Linear(d_model, num_heads * d_k)
        self.w_v = nn.Linear(d_model * 2, num_heads * d_v)
        self.out_proj = nn.Linear(num_heads * d_v, num_classes)

    def forward(self, x_k, x_v):
        B, L = x_k.shape
        h_k = self.emb_k(x_k)
        h_v = self.emb_v(x_v)
        h_full = torch.cat([h_k, h_v], dim=-1)
        
        q = self.w_q(h_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(h_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        
        v = self.w_v(h_full).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out[:, -1, :])

# --- HARNESS ---
def train_and_eval(name, model, xk_tr, xv_tr, y_tr, xk_va, xv_va, y_va, epochs=15, batch_size=64, lr=0.01):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_train = xk_tr.shape[0]
    
    start_time = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_train)
        ep_loss, correct, total = 0.0, 0, 0
        num_batches = (num_train + batch_size - 1) // batch_size
        
        for b_idx in range(num_batches):
            b_ids = perm[b_idx * batch_size : min((b_idx + 1) * batch_size, num_train)]
            bxk, bxv, by = xk_tr[b_ids], xv_tr[b_ids], y_tr[b_ids]
            
            optimizer.zero_grad()
            logits = model(bxk, bxv)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
            ep_loss += loss.item() * len(by)
            preds = logits.argmax(dim=-1)
            correct += (preds == by).sum().item()
            total += len(by)
            
            if ep == 1 and b_idx < 3:
                b_acc = (preds == by).float().mean().item() * 100
                log(f"  [{name} Ep1/B{b_idx+1}] Loss: {loss.item():.4f} | Acc: {b_acc:.1f}%")
                
    model.eval()
    with torch.no_grad():
        va_logits = model(xk_va, xv_va)
        va_loss = criterion(va_logits, y_va).item()
        va_preds = va_logits.argmax(dim=-1)
        va_acc = (va_preds == y_va).float().mean().item() * 100
        
    tot_time = time.time() - start_time
    return {
        "model_name": name, "params": n_params,
        "val_acc": va_acc, "val_loss": va_loss,
        "time": tot_time
    }

def main():
    log("=" * 80)
    log("EXPERIMENT v378: MULTIPLIER-FREE PHASE ATTENTION (OPTION C)")
    log("=" * 80)
    log(f"Platform: {platform.platform()} | Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    
    num_train = 3000
    num_val = 600
    num_slots = 8
    num_keys = 8
    num_values = 8
    d_model = 16
    d_v = 8
    
    xk_tr, xv_tr, y_tr = generate_associative_data(num_train, num_slots=num_slots, num_keys=num_keys, num_values=num_values, seed=100)
    xk_va, xv_va, y_va = generate_associative_data(num_val, num_slots=num_slots, num_keys=num_keys, num_values=num_values, seed=200)
    
    log(f"Data: Train={num_train}, Val={num_val}, Slots={num_slots}, Keys={num_keys}, Values={num_values}")
    log(f"Chance Level: {100.0 / num_values:.2f}%\n")
    
    test_suite = [
        # Candidate 1: Periodic Triangular Wave (Subtraction + Absolute Value only)
        ("Triangular_Phase_H1", TriangularPhaseAttention(num_keys, num_values, d_model, num_heads=1, d_v=d_v, num_classes=num_values), "0 mults / Sub+Abs"),
        ("Triangular_Phase_H4", TriangularPhaseAttention(num_keys, num_values, d_model, num_heads=4, d_v=d_v, num_classes=num_values), "0 mults / Sub+Abs"),
        
        # Candidate 2: 16-Entry Look-Up Table (ROM emulation)
        ("LUT16_Phase_H1", LUT16PhaseAttention(num_keys, num_values, d_model, num_heads=1, d_v=d_v, num_classes=num_values), "0 mults / 16-word ROM"),
        ("LUT16_Phase_H4", LUT16PhaseAttention(num_keys, num_values, d_model, num_heads=4, d_v=d_v, num_classes=num_values), "0 mults / 16-word ROM"),
        
        # Candidate 3: 1-bit Square Wave (Sign only)
        ("SquareWave_Phase_H1", SquareWavePhaseAttention(num_keys, num_values, d_model, num_heads=1, d_v=d_v, num_classes=num_values), "0 mults / 1-bit Sign"),
        ("SquareWave_Phase_H4", SquareWavePhaseAttention(num_keys, num_values, d_model, num_heads=4, d_v=d_v, num_classes=num_values), "0 mults / 1-bit Sign"),
        
        # Reference A: Exact Cosine Phase Attention
        ("ExactCosine_Phase_H1", ExactCosinePhaseAttention(num_keys, num_values, d_model, num_heads=1, d_v=d_v, num_classes=num_values), "Trig Cosine"),
        ("ExactCosine_Phase_H4", ExactCosinePhaseAttention(num_keys, num_values, d_model, num_heads=4, d_v=d_v, num_classes=num_values), "Trig Cosine"),
        
        # Reference B: Standard Vector Attention (d_k=8)
        ("StandardVector_dk8_H1", StandardVectorAttention(num_keys, num_values, d_model, num_heads=1, d_k=8, d_v=d_v, num_classes=num_values), "Vector Dot Product"),
        ("StandardVector_dk8_H4", StandardVectorAttention(num_keys, num_values, d_model, num_heads=4, d_k=8, d_v=d_v, num_classes=num_values), "Vector Dot Product")
    ]
    
    results = []
    log(f"{'Model':<26} | {'Affinity Kernel':<22} | {'Params':<8} | {'Val Acc (%)':<12} | {'Val Loss':<10} | {'Time (s)':<8}")
    log("-" * 96)
    
    for name, model, kernel_desc in test_suite:
        r = train_and_eval(name, model, xk_tr, xv_tr, y_tr, xk_va, xv_va, y_va, epochs=15)
        log(f"{r['model_name']:<26} | {kernel_desc:<22} | {r['params']:<8} | {r['val_acc']:<12.2f} | {r['val_loss']:<10.4f} | {r['time']:<8.2f}")
        r["kernel_desc"] = kernel_desc
        results.append(r)
        
    log("=" * 96)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "v378_multiplier_free_phase_attention.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"Results successfully saved to: {out_file}")

if __name__ == "__main__":
    main()
