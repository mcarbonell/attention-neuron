#!/usr/bin/env python3
"""
================================================================================
EXPERIMENT v376: ANGULAR CAPACITY FRONTIER IN PHASE ATTENTION
================================================================================
Hypothesis:
  In the unitary complex circle U(1) and C^1, how many discrete keys can a single
  scalar phase head (H=1) resolve before angular crowding degrades accuracy?
  Does multi-head phase attention (H=2, H=4) scale capacity proportionally by
  partitioning the keys across multiple phase circles?
  How does Real Scalar Attention (R^1) degrade as K scales from 8 to 64?

Design:
  - Sweep K in {8, 16, 32, 64} keys in direct content retrieval.
  - Architectures:
      1. PhaseOnly (H=1, H=2, H=4)
      2. ComplexScalar (H=1, H=2, H=4)
      3. RealScalar (H=1, H=4)
      4. StandardVector baseline (H=1, H=4, d_k=8)
  - 100% Vectorized forward pass.
  - Fast feedback in first 5 batches of Epoch 1.
  - Persist JSON results to results/raw/v376_angular_capacity_stress.json
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

# --- SYNTHETIC DATA GENERATOR ---
def generate_capacity_data(num_samples, num_slots, num_keys, num_values, seed=42):
    gen = torch.Generator().manual_seed(seed)
    X_k = torch.zeros(num_samples, num_slots + 1, dtype=torch.long)
    X_v = torch.zeros(num_samples, num_slots + 1, dtype=torch.long)
    Y = torch.zeros(num_samples, dtype=torch.long)
    
    for i in range(num_samples):
        # Sample num_slots distinct keys out of num_keys
        keys = torch.randperm(num_keys, generator=gen)[:num_slots] + 1
        vals = torch.randint(1, num_values + 1, (num_slots,), generator=gen)
        
        X_k[i, :num_slots] = keys
        X_v[i, :num_slots] = vals
        
        # Pick target slot for query
        target_idx = torch.randint(0, num_slots, (1,), generator=gen).item()
        X_k[i, num_slots] = keys[target_idx]
        X_v[i, num_slots] = 0 # masked
        
        Y[i] = vals[target_idx] - 1
        
    return X_k, X_v, Y

# --- MODELS ---

class PhaseOnlyAttention(nn.Module):
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

class ComplexScalarAttention(nn.Module):
    def __init__(self, num_keys, num_values, d_model, num_heads, d_v, num_classes):
        super().__init__()
        self.num_heads = num_heads
        self.d_v = d_v
        self.emb_k = nn.Embedding(num_keys + 1, d_model)
        self.emb_v = nn.Embedding(num_values + 1, d_model)
        
        self.w_q_re = nn.Linear(d_model, num_heads)
        self.w_q_im = nn.Linear(d_model, num_heads)
        self.w_k_re = nn.Linear(d_model, num_heads)
        self.w_k_im = nn.Linear(d_model, num_heads)
        
        self.w_v = nn.Linear(d_model * 2, num_heads * d_v)
        self.out_proj = nn.Linear(num_heads * d_v, num_classes)
        self.scale = nn.Parameter(torch.tensor(6.0))

    def forward(self, x_k, x_v):
        B, L = x_k.shape
        h_k = self.emb_k(x_k)
        h_v = self.emb_v(x_v)
        h_full = torch.cat([h_k, h_v], dim=-1)
        
        q_re = self.w_q_re(h_k).transpose(1, 2).unsqueeze(-1)
        q_im = self.w_q_im(h_k).transpose(1, 2).unsqueeze(-1)
        k_re = self.w_k_re(h_k).transpose(1, 2).unsqueeze(-2)
        k_im = self.w_k_im(h_k).transpose(1, 2).unsqueeze(-2)
        
        scores = (q_re * k_re + q_im * k_im) * self.scale
        attn = F.softmax(scores, dim=-1)
        
        v = self.w_v(h_full).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out[:, -1, :])

class RealScalarAttention(nn.Module):
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
        self.scale = nn.Parameter(torch.tensor(4.0))

    def forward(self, x_k, x_v):
        B, L = x_k.shape
        h_k = self.emb_k(x_k)
        h_v = self.emb_v(x_v)
        h_full = torch.cat([h_k, h_v], dim=-1)
        
        q = self.w_q(h_k).transpose(1, 2).unsqueeze(-1)
        k = self.w_k(h_k).transpose(1, 2).unsqueeze(-2)
        
        scores = (q * k) * self.scale
        attn = F.softmax(scores, dim=-1)
        
        v = self.w_v(h_full).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out[:, -1, :])

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
def train_and_eval(name, model, xk_tr, xv_tr, y_tr, xk_va, xv_va, y_va, epochs=15, batch_size=64, lr=0.01, verbose=True):
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
            
            if ep == 1 and b_idx < 3 and verbose:
                b_acc = (preds == by).float().mean().item() * 100
                log(f"  [{name} Ep1/B{b_idx+1}] Loss: {loss.item():.4f} | Acc: {b_acc:.1f}%")
                
    # Final eval
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
    log("EXPERIMENT v376: ANGULAR CAPACITY FRONTIER IN PHASE ATTENTION")
    log("=" * 80)
    log(f"Platform: {platform.platform()} | Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    
    k_sweep = [8, 16, 32, 64]
    num_train = 2400
    num_val = 600
    d_model = 16
    d_v = 8
    
    all_results = {}
    
    for K in k_sweep:
        num_slots = min(K, 8) # Active items in context
        chance = 100.0 / K
        log(f"\n================================================================================")
        log(f">>> BENCHMARK SWEEP: K = {K} KEYS | Slots = {num_slots} | Chance Acc = {chance:.2f}% <<<")
        log(f"================================================================================")
        
        xk_tr, xv_tr, y_tr = generate_capacity_data(num_train, num_slots=num_slots, num_keys=K, num_values=K, seed=100 + K)
        xk_va, xv_va, y_va = generate_capacity_data(num_val, num_slots=num_slots, num_keys=K, num_values=K, seed=200 + K)
        
        k_results = []
        
        # 1. Candidate A: Phase-Only U(1) with H=1, H=2, H=4
        for H in [1, 2, 4]:
            m = PhaseOnlyAttention(K, K, d_model, num_heads=H, d_v=d_v, num_classes=K)
            r = train_and_eval(f"PhaseOnly_U1_H{H}", m, xk_tr, xv_tr, y_tr, xk_va, xv_va, y_va, epochs=15)
            log(f"  [Result K={K:2d}] {r['model_name']:<18} | Acc: {r['val_acc']:6.2f}% | Loss: {r['val_loss']:.4f} | Params: {r['params']}")
            k_results.append(r)
            
        # 2. Candidate B: ComplexScalar C^1 with H=1, H=2, H=4
        for H in [1, 2, 4]:
            m = ComplexScalarAttention(K, K, d_model, num_heads=H, d_v=d_v, num_classes=K)
            r = train_and_eval(f"ComplexScalar_C1_H{H}", m, xk_tr, xv_tr, y_tr, xk_va, xv_va, y_va, epochs=15)
            log(f"  [Result K={K:2d}] {r['model_name']:<18} | Acc: {r['val_acc']:6.2f}% | Loss: {r['val_loss']:.4f} | Params: {r['params']}")
            k_results.append(r)
            
        # 3. Candidate C: RealScalar R^1 with H=1, H=4
        for H in [1, 4]:
            m = RealScalarAttention(K, K, d_model, num_heads=H, d_v=d_v, num_classes=K)
            r = train_and_eval(f"RealScalar_R1_H{H}", m, xk_tr, xv_tr, y_tr, xk_va, xv_va, y_va, epochs=15)
            log(f"  [Result K={K:2d}] {r['model_name']:<18} | Acc: {r['val_acc']:6.2f}% | Loss: {r['val_loss']:.4f} | Params: {r['params']}")
            k_results.append(r)
            
        # 4. Baseline: StandardVector with H=1 (d_k=8), H=4 (d_k=8)
        for H in [1, 4]:
            m = StandardVectorAttention(K, K, d_model, num_heads=H, d_k=8, d_v=d_v, num_classes=K)
            r = train_and_eval(f"StandardVector_dk8_H{H}", m, xk_tr, xv_tr, y_tr, xk_va, xv_va, y_va, epochs=15)
            log(f"  [Result K={K:2d}] {r['model_name']:<18} | Acc: {r['val_acc']:6.2f}% | Loss: {r['val_loss']:.4f} | Params: {r['params']}")
            k_results.append(r)
            
        all_results[f"K_{K}"] = k_results
        
    # --- GLOBAL SUMMARY TABLE ---
    log("\n" + "=" * 95)
    log("GLOBAL CAPACITY SUMMARY: VALIDATION ACCURACY (%) ACROSS VOCABULARY K")
    log("=" * 95)
    header = f"{'Model':<24} | " + " | ".join([f"K={K} (ch={100/K:.1f}%)" for K in k_sweep])
    log(header)
    log("-" * 95)
    
    model_names = [
        "PhaseOnly_U1_H1", "PhaseOnly_U1_H2", "PhaseOnly_U1_H4",
        "ComplexScalar_C1_H1", "ComplexScalar_C1_H2", "ComplexScalar_C1_H4",
        "RealScalar_R1_H1", "RealScalar_R1_H4",
        "StandardVector_dk8_H1", "StandardVector_dk8_H4"
    ]
    
    for name in model_names:
        row = f"{name:<24} | "
        cols = []
        for K in k_sweep:
            matches = [r for r in all_results[f"K_{K}"] if r["model_name"] == name]
            if matches:
                cols.append(f"{matches[0]['val_acc']:12.2f}%")
            else:
                cols.append(f"{'N/A':>12}")
        row += " | ".join(cols)
        log(row)
    log("=" * 95)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "v376_angular_capacity_stress.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    log(f"Results successfully saved to: {out_file}")

if __name__ == "__main__":
    main()
