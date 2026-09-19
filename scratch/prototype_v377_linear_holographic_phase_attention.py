#!/usr/bin/env python3
"""
================================================================================
EXPERIMENT v377: LINEAR HOLOGRAPHIC PHASE ATTENTION O(N) (OPTION B)
================================================================================
Theoretical Insight:
  Cosine affinity cos(theta_q - theta_k) is an EXACT separable rank-2 kernel:
      cos(theta_q - theta_k) = [cos(theta_q), sin(theta_q)] @ [cos(theta_k), sin(theta_k)]^T
                             = phi(q)^T phi(k)
  
  This means Phase Attention can be computed in EXACT O(N) time with an O(1)
  recurrent state S_t in R^(2 x d_v) per head, WITHOUT ANY Taylor expansion or
  kernel approximation!

Hypothesis:
  1. Pure wave superposition (LinearHolographicPhase) will eliminate the O(N^2)
     attention matrix while maintaining associative recall through constructive/destructive
     phase interference.
  2. Delta-rule phase update (DeltaPhaseLinear) will prevent memory saturation in
     recurrent accumulation.
  3. Causal linear phase attention will match or beat standard real Linear Attention (ELU+1)
     at a fraction of the state size.

Harness & Traceability (GEMINI.md):
  - Timestamps [+HH:MM:SS.ss] with flush=True
  - Parameter inventory and model configurations
  - Fast feedback in first 5 batches of Epoch 1
  - Persistence to results/raw/v377_linear_holographic_phase_attention.json
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

# --- DATASET: CAUSAL ASSOCIATIVE RECALL ---
def generate_causal_data(num_samples, num_slots=8, num_keys=8, num_values=8, seed=42):
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
        
        # Query at last slot
        target_idx = torch.randint(0, num_slots, (1,), generator=gen).item()
        X_k[i, num_slots] = keys[target_idx]
        X_v[i, num_slots] = 0 # masked
        
        Y[i] = vals[target_idx] - 1
        
    return X_k, X_v, Y

# --- ARCHITECTURES ---

# 1. BASELINE A: Quadratic Causal Softmax Phase Attention O(N^2)
class SoftmaxPhaseAttention(nn.Module):
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
        # Causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=scores.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        
        v = self.w_v(h_full).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out[:, -1, :])

# 2. CANDIDATE B1: Linear Holographic Phase Attention O(N) (Pure Wave Superposition)
# State S_t = S_(t-1) + phi(k_t) v_t^T in R^(2 x d_v) per head
class LinearHolographicPhaseAttention(nn.Module):
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

    def forward(self, x_k, x_v):
        B, L = x_k.shape
        h_k = self.emb_k(x_k)
        h_v = self.emb_v(x_v)
        h_full = torch.cat([h_k, h_v], dim=-1)
        
        theta_q = torch.tanh(self.w_q(h_k)) * math.pi # (B, L, H)
        theta_k = torch.tanh(self.w_k(h_k)) * math.pi
        
        # phi(k): (B, H, L, 2)
        phi_k = torch.stack([torch.cos(theta_k), torch.sin(theta_k)], dim=-1).transpose(1, 2)
        phi_q = torch.stack([torch.cos(theta_q), torch.sin(theta_q)], dim=-1).transpose(1, 2)
        
        # v: (B, H, L, d_v)
        v = self.w_v(h_full).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        
        # O(N) Causal prefix sum:
        # kv_pairs: phi_k^T * v -> outer product of (2,) and (d_v,) = (B, H, L, 2, d_v)
        kv_pairs = phi_k.unsqueeze(-1) * v.unsqueeze(-2) # (B, H, L, 2, d_v)
        # Cumulative state accumulation
        state = torch.cumsum(kv_pairs, dim=2) # (B, H, L, 2, d_v)
        
        # Retrieval: phi_q @ state
        # phi_q: (B, H, L, 1, 2) @ (B, H, L, 2, d_v) -> (B, H, L, 1, d_v)
        out = torch.matmul(phi_q.unsqueeze(-2), state).squeeze(-2) # (B, H, L, d_v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out[:, -1, :])

# 3. CANDIDATE B2: Linear Delta-Phase Attention O(N) (Recurrent Error-Correcting Delta Rule)
# S_t = S_(t-1) + beta * (v_t - phi(k_t)^T S_(t-1)) outer phi(k_t)
class DeltaPhaseLinearAttention(nn.Module):
    def __init__(self, num_keys, num_values, d_model, num_heads, d_v, num_classes):
        super().__init__()
        self.num_heads = num_heads
        self.d_v = d_v
        self.emb_k = nn.Embedding(num_keys + 1, d_model)
        self.emb_v = nn.Embedding(num_values + 1, d_model)
        self.w_q = nn.Linear(d_model, num_heads)
        self.w_k = nn.Linear(d_model, num_heads)
        self.w_v = nn.Linear(d_model * 2, num_heads * d_v)
        self.w_beta = nn.Linear(d_model, num_heads) # learned write rate per step
        self.out_proj = nn.Linear(num_heads * d_v, num_classes)

    def forward(self, x_k, x_v):
        B, L = x_k.shape
        h_k = self.emb_k(x_k)
        h_v = self.emb_v(x_v)
        h_full = torch.cat([h_k, h_v], dim=-1)
        
        theta_q = torch.tanh(self.w_q(h_k)) * math.pi
        theta_k = torch.tanh(self.w_k(h_k)) * math.pi
        beta = torch.sigmoid(self.w_beta(h_k)) # (B, L, H) in [0, 1]
        
        phi_k = torch.stack([torch.cos(theta_k), torch.sin(theta_k)], dim=-1).transpose(1, 2) # (B, H, L, 2)
        phi_q = torch.stack([torch.cos(theta_q), torch.sin(theta_q)], dim=-1).transpose(1, 2)
        v = self.w_v(h_full).view(B, L, self.num_heads, self.d_v).transpose(1, 2) # (B, H, L, d_v)
        beta = beta.transpose(1, 2).unsqueeze(-1).unsqueeze(-1) # (B, H, L, 1, 1)
        
        # Sequential recurrent scan for Delta Rule
        # State S is (B, H, 2, d_v)
        state = torch.zeros(B, self.num_heads, 2, self.d_v, device=x_k.device)
        outs = []
        
        for t in range(L):
            pk_t = phi_k[:, :, t, :].unsqueeze(-2) # (B, H, 1, 2)
            v_t = v[:, :, t, :]                   # (B, H, d_v)
            b_t = beta[:, :, t, :, :]             # (B, H, 1, 1)
            
            # Predict existing: pk_t @ S -> (B, H, 1, d_v)
            pred_v = torch.matmul(pk_t, state).squeeze(-2) # (B, H, d_v)
            err = v_t - pred_v
            
            # Delta update: S = S + beta * (pk_t^T @ err)
            delta = torch.matmul(pk_t.transpose(-2, -1), err.unsqueeze(-2)) # (B, H, 2, d_v)
            state = state + b_t * delta
            
            # Readout with query at current step
            pq_t = phi_q[:, :, t, :].unsqueeze(-2) # (B, H, 1, 2)
            y_t = torch.matmul(pq_t, state).squeeze(-2) # (B, H, d_v)
            outs.append(y_t)
            
        out_seq = torch.stack(outs, dim=2) # (B, H, L, d_v)
        out = out_seq.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out[:, -1, :])

# 4. BASELINE B: Standard Real Linear Attention (ELU+1 feature map)
class StandardRealLinearAttention(nn.Module):
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
        
        q = F.elu(self.w_q(h_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)) + 1.0 # (B, H, L, d_k)
        k = F.elu(self.w_k(h_k).view(B, L, self.num_heads, self.d_k).transpose(1, 2)) + 1.0
        v = self.w_v(h_full).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        
        # kv prefix sum in R^(d_k x d_v)
        kv = k.unsqueeze(-1) * v.unsqueeze(-2) # (B, H, L, d_k, d_v)
        state = torch.cumsum(kv, dim=2)
        
        # z accumulator for denominator
        z = torch.cumsum(k, dim=2) # (B, H, L, d_k)
        
        num = torch.matmul(q.unsqueeze(-2), state).squeeze(-2) # (B, H, L, d_v)
        den = (q * z).sum(dim=-1, keepdim=True).clamp(min=1e-5) # (B, H, L, 1)
        
        out = (num / den).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
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
    log("EXPERIMENT v377: LINEAR HOLOGRAPHIC PHASE ATTENTION O(N)")
    log("=" * 80)
    log(f"Platform: {platform.platform()} | Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    
    num_train = 3000
    num_val = 600
    num_slots = 8
    num_keys = 8
    num_values = 8
    d_model = 16
    d_v = 8
    
    xk_tr, xv_tr, y_tr = generate_causal_data(num_train, num_slots=num_slots, num_keys=num_keys, num_values=num_values, seed=100)
    xk_va, xv_va, y_va = generate_causal_data(num_val, num_slots=num_slots, num_keys=num_keys, num_values=num_values, seed=200)
    
    log(f"Data: Train={num_train}, Val={num_val}, Slots={num_slots}, Keys={num_keys}, Values={num_values}")
    log(f"Chance Level: {100.0 / num_values:.2f}%\n")
    
    models_to_test = [
        # Quadratic O(N^2) Softmax Reference
        ("SoftmaxPhase_O(N2)_H1", SoftmaxPhaseAttention(num_keys, num_values, d_model, num_heads=1, d_v=d_v, num_classes=num_values)),
        ("SoftmaxPhase_O(N2)_H4", SoftmaxPhaseAttention(num_keys, num_values, d_model, num_heads=4, d_v=d_v, num_classes=num_values)),
        
        # Candidate 1: Linear Holographic Phase O(N) - Pure Wave Superposition
        ("LinearHolographic_O(N)_H1", LinearHolographicPhaseAttention(num_keys, num_values, d_model, num_heads=1, d_v=d_v, num_classes=num_values)),
        ("LinearHolographic_O(N)_H4", LinearHolographicPhaseAttention(num_keys, num_values, d_model, num_heads=4, d_v=d_v, num_classes=num_values)),
        
        # Candidate 2: Delta-Phase Linear Attention O(N) - Error-Correcting Recurrence
        ("DeltaPhaseLinear_O(N)_H1", DeltaPhaseLinearAttention(num_keys, num_values, d_model, num_heads=1, d_v=d_v, num_classes=num_values)),
        ("DeltaPhaseLinear_O(N)_H4", DeltaPhaseLinearAttention(num_keys, num_values, d_model, num_heads=4, d_v=d_v, num_classes=num_values)),
        
        # Baseline: Standard Real Linear Attention (ELU+1)
        ("RealLinear_ELU_O(N)_H1", StandardRealLinearAttention(num_keys, num_values, d_model, num_heads=1, d_k=8, d_v=d_v, num_classes=num_values)),
        ("RealLinear_ELU_O(N)_H4", StandardRealLinearAttention(num_keys, num_values, d_model, num_heads=4, d_k=8, d_v=d_v, num_classes=num_values))
    ]
    
    results = []
    log(f"{'Model':<28} | {'Complexity':<10} | {'Params':<8} | {'Val Acc (%)':<12} | {'Val Loss':<10} | {'Time (s)':<8}")
    log("-" * 88)
    
    for name, model in models_to_test:
        complexity = "O(N^2)" if "O(N2)" in name else "O(N)"
        r = train_and_eval(name, model, xk_tr, xv_tr, y_tr, xk_va, xv_va, y_va, epochs=15)
        log(f"{r['model_name']:<28} | {complexity:<10} | {r['params']:<8} | {r['val_acc']:<12.2f} | {r['val_loss']:<10.4f} | {r['time']:<8.2f}")
        r["complexity"] = complexity
        results.append(r)
        
    log("=" * 88)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "v377_linear_holographic_phase_attention.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"Results successfully saved to: {out_file}")

if __name__ == "__main__":
    main()
