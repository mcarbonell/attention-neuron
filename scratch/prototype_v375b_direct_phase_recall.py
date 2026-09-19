#!/usr/bin/env python3
"""
================================================================================
EXPERIMENT v375b: DIRECT ASSOCIATIVE RETRIEVAL (1D CONTENT ADDRESSABILITY)
================================================================================
Theoretical Core:
  In 1D Real Space (R^1), dot products q * k are monotonic. A single query q cannot
  selectively attend to an arbitrary intermediate key among K keys without attending
  even more strongly to extreme keys.
  In 1D Complex Space (C^1 and U(1)), the circular geometry allows continuous
  orthogonality (cos(theta_q - theta_k) = 1 for match, <= 0 for all others).

  Therefore:
  - PhaseOnlyAttention (U(1), H=1): Predicted to achieve near 100% selective recall.
  - ComplexScalarAttention (C^1, H=1): Predicted to achieve near 100% selective recall.
  - RealScalarAttention (R^1, H=1): Predicted to FAIL/COLLAPSE on intermediate keys.
  - StandardVectorAttention (R^d, H=1): Can solve it, but requires d=8 floats per head.
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

# --- DATASET: DIRECT CONTENT ADDRESSING ---
# Each token in the memory slots (positions 0..num_slots-1) contains:
# [Key_ID, Value_ID].
# Position num_slots contains the Query: [Key_Query, 0].
# Target: Predict Value_ID of the matching Key.
def generate_direct_associative_data(num_samples, num_slots=8, num_keys=8, num_values=8, seed=42):
    gen = torch.Generator().manual_seed(seed)
    
    # Inputs: (num_samples, num_slots + 1, 2)
    # [:, :, 0] is Key (1..num_keys)
    # [:, :, 1] is Value (1..num_values)
    # Target: (num_samples,) in 0..num_values-1
    
    X_k = torch.zeros(num_samples, num_slots + 1, dtype=torch.long)
    X_v = torch.zeros(num_samples, num_slots + 1, dtype=torch.long)
    Y = torch.zeros(num_samples, dtype=torch.long)
    
    for i in range(num_samples):
        # Pick distinct keys for memory slots
        keys = torch.randperm(num_keys, generator=gen)[:num_slots] + 1
        vals = torch.randint(1, num_values + 1, (num_slots,), generator=gen)
        
        X_k[i, :num_slots] = keys
        X_v[i, :num_slots] = vals
        
        # Query: pick one of the keys
        target_idx = torch.randint(0, num_slots, (1,), generator=gen).item()
        X_k[i, num_slots] = keys[target_idx]
        X_v[i, num_slots] = 0 # Value masked out in query slot
        
        Y[i] = vals[target_idx] - 1 # 0-indexed target class
        
    return X_k, X_v, Y

# --- MODEL ARCHITECTURES ---

class DirectPhaseOnlyAttention(nn.Module):
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
        
        # Phase angles in [-pi, pi)
        theta_q = (torch.tanh(self.w_q(h_k)) * math.pi).transpose(1, 2).unsqueeze(-1) # (B, H, L, 1)
        theta_k = (torch.tanh(self.w_k(h_k)) * math.pi).transpose(1, 2).unsqueeze(-2) # (B, H, 1, L)
        
        scores = torch.cos(theta_q - theta_k) * self.scale
        attn = F.softmax(scores, dim=-1) # (B, H, L, L)
        
        v = self.w_v(h_full).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        return self.out_proj(out[:, -1, :])

class DirectComplexScalarAttention(nn.Module):
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
        self.scale = nn.Parameter(torch.tensor(3.0))

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

class DirectRealScalarAttention(nn.Module):
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
        self.scale = nn.Parameter(torch.tensor(3.0))

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

class DirectStandardVectorAttention(nn.Module):
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
def train_and_eval(model_name, model, xk_tr, xv_tr, y_tr, xk_va, xv_va, y_va, epochs=15, batch_size=64, lr=0.01):
    log(f"--- Starting: {model_name} ---")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Trainable parameters: {n_params}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_train = xk_tr.shape[0]
    
    history = []
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
            
            if ep == 1 and b_idx < 5:
                b_acc = (preds == by).float().mean().item() * 100
                log(f"[Fast Feedback Ep1/B{b_idx+1}] Loss: {loss.item():.4f} | Acc: {b_acc:.1f}%")
                
        tr_loss = ep_loss / total
        tr_acc = (correct / total) * 100
        
        model.eval()
        with torch.no_grad():
            va_logits = model(xk_va, xv_va)
            va_loss = criterion(va_logits, y_va).item()
            va_preds = va_logits.argmax(dim=-1)
            va_acc = (va_preds == y_va).float().mean().item() * 100
            
        elapsed = time.time() - start_time
        speed = ep / max(elapsed, 1e-4)
        eta = (epochs - ep) / max(speed, 1e-4)
        
        if ep % 3 == 0 or ep == epochs or ep == 1:
            log(f"Ep {ep:02d}/{epochs:02d} | Train: {tr_loss:.4f}, {tr_acc:5.1f}% | Val: {va_loss:.4f}, {va_acc:5.1f}% | ETA: {eta:.1f}s")
            
        history.append({
            "epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc
        })
        
    tot_time = time.time() - start_time
    log(f"Finished {model_name}: Final Val Acc = {va_acc:.2f}%, Time = {tot_time:.2f}s\n")
    return {
        "model_name": model_name, "params": n_params,
        "final_val_acc": va_acc, "final_val_loss": va_loss,
        "wall_clock_time": tot_time, "history": history
    }

def main():
    log("=" * 80)
    log("EXPERIMENT v375b: DIRECT ASSOCIATIVE RETRIEVAL (1D CONTENT ADDRESSABILITY)")
    log("=" * 80)
    log(f"Platform: {platform.platform()} | Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    
    num_train = 3000
    num_val = 600
    num_slots = 6 # 6 distinct memory items per sequence
    num_keys = 8
    num_values = 8
    
    xk_tr, xv_tr, y_tr = generate_direct_associative_data(num_train, num_slots, num_keys, num_values, seed=100)
    xk_va, xv_va, y_va = generate_direct_associative_data(num_val, num_slots, num_keys, num_values, seed=200)
    
    log(f"Data: Train={num_train}, Val={num_val}, Slots={num_slots}, Keys={num_keys}, Values={num_values}")
    log(f"Chance Level: {100.0 / num_values:.2f}%\n")
    
    d_model = 16
    d_v = 8
    num_classes = num_values
    
    results = []
    
    # -------------------------------------------------------------------------
    # SUITE 1: SINGLE-HEAD (H=1) -> THE PURE TEST OF 1D GEOMETRY
    # -------------------------------------------------------------------------
    log(">>> SUITE 1: SINGLE HEAD (H=1) - PURE 1D BOTTLENECK <<<")
    
    # 1. Phase-Only Attention U(1)
    m1 = DirectPhaseOnlyAttention(num_keys, num_values, d_model, num_heads=1, d_v=d_v, num_classes=num_classes)
    r1 = train_and_eval("PhaseOnly_U1_H1", m1, xk_tr, xv_tr, y_tr, xk_va, xv_va, y_va, epochs=15)
    results.append(r1)
    
    # 2. Complex Scalar Attention C^1
    m2 = DirectComplexScalarAttention(num_keys, num_values, d_model, num_heads=1, d_v=d_v, num_classes=num_classes)
    r2 = train_and_eval("ComplexScalar_C1_H1", m2, xk_tr, xv_tr, y_tr, xk_va, xv_va, y_va, epochs=15)
    results.append(r2)
    
    # 3. Real Scalar Attention R^1
    m3 = DirectRealScalarAttention(num_keys, num_values, d_model, num_heads=1, d_v=d_v, num_classes=num_classes)
    r3 = train_and_eval("RealScalar_R1_H1", m3, xk_tr, xv_tr, y_tr, xk_va, xv_va, y_va, epochs=15)
    results.append(r3)
    
    # 4. Standard Vector Attention R^d (d_k=8)
    m4 = DirectStandardVectorAttention(num_keys, num_values, d_model, num_heads=1, d_k=8, d_v=d_v, num_classes=num_classes)
    r4 = train_and_eval("StandardVector_dk8_H1", m4, xk_tr, xv_tr, y_tr, xk_va, xv_va, y_va, epochs=15)
    results.append(r4)
    
    # -------------------------------------------------------------------------
    # SUITE 2: MULTI-HEAD (H=4) -> CAN MULTIPLE REAL HEADS COMPENSATE?
    # -------------------------------------------------------------------------
    log(">>> SUITE 2: MULTI-HEAD (H=4) - CAN MULTIPLE HEADS OVERCOME 1D? <<<")
    
    # 5. Phase-Only Attention U(1) H=4
    m5 = DirectPhaseOnlyAttention(num_keys, num_values, d_model, num_heads=4, d_v=d_v, num_classes=num_classes)
    r5 = train_and_eval("PhaseOnly_U1_H4", m5, xk_tr, xv_tr, y_tr, xk_va, xv_va, y_va, epochs=15)
    results.append(r5)
    
    # 6. Complex Scalar Attention C^1 H=4
    m6 = DirectComplexScalarAttention(num_keys, num_values, d_model, num_heads=4, d_v=d_v, num_classes=num_classes)
    r6 = train_and_eval("ComplexScalar_C1_H4", m6, xk_tr, xv_tr, y_tr, xk_va, xv_va, y_va, epochs=15)
    results.append(r6)
    
    # 7. Real Scalar Attention R^1 H=4
    m7 = DirectRealScalarAttention(num_keys, num_values, d_model, num_heads=4, d_v=d_v, num_classes=num_classes)
    r7 = train_and_eval("RealScalar_R1_H4", m7, xk_tr, xv_tr, y_tr, xk_va, xv_va, y_va, epochs=15)
    results.append(r7)
    
    # 8. Standard Vector Attention R^d (H=4, d_k=8)
    m8 = DirectStandardVectorAttention(num_keys, num_values, d_model, num_heads=4, d_k=8, d_v=d_v, num_classes=num_classes)
    r8 = train_and_eval("StandardVector_dk8_H4", m8, xk_tr, xv_tr, y_tr, xk_va, xv_va, y_va, epochs=15)
    results.append(r8)
    
    log("=" * 80)
    log(f"{'Model':<28} | {'Params':<8} | {'Val Acc (%)':<12} | {'Val Loss':<10} | {'Time (s)':<8}")
    log("-" * 80)
    for res in results:
        log(f"{res['model_name']:<28} | {res['params']:<8} | {res['final_val_acc']:<12.2f} | {res['final_val_loss']:<10.4f} | {res['wall_clock_time']:<8.2f}")
    log("=" * 80)
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "v375b_direct_associative_attention.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"Raw results saved to: {out_file}")

if __name__ == "__main__":
    main()
