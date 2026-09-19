#!/usr/bin/env python3
"""
================================================================================
EXPERIMENT v375: MINI-TRANSFORMER SCALAR & COMPLEX ATTENTION
================================================================================
Hypothesis:
  1D Scalar Attention in the complex domain (Phase-Only U(1) and Complex C^1)
  restores continuous orthogonality and selective associative recall, outperforming
  1D Real Scalar Attention (which collapses to a monotonic soft-ranker) while using
  a fraction of the parameters of standard multi-dimensional vector attention.

Contract Compliance (GEMINI.md):
  - Execution timestamp formatted: [+HH:MM:SS.ss] with flush=True
  - Execution metadata & architecture parameter breakdown in header
  - Fast feedback within the first 5 batches of Epoch 1
  - Rule of Thumb: Candidates evaluated first (PhaseOnly, Complex, Real, then Vector)
  - Fully vectorized forward passes (zero Python tensor loops)
  - Separate train & validation evaluations
  - Output results persisted to results/raw/v375_scalar_complex_attention.json
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

# --- TIMESTAMP HELPER ---
T0 = time.time()
def log(msg):
    elapsed = time.time() - T0
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60
    print(f"[+{h:02d}:{m:02d}:{s:05.2f}] {msg}", flush=True)

# --- REPRODUCIBILITY & DEVICE ---
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device('cpu')

# --- DATASET: ASSOCIATIVE SELECTIVE RECALL (MQAR-LITE 1D) ---
# Sequence length L=16. 4 Key-Value pairs placed at early positions.
# Distractor tokens in between.
# Position L-1 contains a Query Key; target is the paired Value class.
def generate_associative_data(num_samples, seq_len=16, num_pairs=4, num_keys=8, num_values=8, seed=42):
    gen = torch.Generator().manual_seed(seed)
    
    # Vocab layout:
    # 0: PAD / DISTRACTOR
    # 1 .. num_keys: KEY tokens
    # num_keys+1 .. num_keys+num_values: VALUE tokens
    val_offset = num_keys + 1
    vocab_size = num_keys + num_values + 2
    
    X = torch.zeros(num_samples, seq_len, dtype=torch.long)
    Y = torch.zeros(num_samples, dtype=torch.long)
    
    for i in range(num_samples):
        # Pick distinct keys
        chosen_keys = torch.randperm(num_keys, generator=gen)[:num_pairs] + 1
        chosen_vals = torch.randint(0, num_values, (num_pairs,), generator=gen)
        
        # Place pairs in first seq_len - 2 positions (2 slots per pair)
        # slots: 0, 1, 2, 3, 4, 5, 6, 7
        for p in range(num_pairs):
            k_pos = 2 * p
            v_pos = 2 * p + 1
            X[i, k_pos] = chosen_keys[p]
            X[i, v_pos] = chosen_vals[p] + val_offset
            
        # Distractors in remaining slots except last
        for d in range(2 * num_pairs, seq_len - 1):
            X[i, d] = 0
            
        # Query at last slot
        query_idx = torch.randint(0, num_pairs, (1,), generator=gen).item()
        X[i, seq_len - 1] = chosen_keys[query_idx]
        Y[i] = chosen_vals[query_idx]
        
    return X, Y, vocab_size

# --- MODEL DEFINITIONS ---

# 1. CANDIDATE A: Phase-Only Unitary Attention (U(1), d_k=1)
# Zero multiplications in affinity: S_ij = cos(theta_q - theta_k) * scale
class PhaseOnlyAttentionModel(nn.Module):
    def __init__(self, vocab_size, d_in, num_heads, d_v, num_classes):
        super().__init__()
        self.num_heads = num_heads
        self.d_v = d_v
        self.emb = nn.Embedding(vocab_size, d_in)
        self.w_q = nn.Linear(d_in, num_heads)
        self.w_k = nn.Linear(d_in, num_heads)
        self.w_v = nn.Linear(d_in, num_heads * d_v)
        self.out_proj = nn.Linear(num_heads * d_v, num_classes)
        self.scale = nn.Parameter(torch.tensor(3.0))

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x) # (B, L, d_in)
        theta_q = (torch.tanh(self.w_q(h)) * math.pi).transpose(1, 2).unsqueeze(-1) # (B, H, L, 1)
        theta_k = (torch.tanh(self.w_k(h)) * math.pi).transpose(1, 2).unsqueeze(-2) # (B, H, 1, L)
        delta_theta = theta_q - theta_k # (B, H, L, L)
        scores = torch.cos(delta_theta) * self.scale
        attn = F.softmax(scores, dim=-1)
        v = self.w_v(h).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        logits = self.out_proj(out[:, -1, :]) # Query output at last position
        return logits

# 2. CANDIDATE B: Complex Scalar Attention (C^1, d_k=1)
# Inner product: Re(q k*) = q_re * k_re + q_im * k_im = r_q r_k cos(delta_theta)
class ComplexScalarAttentionModel(nn.Module):
    def __init__(self, vocab_size, d_in, num_heads, d_v, num_classes):
        super().__init__()
        self.num_heads = num_heads
        self.d_v = d_v
        self.emb = nn.Embedding(vocab_size, d_in)
        self.w_q_re = nn.Linear(d_in, num_heads)
        self.w_q_im = nn.Linear(d_in, num_heads)
        self.w_k_re = nn.Linear(d_in, num_heads)
        self.w_k_im = nn.Linear(d_in, num_heads)
        self.w_v = nn.Linear(d_in, num_heads * d_v)
        self.out_proj = nn.Linear(num_heads * d_v, num_classes)
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x)
        q_re = self.w_q_re(h).transpose(1, 2).unsqueeze(-1) # (B, H, L, 1)
        q_im = self.w_q_im(h).transpose(1, 2).unsqueeze(-1)
        k_re = self.w_k_re(h).transpose(1, 2).unsqueeze(-2) # (B, H, 1, L)
        k_im = self.w_k_im(h).transpose(1, 2).unsqueeze(-2)
        scores = (q_re * k_re + q_im * k_im) * self.scale
        attn = F.softmax(scores, dim=-1)
        v = self.w_v(h).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        logits = self.out_proj(out[:, -1, :])
        return logits

# 3. CANDIDATE C: Real Scalar Attention (R^1, d_k=1)
# Inner product: q * k (1D scalar multiplication, soft-ranker)
class RealScalarAttentionModel(nn.Module):
    def __init__(self, vocab_size, d_in, num_heads, d_v, num_classes):
        super().__init__()
        self.num_heads = num_heads
        self.d_v = d_v
        self.emb = nn.Embedding(vocab_size, d_in)
        self.w_q = nn.Linear(d_in, num_heads)
        self.w_k = nn.Linear(d_in, num_heads)
        self.w_v = nn.Linear(d_in, num_heads * d_v)
        self.out_proj = nn.Linear(num_heads * d_v, num_classes)
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x)
        q = self.w_q(h).transpose(1, 2).unsqueeze(-1) # (B, H, L, 1)
        k = self.w_k(h).transpose(1, 2).unsqueeze(-2) # (B, H, 1, L)
        scores = (q * k) * self.scale
        attn = F.softmax(scores, dim=-1)
        v = self.w_v(h).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        logits = self.out_proj(out[:, -1, :])
        return logits

# 4. BASELINE: Standard Multi-Head Vector Attention (R^d_k)
class StandardVectorAttentionModel(nn.Module):
    def __init__(self, vocab_size, d_in, num_heads, d_k, d_v, num_classes):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.emb = nn.Embedding(vocab_size, d_in)
        self.w_q = nn.Linear(d_in, num_heads * d_k)
        self.w_k = nn.Linear(d_in, num_heads * d_k)
        self.w_v = nn.Linear(d_in, num_heads * d_v)
        self.out_proj = nn.Linear(num_heads * d_v, num_classes)

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x)
        q = self.w_q(h).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(h).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(h).view(B, L, self.num_heads, self.d_v).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_v)
        logits = self.out_proj(out[:, -1, :])
        return logits

# --- TRAINING HARNESS ---
def train_and_eval(model_name, model, x_train, y_train, x_val, y_val, epochs=15, batch_size=64, lr=0.01):
    log(f"--- Starting Training: {model_name} ---")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Trainable parameters: {n_params}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    num_train = x_train.shape[0]
    indices = torch.arange(num_train)
    
    history = []
    start_time = time.time()
    
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_train)
        ep_loss = 0.0
        correct = 0
        total = 0
        
        num_batches = (num_train + batch_size - 1) // batch_size
        for b_idx in range(num_batches):
            b_ids = perm[b_idx * batch_size : min((b_idx + 1) * batch_size, num_train)]
            bx, by = x_train[b_ids], y_train[b_ids]
            
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
            ep_loss += loss.item() * len(by)
            preds = logits.argmax(dim=-1)
            correct += (preds == by).sum().item()
            total += len(by)
            
            # Fast Feedback: print first 5 batches of Epoch 1
            if ep == 1 and b_idx < 5:
                batch_acc = (preds == by).float().mean().item() * 100
                log(f"[Fast Feedback Ep1/B{b_idx+1}] Loss: {loss.item():.4f} | Batch Acc: {batch_acc:.1f}%")
                
        train_loss = ep_loss / total
        train_acc = (correct / total) * 100
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = criterion(val_logits, y_val).item()
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == y_val).float().mean().item() * 100
            
        elapsed_ep = time.time() - start_time
        speed = ep / max(elapsed_ep, 1e-4)
        eta = (epochs - ep) / max(speed, 1e-4)
        
        if ep % 3 == 0 or ep == epochs or ep == 1:
            log(f"Ep {ep:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:5.1f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:5.1f}% | ETA: {eta:.1f}s")
            
        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
    total_time = time.time() - start_time
    log(f"Finished {model_name}: Final Val Acc = {val_acc:.2f}%, WallClock = {total_time:.2f}s\n")
    
    return {
        "model_name": model_name,
        "params": n_params,
        "final_val_acc": val_acc,
        "final_val_loss": val_loss,
        "wall_clock_time": total_time,
        "history": history
    }

# --- MAIN BENCHMARK RUNNER ---
def main():
    log("=" * 70)
    log("EXPERIMENT v375: MINI-TRANSFORMER SCALAR & COMPLEX ATTENTION")
    log("=" * 70)
    log(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    log(f"Platform: {platform.platform()} | Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    log(f"Device: {device}")
    
    # 1. Generate synthetic dataset
    num_train = 2400
    num_val = 600
    seq_len = 16
    num_pairs = 4
    num_keys = 8
    num_values = 8
    
    x_train, y_train, vocab_size = generate_associative_data(
        num_train, seq_len=seq_len, num_pairs=num_pairs, num_keys=num_keys, num_values=num_values, seed=100
    )
    x_val, y_val, _ = generate_associative_data(
        num_val, seq_len=seq_len, num_pairs=num_pairs, num_keys=num_keys, num_values=num_values, seed=200
    )
    
    log(f"Dataset generated: Train={num_train}, Val={num_val}, SeqLen={seq_len}, Pairs={num_pairs}, Vocab={vocab_size}")
    log(f"Theoretical Chance Accuracy: {100.0 / num_values:.2f}%\n")
    
    d_in = 16
    d_v = 8
    num_classes = num_values
    
    results = []
    
    # =========================================================================
    # SUITE 1: SINGLE-HEAD ATTENTION (H=1, Pure 1D Dimensionality Test)
    # =========================================================================
    log(">>> SUITE 1: SINGLE-HEAD ATTENTION (H=1, PURE 1D BOTTLENECK) <<<")
    
    # Candidate 1: Phase-Only U(1)
    m1 = PhaseOnlyAttentionModel(vocab_size, d_in, num_heads=1, d_v=d_v, num_classes=num_classes).to(device)
    r1 = train_and_eval("PhaseOnly_U1_H1", m1, x_train, y_train, x_val, y_val, epochs=15)
    results.append(r1)
    
    # Candidate 2: Complex Scalar C^1
    m2 = ComplexScalarAttentionModel(vocab_size, d_in, num_heads=1, d_v=d_v, num_classes=num_classes).to(device)
    r2 = train_and_eval("ComplexScalar_C1_H1", m2, x_train, y_train, x_val, y_val, epochs=15)
    results.append(r2)
    
    # Candidate 3: Real Scalar R^1
    m3 = RealScalarAttentionModel(vocab_size, d_in, num_heads=1, d_v=d_v, num_classes=num_classes).to(device)
    r3 = train_and_eval("RealScalar_R1_H1", m3, x_train, y_train, x_val, y_val, epochs=15)
    results.append(r3)
    
    # Baseline: Vector Attention (d_k=8)
    m4 = StandardVectorAttentionModel(vocab_size, d_in, num_heads=1, d_k=8, d_v=d_v, num_classes=num_classes).to(device)
    r4 = train_and_eval("StandardVector_dk8_H1", m4, x_train, y_train, x_val, y_val, epochs=15)
    results.append(r4)
    
    # =========================================================================
    # SUITE 2: MULTI-HEAD ATTENTION (H=4, Multi-Channel Capacity Test)
    # =========================================================================
    log(">>> SUITE 2: MULTI-HEAD ATTENTION (H=4, MULTI-CHANNEL CAPACITY) <<<")
    
    # Candidate 1: Phase-Only U(1) Multi-Head
    m5 = PhaseOnlyAttentionModel(vocab_size, d_in, num_heads=4, d_v=d_v, num_classes=num_classes).to(device)
    r5 = train_and_eval("PhaseOnly_U1_H4", m5, x_train, y_train, x_val, y_val, epochs=15)
    results.append(r5)
    
    # Candidate 2: Complex Scalar C^1 Multi-Head
    m6 = ComplexScalarAttentionModel(vocab_size, d_in, num_heads=4, d_v=d_v, num_classes=num_classes).to(device)
    r6 = train_and_eval("ComplexScalar_C1_H4", m6, x_train, y_train, x_val, y_val, epochs=15)
    results.append(r6)
    
    # Candidate 3: Real Scalar R^1 Multi-Head
    m7 = RealScalarAttentionModel(vocab_size, d_in, num_heads=4, d_v=d_v, num_classes=num_classes).to(device)
    r7 = train_and_eval("RealScalar_R1_H4", m7, x_train, y_train, x_val, y_val, epochs=15)
    results.append(r7)
    
    # Baseline: Vector Attention Multi-Head (H=4, d_k=8)
    m8 = StandardVectorAttentionModel(vocab_size, d_in, num_heads=4, d_k=8, d_v=d_v, num_classes=num_classes).to(device)
    r8 = train_and_eval("StandardVector_dk8_H4", m8, x_train, y_train, x_val, y_val, epochs=15)
    results.append(r8)
    
    # =========================================================================
    # SUMMARY TABLE & EXPORT
    # =========================================================================
    log("=" * 80)
    log(f"{'Model':<26} | {'Params':<8} | {'Val Acc (%)':<12} | {'Val Loss':<10} | {'Time (s)':<8}")
    log("-" * 80)
    for res in results:
        log(f"{res['model_name']:<26} | {res['params']:<8} | {res['final_val_acc']:<12.2f} | {res['final_val_loss']:<10.4f} | {res['wall_clock_time']:<8.2f}")
    log("=" * 80)
    
    # Persist JSON
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "v375_scalar_complex_attention.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"Raw results successfully exported to: {out_file}")

if __name__ == "__main__":
    main()
