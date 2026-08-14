"""
run_head_to_head_isomemory_reproduced.py
==========================================
Dynamic On-The-Fly MQAR Harness (v305/v307 Standard).
Generates fresh MQAR key-value pairs dynamically on-the-fly per batch to prevent static memorization,
forcing models to learn the in-context associative Delta Rule algorithm.
Evaluates Real Gated DeltaNet (d_k=32, d_k=45, d_k=64) vs Complex DeltaPhase (d_k=32, d_k=45).
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_dynamic_mqar_batch(batch_size=64, num_pairs=16, seq_len=64, vocab_size=64, device='cpu'):
    """
    Generates fresh, dynamic MQAR key-value retrieval sequences per batch.
    """
    inputs = torch.randint(4, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    
    for b in range(batch_size):
        keys = torch.randperm(vocab_size - 4)[:num_pairs] + 4
        vals = torch.randperm(vocab_size - 4)[:num_pairs] + 4
        
        for i in range(num_pairs):
            pos_k = 2 * i
            pos_v = 2 * i + 1
            if pos_v < seq_len - 2:
                inputs[b, pos_k] = keys[i]
                inputs[b, pos_v] = vals[i]
                
        q_idx = torch.randint(0, num_pairs, (1,)).item()
        query_key = keys[q_idx]
        target_val = vals[q_idx]
        
        inputs[b, -2] = query_key
        targets[b, -1] = target_val
        
    return inputs, targets

class RealGatedDeltaNetLM(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, n_heads=4, d_k=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.inv_dk = 1.0 / float(d_k)
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_beta = nn.Linear(d_model, n_heads, bias=False)
        self.out_proj = nn.Linear(n_heads * d_k, d_model, bias=False)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, L = idx.shape
        x = self.embed(idx)
        
        k = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        beta = torch.sigmoid(self.w_beta(x)).transpose(1, 2)
        
        M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, device=idx.device)
        out_list = []
        for t in range(L):
            kt, qt, vt, bt = k[:, :, t], q[:, :, t], v[:, :, t], beta[:, :, t]
            v_old = torch.matmul(M, kt.unsqueeze(-1)).squeeze(-1) * self.inv_dk
            err = vt - v_old
            M = M + bt.unsqueeze(-1).unsqueeze(-1) * torch.matmul(err.unsqueeze(-1), kt.unsqueeze(-2))
            out_t = torch.matmul(M, qt.unsqueeze(-1)).squeeze(-1) * self.inv_dk
            out_list.append(out_t)
            
        out_concat = torch.cat(out_list, dim=-1).view(B, self.n_heads, L, self.d_k).transpose(1, 2).reshape(B, L, self.n_heads * self.d_k)
        retrieved = self.out_proj(out_concat)
        logits = self.head(retrieved)
        return logits

class ComplexDeltaPhaseLM(nn.Module):
    def __init__(self, vocab_size=64, d_model=128, n_heads=4, d_k=32):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.inv_dk = 1.0 / float(d_k)
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_beta = nn.Linear(d_model, n_heads, bias=False)
        self.out_proj = nn.Linear(n_heads * d_k, d_model, bias=False)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, L = idx.shape
        x = self.embed(idx)
        
        theta_k = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        theta_q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        beta = 2.0 * torch.sigmoid(self.w_beta(x)).transpose(1, 2)
        
        K = torch.complex(torch.cos(theta_k), torch.sin(theta_k))
        Q = torch.complex(torch.cos(theta_q), torch.sin(theta_q))
        
        M = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=idx.device)
        out_list = []
        for t in range(L):
            kt, qt, vt, bt = K[:, :, t], Q[:, :, t], v[:, :, t], beta[:, :, t]
            v_old = torch.matmul(M, torch.conj(kt).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            err = vt - v_old
            M = M + bt.unsqueeze(-1).unsqueeze(-1) * torch.matmul(err.to(torch.complex64).unsqueeze(-1), kt.unsqueeze(-2))
            out_t = torch.matmul(M, torch.conj(qt).unsqueeze(-1)).squeeze(-1).real * self.inv_dk
            out_list.append(out_t)
            
        out_concat = torch.cat(out_list, dim=-1).view(B, self.n_heads, L, self.d_k).transpose(1, 2).reshape(B, L, self.n_heads * self.d_k)
        retrieved = self.out_proj(out_concat)
        logits = self.head(retrieved)
        return logits

def train_and_eval_dynamic(model_cls, d_k, name, seeds=[42, 43, 44], total_steps=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accs = []
    
    print(f"\n--- Training {name} (d_k={d_k}, Dynamic On-The-Fly Generation) ---", flush=True)
    for seed in seeds:
        torch.manual_seed(seed)
        model = model_cls(vocab_size=64, d_model=128, n_heads=4, d_k=d_k).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        
        t0 = time.time()
        for step in range(1, total_steps + 1):
            model.train()
            x_b, y_b = generate_dynamic_mqar_batch(batch_size=64, num_pairs=16, seq_len=64, vocab_size=64, device=device)
            
            logits = model(x_b)
            loss = F.cross_entropy(logits[:, -1, :], y_b[:, -1])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0 or step == 1:
                preds = logits[:, -1, :].argmax(dim=-1)
                train_acc = (preds == y_b[:, -1]).float().mean().item() * 100.0
                dt_step = time.time() - t0
                print(f"  [Seed {seed} Step {step:>3}/{total_steps}] Loss: {loss.item():.4f} | Dynamic Acc: {train_acc:.2f}% | Time: {dt_step:.2f}s", flush=True)
                
        model.eval()
        with torch.no_grad():
            eval_in, eval_tgt = generate_dynamic_mqar_batch(batch_size=500, num_pairs=16, seq_len=64, vocab_size=64, device=device)
            eval_logits = model(eval_in)
            eval_preds = eval_logits[:, -1, :].argmax(dim=-1)
            val_acc = (eval_preds == eval_tgt[:, -1]).float().mean().item() * 100.0
            accs.append(val_acc)
            print(f"  [Seed {seed} Final Dynamic Val Accuracy]: {val_acc:.2f}%", flush=True)
            
    mean_acc = sum(accs) / len(accs)
    se_acc = (sum((a - mean_acc)**2 for a in accs) / max(1, len(accs) - 1))**0.5 / (len(accs)**0.5)
    return mean_acc, se_acc

def main():
    print("=" * 95, flush=True)
    print("DYNAMIC ON-THE-FLY MQAR BENCHMARK (v305/v307 STANDARD)", flush=True)
    print("=" * 95, flush=True)
    print(f"CHANCE LEVEL BASELINE: 1/64 = 1.56% Accuracy (4.1588 nats Loss)", flush=True)
    print("=" * 95, flush=True)
    
    # 1. POSITIVE CONTROLS
    acc_real_32, se_real_32 = train_and_eval_dynamic(RealGatedDeltaNetLM, d_k=32, name="Positive Control: Real Gated DeltaNet (d_k=32, 1024 Floats)")
    acc_complex_32, se_complex_32 = train_and_eval_dynamic(ComplexDeltaPhaseLM, d_k=32, name="Positive Control: Complex DeltaPhase (d_k=32, 2048 Floats)")
    
    # 2. ISO-MEMORY CONTROL (Real d_k=45 [2025 Floats])
    acc_real_45, se_real_45 = train_and_eval_dynamic(RealGatedDeltaNetLM, d_k=45, name="Iso-Memory: Real Gated DeltaNet (d_k=45, 2025 Floats)")
    
    print("\n" + "=" * 95, flush=True)
    print("SUMMARY OF DYNAMIC MQAR BENCHMARK RESULTS (Mean ± Standard Error)", flush=True)
    print("=" * 95, flush=True)
    print(f"Chance Level Baseline: 1.56%", flush=True)
    print(f"1. Real Gated DeltaNet  (d_k=32, 1024 Floats): {acc_real_32:.2f}% ± {se_real_32:.2f}%", flush=True)
    print(f"2. Complex DeltaPhase   (d_k=32, 2048 Floats): {acc_complex_32:.2f}% ± {se_complex_32:.2f}% (Gap: {acc_complex_32 - acc_real_32:+.2f}%)", flush=True)
    print(f"3. Real Gated DeltaNet  (d_k=45, 2025 Floats): {acc_real_45:.2f}% ± {se_real_45:.2f}% (Iso vs Complex d_k=32 Gap: {acc_complex_32 - acc_real_45:+.2f}%)", flush=True)
    print("=" * 95, flush=True)

if __name__ == "__main__":
    main()
