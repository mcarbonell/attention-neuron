"""
run_head_to_head_isomemory_certified_v305.py
==============================================
CERTIFIED V305 MQAR BENCHMARK: ISO-MEMORY CONTROL AUDIT (2000 STEPS, LR=0.0004).

Metadata:
---------
- Harness: Certified v305 On-The-Fly Dynamic MQAR Generator (VOCAB_SIZE=514, ignore_index=-100)
- Architecture: 4-Layer Residual Transformer with ShortCausalConv1D, LayerNorm, and FFN (expand=2)
- Numerics: Numerically Stable solve_triangular, Gradient Norm Clipping (max_norm=1.0), FP32 Safe
- Embeddings: Learned Absolute Positional Embedding + Token Embedding (d_model=128, n_heads=4)
- Optimization: AdamW (lr=4e-4, weight_decay=1e-4, steps=2000, batch_size=32)
- Evaluates:
    1. Positive Control: Real Gated DeltaNet (d_k=32, 1024 Floats State/Head)
    2. Positive Control: Complex DeltaPhase (d_k=32, 2048 Floats State/Head)
    3. Iso-Memory Control: Real Gated DeltaNet (d_k=45, 2025 Floats State/Head)
- Chance Level Baseline: 1/512 = 0.195% Accuracy (6.2383 nats Loss)
"""

import math, time, os, json, sys, platform, torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()

def ts():
    elapsed = time.time() - T0
    h = int(elapsed // 3600)
    m = int(elapsed % 3600 // 60)
    s = elapsed % 60
    return f"[{h:02d}:{m:02d}:{s:05.2f}]"

def print_log_header(device_str):
    print("===============================================================================================", flush=True)
    print("                               EXECUTION METADATA LOG HEADER                                   ", flush=True)
    print("===============================================================================================", flush=True)
    print(f" Timestamp:              {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", flush=True)
    print(f" Platform / Python:      {platform.platform()} | Python {sys.version.split()[0]}", flush=True)
    print(f" PyTorch Version:        {torch.__version__}", flush=True)
    print(f" Execution Device:       {device_str} (CPU Threads: {torch.get_num_threads()})", flush=True)
    print(f" Model Source File:      scratch/run_head_to_head_isomemory_certified_v305.py", flush=True)
    print(f" Dataset / Harness:      Certified v305 On-The-Fly MQAR Dynamic Batch Generator (VOCAB_SIZE=514)", flush=True)
    print(f" Loss Function:          CrossEntropyLoss(ignore_index=-100)", flush=True)
    print(f" Numerical Stabilizers:  solve_triangular, grad_norm_clipping(1.0), AdamW(lr=4e-4, wd=1e-4)", flush=True)
    print(f" Chance Level Baseline:  1/512 = 0.195% Accuracy (Theoretical Loss = 6.2383 nats)", flush=True)
    print(" Hyperparameters:", flush=True)
    print("   - d_model: 128 | n_heads: 4 | n_layers: 4 (Residual Transformer)", flush=True)
    print("   - Conv1D Kernel: 4 | FFN Expansion: 2x (SiLU)", flush=True)
    print("   - Optimizer: AdamW | Learning Rate: 4.00e-04 | Weight Decay: 1.00e-04", flush=True)
    print("   - Sequence Specs: seq_len = 128 | n_pairs = 29 | Batch Size = 32 | Steps = 1000", flush=True)
    print(" State Memory Configurations Evaluated:", flush=True)
    print("   1. Real Gated DeltaNet  (d_k=32): 1024 Real Floats State RAM / Head", flush=True)
    print("   2. Complex DeltaPhase   (d_k=32): 2048 Real Floats State RAM / Head", flush=True)
    print("   3. Real Gated DeltaNet  (d_k=45): 2025 Real Floats State RAM / Head (Iso-Memory Control)", flush=True)
    print("===============================================================================================\n", flush=True)

PAD_ID = 0
TOKEN_OFFSET = 1
NUM_TOKENS = 512
QUERY_MARKER = TOKEN_OFFSET + NUM_TOKENS
VOCAB_SIZE = QUERY_MARKER + 1

# ── 1. Certified v305 Dynamic On-The-Fly MQAR Batch Generator ─────────────

def generate_mqar_batch(batch_size=32, n_pairs=29, seq_len=128, num_tokens=512, device='cpu'):
    tokens_needed = 2 * n_pairs
    rand_t = torch.rand(batch_size, num_tokens, device=device)
    sampled = torch.argsort(rand_t, dim=-1)[:, :tokens_needed] + TOKEN_OFFSET
    keys = sampled[:, :n_pairs]
    vals = sampled[:, n_pairs:]
    
    x = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    
    kv = torch.stack([keys, vals], dim=2).view(batch_size, 2 * n_pairs)
    x[:, :2 * n_pairs] = kv
    
    q_perm = torch.argsort(torch.rand(batch_size, n_pairs, device=device), dim=-1)
    query_keys = torch.gather(keys, 1, q_perm)
    query_vals = torch.gather(vals, 1, q_perm)
    
    gap = 2
    pos_q = (2 * n_pairs + gap + 2 * torch.arange(n_pairs, device=device)).unsqueeze(0).expand(batch_size, -1)
    x.scatter_(1, pos_q, QUERY_MARKER)
    x.scatter_(1, pos_q + 1, query_keys)
    y.scatter_(1, pos_q + 1, query_vals)
    return x, y

# ── 2. Building Blocks (Certified v305 with Numerically Stable Solvers) ───

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, max_len=1024, d_model=128):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        return self.pe(pos)

class ShortCausalConv1D(nn.Module):
    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size-1, groups=d_model)
        self.act = nn.SiLU()
    def forward(self, x):
        B, L, D = x.shape
        return x + self.act(self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2))

class FFN(nn.Module):
    def __init__(self, d_model, expand=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model * expand), nn.SiLU(), nn.Linear(d_model * expand, d_model))
    def forward(self, x): return self.net(x)

class ChunkwiseComplexDeltaPhaseBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_k=32, chunk_size=64):
        super().__init__()
        self.d_model, self.n_heads, self.d_k, self.chunk_size = d_model, n_heads, d_k, chunk_size
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.theta_k_proj = nn.Linear(d_model, n_heads * d_k)
        self.theta_q_proj = nn.Linear(d_model, n_heads * d_k)
        self.val_proj = nn.Linear(d_model, n_heads * d_k)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(n_heads * d_k, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x; conv_x = self.causal_conv(self.norm1(x))
        B, L, D = conv_x.shape; C = self.chunk_size; inv_dk = 1.0 / float(self.d_k)
        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            conv_x = F.pad(conv_x, (0, 0, 0, pad_len)); L_padded = L + pad_len
        else: L_padded = L
        theta_k = self.theta_k_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        theta_q = self.theta_q_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        v = self.val_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(conv_x)).transpose(1, 2)
        K = torch.polar(torch.ones_like(theta_k), theta_k)
        Q = torch.polar(torch.ones_like(theta_q), theta_q)
        num_chunks = L_padded // C
        Q_c = Q.view(B, self.n_heads, num_chunks, C, self.d_k)
        K_c = K.view(B, self.n_heads, num_chunks, C, self.d_k)
        V_c = v.view(B, self.n_heads, num_chunks, C, self.d_k)
        beta_c = beta.view(B, self.n_heads, num_chunks, C)
        Gram_real = torch.matmul(K_c, torch.conj(K_c).transpose(-1, -2)).real * inv_dk
        L_mat = torch.triu(Gram_real * beta_c.unsqueeze(-1), diagonal=1)
        I_mat = torch.eye(C, device=x.device).view(1, 1, 1, C, C)
        T_mat = torch.linalg.solve_triangular(I_mat + L_mat.transpose(-1, -2), I_mat, upper=False)
        M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=x.device)
        out_chunks = []
        for c in range(num_chunks):
            qc, kc, vc, bc, tc = Q_c[:,:,c], K_c[:,:,c], V_c[:,:,c], beta_c[:,:,c], T_mat[:,:,c]
            v_old = torch.matmul(M_state, torch.conj(kc).transpose(-1,-2)).real.transpose(-1,-2) * inv_dk
            E_c = torch.matmul(tc, vc - v_old)
            U_c = bc.unsqueeze(-1) * E_c
            o_inter = torch.matmul(M_state, torch.conj(qc).transpose(-1,-2)).real.transpose(-1,-2) * inv_dk
            A_intra = torch.tril(torch.matmul(qc, torch.conj(kc).transpose(-1,-2)).real) * inv_dk
            out_chunks.append(torch.matmul(A_intra, U_c) + o_inter)
            M_state = M_state + torch.matmul(U_c.to(torch.complex64).transpose(-1,-2), kc)
        retrieved = torch.cat(out_chunks, dim=2)[:,:,:L].transpose(1,2).reshape(B, L, self.n_heads*self.d_k)
        out = res + self.out_proj(retrieved)
        return out + self.ffn(self.norm2(out))

class RealGatedDeltaNetBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_k=32, chunk_size=64):
        super().__init__()
        self.d_model, self.n_heads, self.d_k, self.chunk_size = d_model, n_heads, d_k, chunk_size
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.k_proj = nn.Linear(d_model, n_heads * d_k)
        self.q_proj = nn.Linear(d_model, n_heads * d_k)
        self.val_proj = nn.Linear(d_model, n_heads * d_k)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(n_heads * d_k, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x; conv_x = self.causal_conv(self.norm1(x))
        B, L, D = conv_x.shape; C = self.chunk_size; inv_dk = 1.0 / float(self.d_k)
        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            conv_x = F.pad(conv_x, (0, 0, 0, pad_len)); L_padded = L + pad_len
        else: L_padded = L
        k = self.k_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        q = self.q_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        v = self.val_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(conv_x)).transpose(1, 2)
        num_chunks = L_padded // C
        Q_c = q.view(B, self.n_heads, num_chunks, C, self.d_k)
        K_c = k.view(B, self.n_heads, num_chunks, C, self.d_k)
        V_c = v.view(B, self.n_heads, num_chunks, C, self.d_k)
        beta_c = beta.view(B, self.n_heads, num_chunks, C)
        Gram_real = torch.matmul(K_c, K_c.transpose(-1, -2)) * inv_dk
        L_mat = torch.triu(Gram_real * beta_c.unsqueeze(-1), diagonal=1)
        I_mat = torch.eye(C, device=x.device).view(1, 1, 1, C, C)
        T_mat = torch.linalg.solve_triangular(I_mat + L_mat.transpose(-1, -2), I_mat, upper=False)
        M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, device=x.device)
        out_chunks = []
        for c in range(num_chunks):
            qc, kc, vc, bc, tc = Q_c[:,:,c], K_c[:,:,c], V_c[:,:,c], beta_c[:,:,c], T_mat[:,:,c]
            v_old = torch.matmul(M_state, kc.transpose(-1,-2)).transpose(-1,-2) * inv_dk
            E_c = torch.matmul(tc, vc - v_old)
            U_c = bc.unsqueeze(-1) * E_c
            o_inter = torch.matmul(M_state, qc.transpose(-1,-2)).transpose(-1,-2) * inv_dk
            A_intra = torch.tril(torch.matmul(qc, kc.transpose(-1,-2))) * inv_dk
            out_chunks.append(torch.matmul(A_intra, U_c) + o_inter)
            M_state = M_state + torch.matmul(U_c.transpose(-1,-2), kc)
        retrieved = torch.cat(out_chunks, dim=2)[:,:,:L].transpose(1,2).reshape(B, L, self.n_heads*self.d_k)
        out = res + self.out_proj(retrieved)
        return out + self.ffn(self.norm2(out))

class FullModelLM(nn.Module):
    def __init__(self, block_cls, vocab_size=VOCAB_SIZE, d_model=128, n_layers=4, n_heads=4, d_k=32):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = AbsolutePositionalEmbedding(max_len=1024, d_model=d_model)
        self.layers = nn.ModuleList([block_cls(d_model=d_model, n_heads=n_heads, d_k=d_k) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.tok_embed(x) + self.pos_embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h)

def evaluate_model_on_certified_harness(block_cls, d_k, name, steps=2000, n_pairs=29, seq_len=128, seeds=[42, 43, 44]):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"\n{ts()} --- Certified v305 Training {name} (d_k={d_k}, d_model=128, n_layers=4, steps={steps}, lr=4e-4) ---", flush=True)
    accs = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        model = FullModelLM(block_cls, vocab_size=VOCAB_SIZE, d_model=128, n_layers=4, n_heads=4, d_k=d_k).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
        
        for step in range(1, steps + 1):
            model.train()
            x_b, y_b = generate_mqar_batch(batch_size=32, n_pairs=n_pairs, seq_len=seq_len, num_tokens=NUM_TOKENS, device=device)
            
            logits = model(x_b)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y_b.view(-1), ignore_index=-100)
            
            if torch.isnan(loss):
                print(f"{ts()}  [WARNING] NaN Loss detected at step {step}, skipping optimizer step!", flush=True)
                continue
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if seed == 42 and (step % 100 == 0 or step == 1):
                with torch.no_grad():
                    mask = (y_b != -100)
                    preds = logits.argmax(dim=-1)
                    correct = (preds[mask] == y_b[mask]).sum().item()
                    total = mask.sum().item()
                    step_acc = (correct / max(1, total)) * 100.0
                    print(f"{ts()}  [Seed 42 Step {step:>4}/{steps}] Loss: {loss.item():.4f} | Dynamic MQAR Acc: {step_acc:.2f}%", flush=True)
                    
        model.eval()
        with torch.no_grad():
            eval_accs = []
            for _ in range(15):
                x_ev, y_ev = generate_mqar_batch(batch_size=32, n_pairs=n_pairs, seq_len=seq_len, num_tokens=NUM_TOKENS, device=device)
                logits_ev = model(x_ev)
                mask_ev = (y_ev != -100)
                preds_ev = logits_ev.argmax(dim=-1)
                acc_ev = (preds_ev[mask_ev] == y_ev[mask_ev]).float().mean().item() * 100.0
                eval_accs.append(acc_ev)
            final_acc = sum(eval_accs) / len(eval_accs)
            accs.append(final_acc)
            print(f"{ts()}  [Seed {seed} Final Val Accuracy]: {final_acc:.2f}%", flush=True)
            
    mean_acc = sum(accs) / len(accs)
    se_acc = (sum((a - mean_acc)**2 for a in accs) / max(1, len(accs) - 1))**0.5 / (len(accs)**0.5)
    return mean_acc, se_acc

def main():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print_log_header(device_str)
    
    # 1. POSITIVE CONTROLS (v305 Replications)
    acc_real_32, se_real_32 = evaluate_model_on_certified_harness(RealGatedDeltaNetBlock, d_k=32, name="Positive Control: Real Gated DeltaNet (d_k=32, 1024 Floats State/Head)")
    acc_complex_32, se_complex_32 = evaluate_model_on_certified_harness(ChunkwiseComplexDeltaPhaseBlock, d_k=32, name="Positive Control: Complex DeltaPhase (d_k=32, 2048 Floats State/Head)")
    
    # 2. ISO-MEMORY CONTROL (Real d_k=45 [2025 Floats State/Head])
    acc_real_45, se_real_45 = evaluate_model_on_certified_harness(RealGatedDeltaNetBlock, d_k=45, name="Iso-Memory Control: Real Gated DeltaNet (d_k=45, 2025 Floats State/Head)")
    
    print("\n" + "=" * 95, flush=True)
    print("SUMMARY OF CERTIFIED V305 ISOMEMORY BENCHMARK RESULTS (Mean ± Standard Error)", flush=True)
    print("=" * 95, flush=True)
    print(f"Chance Level Baseline: 0.195%", flush=True)
    print(f"1. Real Gated DeltaNet  (d_k=32, 1024 Floats/Head): {acc_real_32:.2f}% ± {se_real_32:.2f}%", flush=True)
    print(f"2. Complex DeltaPhase   (d_k=32, 2048 Floats/Head): {acc_complex_32:.2f}% ± {se_complex_32:.2f}% (Gap vs Real d_k=32: {acc_complex_32 - acc_real_32:+.2f}%)", flush=True)
    print(f"3. Real Gated DeltaNet  (d_k=45, 2025 Floats/Head): {acc_real_45:.2f}% ± {se_real_45:.2f}% (Iso Gap vs Complex d_k=32: {acc_complex_32 - acc_real_45:+.2f}%)", flush=True)
    print("=" * 95, flush=True)

if __name__ == "__main__":
    main()
