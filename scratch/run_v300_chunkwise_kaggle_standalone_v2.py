"""
run_v300_chunkwise_kaggle_standalone_v2.py
==========================================
V2 of V300 Chunkwise Benchmark for Kaggle GPU.
Changes vs v1:
  - [HH:MM:SS] timestamps (elapsed from script start) on ALL output lines
  - Epoch-averaged loss (v1 reported only last step)
  - Per-LR training time reported individually (v1 accumulated all 3)
  - Model param count in logs for iso-params auditability
  - Larger eval set (20 batches vs 10) for lower SE on accuracy
  - Scaling asymmetry documented (see block comments)
"""

import math, time, os, json, sys, torch
import torch.nn as nn
import torch.nn.functional as F

# ── Global Timer ─────────────────────────────────────────────────────────
T0 = time.time()

def ts():
    """Returns [HH:MM:SS.ss] elapsed since script start."""
    elapsed = time.time() - T0
    h = int(elapsed // 3600)
    m = int(elapsed % 3600 // 60)
    s = elapsed % 60
    return f"[{h:02d}:{m:02d}:{s:05.2f}]"

# ── Config ───────────────────────────────────────────────────────────────
CFG = {
    "exp_id": "v300_capacity_scaling_chunkwise_standalone_v2",
    "d_k_list": [32, 64, 128],
    "iso_floats_map": {
        32: {"dk_complex": 32, "dk_real": 45, "floats_c": 2048, "floats_r": 2025},
        64: {"dk_complex": 64, "dk_real": 90, "floats_c": 8192, "floats_r": 8100},
        128: {"dk_complex": 128, "dk_real": 181, "floats_c": 32768, "floats_r": 32761},
        "32": {"dk_complex": 32, "dk_real": 45, "floats_c": 2048, "floats_r": 2025},
        "64": {"dk_complex": 64, "dk_real": 90, "floats_c": 8192, "floats_r": 8100},
        "128": {"dk_complex": 128, "dk_real": 181, "floats_c": 32768, "floats_r": 32761}
    },
    "num_pairs_list": [32, 64, 128, 256],
    "num_keys": 256,
    "num_vals": 256,
    "batch_size": 32,
    "chunk_size": 64,
    "n_layers": 3,
    "epochs": 20,
    "steps_per_epoch": 50,
    "lr_grid": [2e-3, 4e-3, 8e-3],
    "eval_batches": 20,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

device = torch.device(CFG["device"])

PAD_ID = 0
KEY_OFFSET = 1
VAL_OFFSET = 1 + CFG["num_keys"]
QUERY_MARKER = VAL_OFFSET + CFG["num_vals"]
VOCAB_SIZE = QUERY_MARKER + 1

# ── 1. Vectorized Dataset Generator (100% on GPU) ───────────────────────
def generate_mqar_batch_vectorized(batch_size, num_pairs, seq_len, num_keys=256, num_vals=256, device=device):
    rand_k = torch.rand(batch_size, num_keys, device=device)
    keys = torch.argsort(rand_k, dim=-1)[:, :num_pairs] + KEY_OFFSET
    vals = torch.randint(0, num_vals, (batch_size, num_pairs), device=device) + VAL_OFFSET
    
    x = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    
    kv_interleaved = torch.stack([keys, vals], dim=2).view(batch_size, 2 * num_pairs)
    x[:, :2 * num_pairs] = kv_interleaved
    
    rand_q = torch.rand(batch_size, num_pairs, device=device)
    query_perm = torch.argsort(rand_q, dim=-1)
    
    q_keys = torch.gather(keys, 1, query_perm)
    q_vals = torch.gather(vals, 1, query_perm)
    
    n_queries = min(num_pairs, (seq_len - 2 * num_pairs - 2) // 2)
    pos_q = (2 * num_pairs + 2 + 2 * torch.arange(n_queries, device=device)).unsqueeze(0).expand(batch_size, -1)
    
    x.scatter_(1, pos_q, QUERY_MARKER)
    x.scatter_(1, pos_q + 1, q_keys[:, :n_queries])
    y.scatter_(1, pos_q + 1, q_vals[:, :n_queries])
    return x, y

def generate_mqar_dataset(num_batches, batch_size, num_pairs, seq_len, seed=42, device=device):
    torch.manual_seed(seed)
    x_list, y_list = [], []
    for _ in range(num_batches):
        x, y = generate_mqar_batch_vectorized(batch_size, num_pairs, seq_len, device=device)
        x_list.append(x)
        y_list.append(y)
    return torch.stack(x_list), torch.stack(y_list)

# ── 2. Building Blocks ──────────────────────────────────────────────────
class SinCosPE(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.shape[1]]

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
    """
    Complex-valued DeltaNet with phase-based keys.
    
    SCALING NOTE: Uses inv_dk = 1/d_k for Gram, A_intra, and retrieval ops.
    This is needed because keys are torch.polar(1, theta) -> each component has
    magnitude 1, so ||k|| = sqrt(d_k). The dot product K@conj(K)^T has entries 
    of magnitude up to d_k. Dividing by d_k normalizes to O(1).
    
    Real blocks use L2-normalized keys (||k||=1), so their dot products are 
    already O(1) without explicit scaling. The asymmetry is INTENTIONAL.
    """
    def __init__(self, d_model, n_heads=2, d_k=64, chunk_size=64):
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
        B, L, D = conv_x.shape
        C = self.chunk_size; inv_dk = 1.0 / float(self.d_k)
        
        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            conv_x = F.pad(conv_x, (0, 0, 0, pad_len))
            L_padded = L + pad_len
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
        T_mat = torch.linalg.inv(I_mat + L_mat.transpose(-1, -2))
        
        M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, dtype=torch.complex64, device=x.device)
        out_chunks = []
        
        for c in range(num_chunks):
            qc, kc, vc, bc, tc = Q_c[:, :, c], K_c[:, :, c], V_c[:, :, c], beta_c[:, :, c], T_mat[:, :, c]
            v_old_inter = torch.matmul(M_state, torch.conj(kc).transpose(-1, -2)).real.transpose(-1, -2) * inv_dk
            v_eff = vc - v_old_inter
            E_c = torch.matmul(tc, v_eff)
            U_c = bc.unsqueeze(-1) * E_c
            o_inter = torch.matmul(M_state, torch.conj(qc).transpose(-1, -2)).real.transpose(-1, -2) * inv_dk
            A_intra = torch.tril(torch.matmul(qc, torch.conj(kc).transpose(-1, -2)).real) * inv_dk
            o_intra = torch.matmul(A_intra, U_c)
            out_chunks.append(o_intra + o_inter)
            M_state = M_state + torch.matmul(U_c.to(torch.complex64).transpose(-1, -2), kc)
            
        retrieved = torch.cat(out_chunks, dim=2)[:, :, :L].transpose(1, 2).reshape(B, L, self.n_heads * self.d_k)
        out = res + self.out_proj(retrieved)
        return out + self.ffn(self.norm2(out))

class ChunkwiseRealDeltaNetBlock(nn.Module):
    """
    Real-valued DeltaNet with L2-normalized keys (square state d_k x d_k).
    
    SCALING NOTE: No explicit inv_dk applied. Keys are L2-normalized (||k||=1),
    so K@K^T entries are in [-1,1] natively. This is asymmetric vs Complex 
    block by design — see Complex block docstring for rationale.
    """
    def __init__(self, d_model, n_heads=2, d_k_real=90, chunk_size=64):
        super().__init__()
        self.d_model, self.n_heads, self.d_k, self.chunk_size = d_model, n_heads, d_k_real, chunk_size
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_k)
        self.q_proj = nn.Linear(d_model, n_heads * self.d_k)
        self.val_proj = nn.Linear(d_model, n_heads * self.d_k)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(n_heads * self.d_k, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x; conv_x = self.causal_conv(self.norm1(x))
        B, L, D = conv_x.shape
        C = self.chunk_size
        
        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            conv_x = F.pad(conv_x, (0, 0, 0, pad_len))
            L_padded = L + pad_len
        else: L_padded = L

        k_raw = self.k_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        q_raw = self.q_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        v = self.val_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(conv_x)).transpose(1, 2)
        
        K = F.normalize(k_raw, p=2, dim=-1)
        Q = F.normalize(q_raw, p=2, dim=-1)
        
        num_chunks = L_padded // C
        Q_c = Q.view(B, self.n_heads, num_chunks, C, self.d_k)
        K_c = K.view(B, self.n_heads, num_chunks, C, self.d_k)
        V_c = v.view(B, self.n_heads, num_chunks, C, self.d_k)
        beta_c = beta.view(B, self.n_heads, num_chunks, C)
        
        Gram = torch.matmul(K_c, K_c.transpose(-1, -2))
        L_mat = torch.triu(Gram * beta_c.unsqueeze(-1), diagonal=1)
        I_mat = torch.eye(C, device=x.device).view(1, 1, 1, C, C)
        T_mat = torch.linalg.inv(I_mat + L_mat.transpose(-1, -2))
        
        M_state = torch.zeros(B, self.n_heads, self.d_k, self.d_k, device=x.device)
        out_chunks = []
        
        for c in range(num_chunks):
            qc, kc, vc, bc, tc = Q_c[:, :, c], K_c[:, :, c], V_c[:, :, c], beta_c[:, :, c], T_mat[:, :, c]
            v_old_inter = torch.matmul(kc, M_state.transpose(-1, -2))
            v_eff = vc - v_old_inter
            E_c = torch.matmul(tc, v_eff)
            U_c = bc.unsqueeze(-1) * E_c
            o_inter = torch.matmul(qc, M_state.transpose(-1, -2))
            A_intra = torch.tril(torch.matmul(qc, kc.transpose(-1, -2)))
            o_intra = torch.matmul(A_intra, U_c)
            out_chunks.append(o_intra + o_inter)
            M_state = M_state + torch.matmul(U_c.transpose(-1, -2), kc)
            
        retrieved = torch.cat(out_chunks, dim=2)[:, :, :L].transpose(1, 2).reshape(B, L, self.n_heads * self.d_k)
        out = res + self.out_proj(retrieved)
        return out + self.ffn(self.norm2(out))

class ChunkwiseRealDeltaNetRectangularBlock(nn.Module):
    """
    Real-valued DeltaNet with rectangular state (d_val x d_key) where d_key=2*d_k.
    Matches Complex state float count exactly: d_k * 2*d_k = 2*d_k^2 floats.
    
    NOTE: This block has ~50% more projection params than Complex because
    k_proj and q_proj map to d_key=2*d_k dimensions vs d_k in Complex.
    This is an inherent cost of the rectangular real parameterization.
    The param counts are logged for transparency.
    """
    def __init__(self, d_model, n_heads=2, d_k=64, chunk_size=64):
        super().__init__()
        self.d_model, self.n_heads, self.d_key, self.d_val, self.chunk_size = d_model, n_heads, 2 * d_k, d_k, chunk_size
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_key)
        self.q_proj = nn.Linear(d_model, n_heads * self.d_key)
        self.val_proj = nn.Linear(d_model, n_heads * self.d_val)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(n_heads * self.d_val, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        res = x; conv_x = self.causal_conv(self.norm1(x))
        B, L, D = conv_x.shape
        C = self.chunk_size
        
        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            conv_x = F.pad(conv_x, (0, 0, 0, pad_len))
            L_padded = L + pad_len
        else: L_padded = L

        k_raw = self.k_proj(conv_x).view(B, L_padded, self.n_heads, self.d_key).transpose(1, 2)
        q_raw = self.q_proj(conv_x).view(B, L_padded, self.n_heads, self.d_key).transpose(1, 2)
        v = self.val_proj(conv_x).view(B, L_padded, self.n_heads, self.d_val).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(conv_x)).transpose(1, 2)
        
        K = F.normalize(k_raw, p=2, dim=-1)
        Q = F.normalize(q_raw, p=2, dim=-1)
        
        num_chunks = L_padded // C
        Q_c = Q.view(B, self.n_heads, num_chunks, C, self.d_key)
        K_c = K.view(B, self.n_heads, num_chunks, C, self.d_key)
        V_c = v.view(B, self.n_heads, num_chunks, C, self.d_val)
        beta_c = beta.view(B, self.n_heads, num_chunks, C)
        
        Gram = torch.matmul(K_c, K_c.transpose(-1, -2))
        L_mat = torch.triu(Gram * beta_c.unsqueeze(-1), diagonal=1)
        I_mat = torch.eye(C, device=x.device).view(1, 1, 1, C, C)
        T_mat = torch.linalg.inv(I_mat + L_mat.transpose(-1, -2))
        
        M_state = torch.zeros(B, self.n_heads, self.d_val, self.d_key, device=x.device)
        out_chunks = []
        
        for c in range(num_chunks):
            qc, kc, vc, bc, tc = Q_c[:, :, c], K_c[:, :, c], V_c[:, :, c], beta_c[:, :, c], T_mat[:, :, c]
            v_old_inter = torch.matmul(kc, M_state.transpose(-1, -2))
            v_eff = vc - v_old_inter
            E_c = torch.matmul(tc, v_eff)
            U_c = bc.unsqueeze(-1) * E_c
            o_inter = torch.matmul(qc, M_state.transpose(-1, -2))
            A_intra = torch.tril(torch.matmul(qc, kc.transpose(-1, -2)))
            o_intra = torch.matmul(A_intra, U_c)
            out_chunks.append(o_intra + o_inter)
            M_state = M_state + torch.matmul(U_c.transpose(-1, -2), kc)
            
        retrieved = torch.cat(out_chunks, dim=2)[:, :, :L].transpose(1, 2).reshape(B, L, self.n_heads * self.d_val)
        out = res + self.out_proj(retrieved)
        return out + self.ffn(self.norm2(out))

class CausalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=2):
        super().__init__()
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
    def forward(self, x):
        res = x; conv_x = self.causal_conv(self.norm1(x))
        B, L, D = conv_x.shape
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=conv_x.device), diagonal=1)
        attn_out, _ = self.mha(conv_x, conv_x, conv_x, attn_mask=causal_mask, is_causal=False)
        return res + attn_out + self.ffn(self.norm2(res + attn_out))

class SequenceModel(nn.Module):
    def __init__(self, block_cls, vocab_size, d_model, n_layers=3, block_kwargs=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = SinCosPE(d_model)
        self.layers = nn.ModuleList([block_cls(d_model=d_model, **(block_kwargs or {})) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        h = self.pe(self.emb(x))
        for layer in self.layers: h = layer(h)
        return self.head(h)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# ── 3. Training Loop with Pre-Generated Datasets ────────────────────────
def train_and_eval_pregenerated(model, model_name, train_x, train_y, eval_x, eval_y,
                                 lr, epochs=15, steps_per_epoch=50, seq_len=256,
                                 n_params=0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    target_batch = CFG["batch_size"]
    max_epochs = CFG["epochs"]
    micro_batch = 2 if seq_len >= 2048 else (4 if seq_len >= 1024 else (16 if seq_len >= 512 else target_batch))
    accum_steps = max(1, target_batch // micro_batch)
    
    epoch_times = []
    start_time = time.time()
    for ep in range(max_epochs):
        ep_start = time.time()
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            step_loss = 0.0
            for acc_i in range(accum_steps):
                idx = (step * accum_steps + acc_i) % len(train_x)
                logits = model(train_x[idx])
                loss = criterion(logits.view(-1, VOCAB_SIZE), train_y[idx].view(-1)) / accum_steps
                loss.backward()
                step_loss += loss.item() * accum_steps
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss_sum += step_loss
            epoch_loss_count += 1
        ep_time = time.time() - ep_start
        epoch_times.append(ep_time)
        avg_loss = epoch_loss_sum / epoch_loss_count
        print(f"  {ts()} [{model_name:38s} | lr={lr:.4f} | {n_params:>8,}p] "
              f"Epoch {ep+1:2d}/{max_epochs:2d} | AvgLoss = {avg_loss:.4f} | EpTime = {ep_time:.2f}s", flush=True)
    train_time = time.time() - start_time
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(len(eval_x)):
            logits = model(eval_x[i]); preds = logits.argmax(dim=-1); mask = (eval_y[i] != -100)
            correct += (preds[mask] == eval_y[i][mask]).sum().item(); total += mask.sum().item()
    acc = (correct / total) * 100.0 if total > 0 else 0.0
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    avg_ep_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    return acc, train_time, avg_ep_time, epoch_times

# ── 4. Main Execution Header & Loop ─────────────────────────────────────
print("=" * 85)
print(f"{ts()} EXPERIMENT BENCHMARK RUN: V300 STANDALONE CHUNKWISE CAPACITY SCALING (V2)")
print("=" * 85)
print(f"{ts()}   * Exp ID:          {CFG['exp_id']}")
print(f"{ts()}   * Hardware Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"{ts()}   * PyTorch Version: {torch.__version__}")
print(f"{ts()}   * CUDA Available:  {torch.cuda.is_available()} (Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'})")
print(f"{ts()}   * Seed:            {CFG['seed']}")
print(f"{ts()}   * Chunk Size C:    {CFG['chunk_size']}")
print(f"{ts()}   * Layers:          {CFG['n_layers']}")
print(f"{ts()}   * Effective Batch: {CFG['batch_size']}")
print(f"{ts()}   * Epochs / Steps:  {CFG['epochs']} epochs, {CFG['steps_per_epoch']} steps/epoch")
print(f"{ts()}   * LR Grid:         {CFG['lr_grid']}")
print(f"{ts()}   * Eval Batches:    {CFG['eval_batches']}")
print(f"{ts()}   * Sweeps d_k:      {CFG['d_k_list']}")
print(f"{ts()}   * KV Pairs Sweep:  {CFG['num_pairs_list']}")
print(f"{ts()}   * ISO-FLOATS MAP:")
for dk, info in CFG["iso_floats_map"].items():
    if isinstance(dk, int):
        print(f"{ts()}       d_k={dk:3d} -> Complex d_k={info['dk_complex']:3d} ({info['floats_c']:5d} floats/head) | Real d_k={info['dk_real']:3d} ({info['floats_r']:5d} floats/head)")
print("=" * 85, flush=True)

results_matrix = {}
for d_k in CFG["d_k_list"]:
    d_k_key = int(d_k)
    iso_info = CFG["iso_floats_map"][d_k_key]
    dk_c, dk_r = iso_info["dk_complex"], iso_info["dk_real"]
    d_model = 2 * dk_c
    print(f"\n{ts()} === SWEEP d_k = {dk_c} (d_model = {d_model}) ===", flush=True)
    results_matrix[d_k_key] = {}
    model_specs = [
        ("ChunkwiseComplexDeltaPhase", ChunkwiseComplexDeltaPhaseBlock, {"d_k": dk_c, "chunk_size": CFG["chunk_size"]}),
        ("ChunkwiseRealDeltaNetSquare", ChunkwiseRealDeltaNetBlock, {"d_k_real": dk_r, "chunk_size": CFG["chunk_size"]}),
        ("ChunkwiseRealDeltaNetRectangular", ChunkwiseRealDeltaNetRectangularBlock, {"d_k": dk_c, "chunk_size": CFG["chunk_size"]}),
        ("CausalAttentionMHA", CausalAttentionBlock, {})
    ]
    for num_pairs in CFG["num_pairs_list"]:
        seq_len = 8 * num_pairs
        print(f"\n{ts()}   >>> Pre-generating GPU Datasets for Load: {num_pairs} Pairs (L={seq_len}) <<<", flush=True)
        
        target_batch = CFG["batch_size"]
        micro_batch = 2 if seq_len >= 2048 else (4 if seq_len >= 1024 else (16 if seq_len >= 512 else target_batch))
        accum_steps = max(1, target_batch // micro_batch)
        
        train_x, train_y = generate_mqar_dataset(CFG["steps_per_epoch"] * accum_steps, micro_batch, num_pairs, seq_len, seed=100, device=device)
        eval_x, eval_y   = generate_mqar_dataset(CFG["eval_batches"], micro_batch, num_pairs, seq_len, seed=200, device=device)
        
        n_eval_seqs = CFG["eval_batches"] * micro_batch
        print(f"{ts()}   Eval set: {CFG['eval_batches']} batches x {micro_batch} = {n_eval_seqs} sequences", flush=True)
        
        for name, block_cls, block_kwargs in model_specs:
            best_acc, best_lr, best_train_time, best_avg_ep = -1.0, None, 0.0, 0.0
            lr_results = []
            for lr in CFG["lr_grid"]:
                torch.manual_seed(CFG["seed"])
                model = SequenceModel(block_cls, VOCAB_SIZE, d_model, CFG["n_layers"], block_kwargs).to(device)
                n_params = count_params(model)
                print(f"\n{ts()}   -- {name} | lr={lr} | Params: {n_params:,} --", flush=True)
                acc, train_time, avg_ep_time, epoch_times = train_and_eval_pregenerated(
                    model, name, train_x, train_y, eval_x, eval_y, lr,
                    epochs=CFG["epochs"], steps_per_epoch=CFG["steps_per_epoch"],
                    seq_len=seq_len, n_params=n_params)
                lr_results.append({
                    "lr": lr, "acc": round(acc, 2),
                    "train_time": round(train_time, 2),
                    "avg_epoch_time": round(avg_ep_time, 2)
                })
                print(f"{ts()}   >> {name} | lr={lr} -> Acc: {acc:.2f}% | TrainTime: {train_time:.2f}s | AvgEp: {avg_ep_time:.2f}s", flush=True)
                if acc > best_acc:
                    best_acc, best_lr = acc, lr
                    best_train_time, best_avg_ep = train_time, avg_ep_time
                del model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            key_name = f"{name}_dk{d_k_key}"
            if key_name not in results_matrix[d_k_key]: results_matrix[d_k_key][key_name] = {}
            results_matrix[d_k_key][key_name][num_pairs] = {
                "best_acc": round(best_acc, 2),
                "best_lr": best_lr,
                "best_train_time_s": round(best_train_time, 2),
                "best_avg_epoch_time_s": round(best_avg_ep, 2),
                "n_params": n_params,
                "all_lr_results": lr_results
            }
            print(f"\n{ts()} *** RESULT: [{name:38s} | d_k={d_k_key}] Pairs={num_pairs:3d} (L={seq_len:4d}) -> "
                  f"Best Acc: {best_acc:6.2f}% (lr={best_lr}) | Params: {n_params:,} ***", flush=True)

# ── 5. Save Results ─────────────────────────────────────────────────────
output_path = "v300_chunkwise_standalone_v2_results.json"
with open(output_path, "w") as f:
    json.dump({"config": CFG, "results": results_matrix}, f, indent=2)
print(f"\n{ts()} BENCHMARK COMPLETE! Results saved to {output_path}", flush=True)
print(f"{ts()} Total elapsed: {time.time() - T0:.2f}s", flush=True)
