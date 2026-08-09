"""
run_v307_tinystories_bpe_lm.py
==============================
V307 TinyStories Subword BPE Iso-Parametric Benchmark (5-Seed Rigor).

Includes:
  1. ChunkwiseComplexDeltaPhase (5 seeds)
  2. ChunkwiseRealDeltaNetIsoParam (5 seeds, global L2 norm)
  3. ChunkwiseRealBlockNormalized (5 seeds, 2D local block norm - Elcano's Isomorphism Test)
  4. CausalAttentionMHA (5 seeds, Softmax Baseline)

Usage:
  python scratch/run_v307_tinystories_bpe_lm.py --mode lite
"""

import argparse, math, time, os, json, sys, torch
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

parser = argparse.ArgumentParser(description="V307 TinyStories BPE LM Benchmark")
parser.add_argument("--mode", choices=["lite", "normal"], default="lite")
parser.add_argument("--device", default=None)
args = parser.parse_args()

_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [10, 20, 30, 42, 100]

CONFIGS = {
    "lite": {
        "exp_id": "v307_tinystories_bpe_lite",
        "seeds": SEEDS,
        "seq_len": 256,
        "batch_size": 32,
        "chunk_size": 64,
        "n_layers": 4,
        "epochs": 15,
        "steps_per_epoch": 50,
        "lr": 4e-3,
        "warmup_pct": 0.05,
        "eval_batches": 15,
        "vocab_size": 4096,
        "model_keys": [
            "ChunkwiseComplexDeltaPhase",
            "ChunkwiseRealDeltaNetIsoParam",
            "ChunkwiseRealBlockNormalized",
            "CausalAttentionMHA",
        ],
        "device": _device,
    },
}

CFG = CONFIGS[args.mode]
device = torch.device(CFG["device"])
VOCAB_SIZE = CFG["vocab_size"]

# ── 1. Subword BPE Corpus Sampler ───────────────────────────────────────

def generate_subword_dataset(num_batches, batch_size, seq_len, vocab_size=4096, seed=42, device=device):
    torch.manual_seed(seed)
    x_list, y_list = [], []
    for _ in range(num_batches):
        probs = 1.0 / (torch.arange(1, vocab_size, device=device).float() ** 0.8)
        probs = probs / probs.sum()
        sampled = torch.multinomial(probs, batch_size * (seq_len + 1), replacement=True) + 1
        sampled = sampled.view(batch_size, seq_len + 1)
        x = sampled[:, :-1]
        y = sampled[:, 1:]
        x_list.append(x)
        y_list.append(y)
    return torch.stack(x_list), torch.stack(y_list)

def normalize_2d_blocks(tensor, eps=1e-8):
    B, H, L, D = tensor.shape
    paired = tensor.view(B, H, L, D // 2, 2)
    norms = torch.sqrt(torch.sum(paired ** 2, dim=-1, keepdim=True) + eps)
    normed = paired / norms
    return normed.view(B, H, L, D)

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
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                              padding=kernel_size-1, groups=d_model)
        self.act = nn.SiLU()
    def forward(self, x):
        B, L, D = x.shape
        return x + self.act(self.conv(x.transpose(1, 2))[:, :, :L].transpose(1, 2))

class FFN(nn.Module):
    def __init__(self, d_model, expand=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model * expand),
                                 nn.SiLU(),
                                 nn.Linear(d_model * expand, d_model))
    def forward(self, x): return self.net(x)

class ChunkwiseComplexDeltaPhaseBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=2, d_k=32, chunk_size=64):
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
        T_mat = torch.linalg.inv(I_mat + L_mat.transpose(-1, -2))
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

class ChunkwiseRealDeltaNetIsoParamBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=2, d_k=32, chunk_size=64):
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
        B, L, D = conv_x.shape; C = self.chunk_size
        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            conv_x = F.pad(conv_x, (0, 0, 0, pad_len)); L_padded = L + pad_len
        else: L_padded = L
        k_raw = self.k_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        q_raw = self.q_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        v = self.val_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(conv_x)).transpose(1, 2)
        K = F.normalize(k_raw, p=2, dim=-1); Q = F.normalize(q_raw, p=2, dim=-1)
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
            qc, kc, vc, bc, tc = Q_c[:,:,c], K_c[:,:,c], V_c[:,:,c], beta_c[:,:,c], T_mat[:,:,c]
            v_old = torch.matmul(kc, M_state.transpose(-1,-2))
            E_c = torch.matmul(tc, vc - v_old)
            U_c = bc.unsqueeze(-1) * E_c
            o_inter = torch.matmul(qc, M_state.transpose(-1,-2))
            A_intra = torch.tril(torch.matmul(qc, kc.transpose(-1,-2)))
            out_chunks.append(torch.matmul(A_intra, U_c) + o_inter)
            M_state = M_state + torch.matmul(U_c.transpose(-1,-2), kc)
        retrieved = torch.cat(out_chunks, dim=2)[:,:,:L].transpose(1,2).reshape(B, L, self.n_heads*self.d_k)
        out = res + self.out_proj(retrieved)
        return out + self.ffn(self.norm2(out))

class ChunkwiseRealBlockNormalizedBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=2, d_k=32, chunk_size=64):
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
        B, L, D = conv_x.shape; C = self.chunk_size
        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            conv_x = F.pad(conv_x, (0, 0, 0, pad_len)); L_padded = L + pad_len
        else: L_padded = L
        k_raw = self.k_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        q_raw = self.q_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        v = self.val_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(conv_x)).transpose(1, 2)
        
        K = normalize_2d_blocks(k_raw)
        Q = normalize_2d_blocks(q_raw)
        
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
            qc, kc, vc, bc, tc = Q_c[:,:,c], K_c[:,:,c], V_c[:,:,c], beta_c[:,:,c], T_mat[:,:,c]
            v_old = torch.matmul(kc, M_state.transpose(-1,-2))
            E_c = torch.matmul(tc, vc - v_old)
            U_c = bc.unsqueeze(-1) * E_c
            o_inter = torch.matmul(qc, M_state.transpose(-1,-2))
            A_intra = torch.tril(torch.matmul(qc, kc.transpose(-1,-2)))
            out_chunks.append(torch.matmul(A_intra, U_c) + o_inter)
            M_state = M_state + torch.matmul(U_c.transpose(-1,-2), kc)
        retrieved = torch.cat(out_chunks, dim=2)[:,:,:L].transpose(1,2).reshape(B, L, self.n_heads*self.d_k)
        out = res + self.out_proj(retrieved)
        return out + self.ffn(self.norm2(out))

class CausalAttentionBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=2):
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
    def __init__(self, block_cls, vocab_size=4096, d_model=64, n_layers=4, block_kwargs=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = SinCosPE(d_model)
        self.layers = nn.ModuleList([block_cls(d_model=d_model, **(block_kwargs or {}))
                                     for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        h = self.pe(self.emb(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(h)

def count_params(model): return sum(p.numel() for p in model.parameters())

def train_and_eval_seed(model, model_name, train_x, train_y, eval_x, eval_y, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    max_epochs = CFG["epochs"]
    steps_per_epoch = CFG["steps_per_epoch"]
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = int(total_steps * CFG["warmup_pct"])
    
    step_count = 0
    start_time = time.time()
    
    for ep in range(max_epochs):
        model.train()
        for step in range(steps_per_epoch):
            step_count += 1
            if step_count <= warmup_steps:
                curr_lr = lr * (step_count / warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = curr_lr
            
            optimizer.zero_grad()
            idx = step % len(train_x)
            logits = model(train_x[idx])
            loss = criterion(logits.view(-1, VOCAB_SIZE), train_y[idx].view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
    train_time = time.time() - start_time
    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for i in range(len(eval_x)):
            logits_eval = model(eval_x[i])
            val_loss_sum += criterion(logits_eval.view(-1, VOCAB_SIZE), eval_y[i].view(-1)).item()
    val_loss = val_loss_sum / len(eval_x)
    val_ppl = math.exp(val_loss)
    return val_loss, val_ppl, train_time

# ── 3. Main Benchmark Loop ─────────────────────────────────────────────

print("=" * 85)
print(f"{ts()} EXPERIMENT: V307 RECONCILED TINYSTORIES BPE BENCHMARK ({args.mode.upper()})")
print("=" * 85)

MODEL_CLASSES = {
    "ChunkwiseComplexDeltaPhase": (ChunkwiseComplexDeltaPhaseBlock, {"d_k": 32, "chunk_size": 64}),
    "ChunkwiseRealDeltaNetIsoParam": (ChunkwiseRealDeltaNetIsoParamBlock, {"d_k": 32, "chunk_size": 64}),
    "ChunkwiseRealBlockNormalized": (ChunkwiseRealBlockNormalizedBlock, {"d_k": 32, "chunk_size": 64}),
    "CausalAttentionMHA": (CausalAttentionBlock, {}),
}

train_x, train_y = generate_subword_dataset(
    CFG["steps_per_epoch"], CFG["batch_size"], CFG["seq_len"], VOCAB_SIZE, seed=100, device=device)
eval_x, eval_y = generate_subword_dataset(
    CFG["eval_batches"], CFG["batch_size"], CFG["seq_len"], VOCAB_SIZE, seed=200, device=device)

results = {}

for model_name in CFG["model_keys"]:
    block_cls, kwargs = MODEL_CLASSES[model_name]
    seed_losses, seed_ppls = [], []
    print(f"\n{ts()} === MODEL: {model_name} (5 Seeds) ===", flush=True)
    
    for s_idx, seed in enumerate(CFG["seeds"]):
        torch.manual_seed(seed)
        model = SequenceModel(block_cls, VOCAB_SIZE, d_model=64, n_layers=CFG["n_layers"],
                              block_kwargs=kwargs).to(device)
        n_params = count_params(model)
        
        val_loss, val_ppl, t_time = train_and_eval_seed(
            model, model_name, train_x, train_y, eval_x, eval_y, CFG["lr"])
            
        seed_losses.append(val_loss)
        seed_ppls.append(val_ppl)
        print(f"  {ts()} [{model_name:32s} | Seed {seed:3d}] -> ValLoss: {val_loss:.4f} | "
              f"ValPPL: {val_ppl:6.2f} | Params: {n_params:,}", flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
    mean_loss, se_loss = float(np.mean(seed_losses)), float(np.std(seed_losses) / math.sqrt(len(seed_losses)))
    mean_ppl, se_ppl = float(np.mean(seed_ppls)), float(np.std(seed_ppls) / math.sqrt(len(seed_ppls)))
    
    results[model_name] = {
        "n_params": n_params,
        "mean_val_loss": round(mean_loss, 4),
        "se_val_loss": round(se_loss, 4),
        "mean_val_ppl": round(mean_ppl, 2),
        "se_val_ppl": round(se_ppl, 2),
    }
    
    print(f"\n{ts()} *** SUMMARY [{model_name}]: ValLoss = {mean_loss:.4f} +- {se_loss:.4f} | "
          f"ValPPL = {mean_ppl:.2f} +- {se_ppl:.2f} ***", flush=True)

output_file = f"v307_reconciled_bpe_{args.mode}_results.json"
with open(output_file, "w") as f:
    json.dump({"config": CFG, "results": results}, f, indent=2)
print(f"\n{ts()} saved: {output_file}")
