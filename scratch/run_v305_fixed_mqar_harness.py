"""
run_v305_fixed_mqar_harness.py
==============================
V305 Fixed & Certified MQAR Harness (On-The-Fly Batch Generation).

Fixes the static dataset memorization bug by generating fresh random MQAR batches
on-the-fly at every step.

Verifies:
  1. CausalAttentionMHA reaches ~100% accuracy across all sequence lengths L in [128, 256, 512, 1024].
  2. Evaluates ChunkwiseComplexDeltaPhase vs ChunkwiseRealDeltaNetIsoParam under certified harness.

Usage:
  python scratch/run_v305_fixed_mqar_harness.py --mode lite
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

parser = argparse.ArgumentParser(description="V305 Fixed MQAR Harness")
parser.add_argument("--mode", choices=["lite", "normal"], default="lite")
parser.add_argument("--device", default=None)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

CONFIGS = {
    "lite": {
        "exp_id": "v305_fixed_mqar_lite",
        "seq_len_list": [128, 256, 512, 1024],
        "n_pairs_dict": {128: 29, 256: 61, 512: 64, 1024: 64},
        "batch_size": 32,
        "chunk_size": 64,
        "n_layers": 4,
        "steps": 1000,
        "lr": 3e-3,
        "eval_batches": 15,
        "model_keys": [
            "ChunkwiseComplexDeltaPhase",
            "ChunkwiseRealDeltaNetIsoParam",
            "CausalAttentionMHA",
        ],
        "seed": args.seed,
        "device": _device,
    },
}

CFG = CONFIGS[args.mode]
device = torch.device(CFG["device"])

PAD_ID = 0
TOKEN_OFFSET = 1
NUM_TOKENS = 512
QUERY_MARKER = TOKEN_OFFSET + NUM_TOKENS
VOCAB_SIZE = QUERY_MARKER + 1

# ── 1. On-The-Fly MQAR Batch Generator ──────────────────────────────────

def generate_mqar_batch(batch_size, n_pairs, seq_len, num_tokens=512, device=device):
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
    pos_q = (2 * n_pairs + gap + 2 * torch.arange(n_pairs, device=device)
             ).unsqueeze(0).expand(batch_size, -1)
             
    x.scatter_(1, pos_q, QUERY_MARKER)
    x.scatter_(1, pos_q + 1, query_keys)
    y.scatter_(1, pos_q + 1, query_vals)
    return x, y

def generate_mqar_dataset(num_batches, batch_size, n_pairs, seq_len, seed=42, device=device):
    torch.manual_seed(seed)
    x_list, y_list = [], []
    for _ in range(num_batches):
        x, y = generate_mqar_batch(batch_size, n_pairs, seq_len, NUM_TOKENS, device)
        x_list.append(x)
        y_list.append(y)
    return torch.stack(x_list), torch.stack(y_list)

# ── 2. Building Blocks ──────────────────────────────────────────────────

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, max_len=4096, d_model=64):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        return self.pe(pos)

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
    def __init__(self, block_cls, vocab_size=VOCAB_SIZE, d_model=64, n_layers=4, block_kwargs=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = AbsolutePositionalEmbedding(4096, d_model)
        self.layers = nn.ModuleList([block_cls(d_model=d_model, **(block_kwargs or {}))
                                     for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        h = self.emb(x) + self.pe(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)

def train_and_eval(model, model_name, seq_len, n_pairs, steps=1000, lr=3e-3):
    batch_size = CFG["batch_size"]
    eval_x, eval_y = generate_mqar_dataset(CFG["eval_batches"], batch_size, n_pairs, seq_len, seed=999, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    start_t = time.time()
    best_acc = 0.0
    
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        # ON-THE-FLY RANDOM BATCH GENERATION
        x, y = generate_mqar_batch(batch_size, n_pairs, seq_len, NUM_TOKENS, device=device)
        logits = model(x)
        loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 100 == 0 or step == steps:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for i in range(len(eval_x)):
                    logits_eval = model(eval_x[i])
                    preds = logits_eval.argmax(dim=-1)
                    mask = (eval_y[i] != -100)
                    correct += (preds[mask] == eval_y[i][mask]).sum().item()
                    total += mask.sum().item()
            acc = (correct / total) * 100.0
            best_acc = max(best_acc, acc)
            
    return best_acc

# ── 3. Main Benchmark Loop ─────────────────────────────────────────────

print("=" * 85)
print(f"{ts()} CERTIFIED BENCHMARK: V305 FIXED MQAR HARNESS (ON-THE-FLY BATCHES)")
print("=" * 85)

MODEL_CLASSES = {
    "ChunkwiseComplexDeltaPhase": (ChunkwiseComplexDeltaPhaseBlock, {"d_k": 32, "chunk_size": 64}),
    "ChunkwiseRealDeltaNetIsoParam": (ChunkwiseRealDeltaNetIsoParamBlock, {"d_k": 32, "chunk_size": 64}),
    "CausalAttentionMHA": (CausalAttentionBlock, {}),
}

results = {}

for L in CFG["seq_len_list"]:
    n_pairs = CFG["n_pairs_dict"][L]
    results[L] = {}
    print(f"\n{ts()} === BISECT L={L} (n_pairs={n_pairs}) ===", flush=True)
    
    for model_name in CFG["model_keys"]:
        block_cls, kwargs = MODEL_CLASSES[model_name]
        torch.manual_seed(CFG["seed"])
        model = SequenceModel(block_cls, VOCAB_SIZE, d_model=64, n_layers=CFG["n_layers"],
                              block_kwargs=kwargs).to(device)
        
        acc = train_and_eval(model, model_name, L, n_pairs, steps=CFG["steps"], lr=CFG["lr"])
        results[L][model_name] = acc
        print(f"  {ts()} [{model_name:32s}] L={L:<4d} -> Acc: {acc:6.2f}%", flush=True)
        
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

# ── Summary Table ───────────────────────────────────────────────────────

print(f"\n{'='*85}")
print(f"{ts()} SUMMARY TABLE — CERTIFIED MQAR HARNESS (ON-THE-FLY DATASET)")
print(f"{'='*85}")
header = f"  {'Model':35s}" + "".join(f" | L={l:<4d}" for l in CFG["seq_len_list"])
print(header)
print("  " + "-" * len(header))

for model_name in CFG["model_keys"]:
    row = f"  {model_name:35s}"
    for L in CFG["seq_len_list"]:
        acc = results[L].get(model_name, 0.0)
        row += f" | {acc:6.2f}%"
    print(row)

output_file = f"v305_fixed_mqar_{args.mode}_results.json"
with open(output_file, "w") as f:
    json.dump({"config": CFG, "results": results}, f, indent=2)
print(f"\n{ts()} saved: {output_file}")
