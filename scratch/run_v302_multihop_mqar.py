"""
run_v302_multihop_mqar.py
=========================
Multi-hop MQAR benchmark: tests whether ComplexDeltaPhase advantage
extends to compositional memory access (chaining lookups across layers).

Modes:
  --mode lite   : Quick feedback (~15 min on T4). Complex vs RealRect only,
                  single d_k, single chain count, 1 LR.
  --mode normal : Paper-quality sweep (~4-6h on T4). All 4 models,
                  2 d_k scales, multiple chain counts, 3 LRs.

Usage:
  python run_v302_multihop_mqar.py --mode lite
  python run_v302_multihop_mqar.py --mode normal --device cuda --seed 42
"""

import argparse, math, time, os, json, sys, torch
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

# ── CLI ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Multi-hop MQAR benchmark for DeltaPhase")
parser.add_argument("--mode", choices=["lite", "normal"], default="lite",
                    help="lite = fast feedback, normal = paper-quality sweep")
parser.add_argument("--device", default=None,
                    help="Force device (default: cuda if available, else cpu)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

# ── Configs ──────────────────────────────────────────────────────────────
CONFIGS = {
    "lite": {
        "exp_id": "v302_multihop_mqar_lite",
        "d_k_list": [32],
        "n_hops_list": [1, 2, 3],
        "n_chains_list": [16, 32],
        "num_tokens": 512,
        "batch_size": 32,
        "chunk_size": 64,
        "n_layers": 4,
        "epochs": 20,
        "steps_per_epoch": 50,
        "lr_grid": [2e-3, 4e-3, 8e-3],
        "eval_batches": 10,
        "model_keys": [
            "ChunkwiseComplexDeltaPhase",
            "ChunkwiseRealDeltaNetRectangular",
            "CausalAttentionMHA",
        ],
        "seed": args.seed,
        "device": _device,
    },
    "normal": {
        "exp_id": "v302_multihop_mqar_normal",
        "d_k_list": [32, 64],
        "n_hops_list": [1, 2, 3],
        "n_chains_list": [16, 32, 64, 128],
        "num_tokens": 512,
        "batch_size": 32,
        "chunk_size": 64,
        "n_layers": 4,
        "epochs": 20,
        "steps_per_epoch": 50,
        "lr_grid": [2e-3, 4e-3, 8e-3],
        "eval_batches": 20,
        "model_keys": [
            "ChunkwiseComplexDeltaPhase",
            "ChunkwiseRealDeltaNetSquare",
            "ChunkwiseRealDeltaNetRectangular",
            "CausalAttentionMHA",
        ],
        "seed": args.seed,
        "device": _device,
    },
}

CFG = CONFIGS[args.mode]
device = torch.device(CFG["device"])

# iso-floats map (same as v300)
ISO_FLOATS = {
    32:  {"dk_complex": 32,  "dk_real": 45,  "floats_c": 2048,   "floats_r": 2025},
    64:  {"dk_complex": 64,  "dk_real": 90,  "floats_c": 8192,   "floats_r": 8100},
    128: {"dk_complex": 128, "dk_real": 181, "floats_c": 32768,  "floats_r": 32761},
}

# ── Vocab ────────────────────────────────────────────────────────────────
PAD_ID = 0
TOKEN_OFFSET = 1
NUM_TOKENS = CFG["num_tokens"]
QUERY_MARKER = TOKEN_OFFSET + NUM_TOKENS
VOCAB_SIZE = QUERY_MARKER + 1

# ── 1. Multi-hop Data Generator (vectorized, GPU) ───────────────────────

def compute_seq_len(n_chains, n_hops, chunk_size=64):
    """Minimum sequence length, rounded up to chunk_size."""
    total_pairs = n_chains * n_hops
    min_len = 2 * total_pairs + 2 + 2 * n_chains  # KV + gap + queries
    return ((min_len + chunk_size - 1) // chunk_size) * chunk_size

def generate_multihop_batch(batch_size, n_chains, n_hops, seq_len,
                            num_tokens=512, device=device):
    """
    Generate one batch of multi-hop MQAR data.
    
    For n_hops=h, each chain uses h+1 distinct tokens: t0, t1, ..., th.
    Stored as pairs: (t0,t1), (t1,t2), ..., (t_{h-1}, th).
    Query: QUERY t0 -> expect th.
    
    For h=1, this reduces to standard MQAR.
    """
    tokens_needed = n_chains * (n_hops + 1)
    assert tokens_needed <= num_tokens, (
        f"Need {tokens_needed} unique tokens for {n_chains} chains x {n_hops} hops, "
        f"but num_tokens={num_tokens}"
    )
    
    # Pick tokens_needed distinct tokens per batch element
    rand_t = torch.rand(batch_size, num_tokens, device=device)
    all_tokens = torch.argsort(rand_t, dim=-1)[:, :tokens_needed] + TOKEN_OFFSET
    
    # Reshape into chains: (B, n_chains, n_hops+1)
    chain_tokens = all_tokens.view(batch_size, n_chains, n_hops + 1)
    
    # Build pairs from chain links: key=t[i], val=t[i+1]
    pair_keys = chain_tokens[:, :, :-1].reshape(batch_size, -1)  # (B, n_chains*n_hops)
    pair_vals = chain_tokens[:, :, 1:].reshape(batch_size, -1)
    total_pairs = n_chains * n_hops
    
    # Shuffle pair order (crucial: model can't rely on adjacency)
    perm = torch.argsort(torch.rand(batch_size, total_pairs, device=device), dim=-1)
    pair_keys = torch.gather(pair_keys, 1, perm)
    pair_vals = torch.gather(pair_vals, 1, perm)
    
    # Build sequence
    x = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    
    # KV section: k1 v1 k2 v2 ...
    kv = torch.stack([pair_keys, pair_vals], dim=2).view(batch_size, 2 * total_pairs)
    x[:, :2 * total_pairs] = kv
    
    # Query section
    query_starts = chain_tokens[:, :, 0]    # first token of each chain
    query_answers = chain_tokens[:, :, -1]  # last token (target after h hops)
    
    # Shuffle query order
    q_perm = torch.argsort(torch.rand(batch_size, n_chains, device=device), dim=-1)
    query_starts = torch.gather(query_starts, 1, q_perm)
    query_answers = torch.gather(query_answers, 1, q_perm)
    
    # Place queries: QUERY_MARKER t0 QUERY_MARKER t0' ...
    gap = 2
    pos_q = (2 * total_pairs + gap + 2 * torch.arange(n_chains, device=device)
             ).unsqueeze(0).expand(batch_size, -1)
    
    x.scatter_(1, pos_q, QUERY_MARKER)
    x.scatter_(1, pos_q + 1, query_starts)
    y.scatter_(1, pos_q + 1, query_answers)
    
    return x, y

def generate_multihop_dataset(num_batches, batch_size, n_chains, n_hops,
                               seq_len, seed=42, device=device):
    torch.manual_seed(seed)
    x_list, y_list = [], []
    for _ in range(num_batches):
        x, y = generate_multihop_batch(batch_size, n_chains, n_hops, seq_len,
                                        NUM_TOKENS, device)
        x_list.append(x)
        y_list.append(y)
    return torch.stack(x_list), torch.stack(y_list)

# ── 2. Building Blocks (identical to v300_v2) ───────────────────────────
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

def compute_kv_mask(x_ids, L_padded):
    B, L = x_ids.shape
    kv_mask = torch.zeros(B, 1, L_padded, device=x_ids.device)
    for b in range(B):
        q_pos = (x_ids[b] == QUERY_MARKER).nonzero(as_tuple=False)
        kv_end = q_pos[0].item() if len(q_pos) > 0 else L
        kv_mask[b, 0, 1:kv_end:2] = 1.0
    return kv_mask

class ChunkwiseComplexDeltaPhaseBlock(nn.Module):
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

    def forward(self, x, x_ids=None):
        res = x; conv_x = self.causal_conv(self.norm1(x))
        B, L, D = conv_x.shape
        C = self.chunk_size; inv_dk = 1.0 / float(self.d_k)
        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            conv_x = F.pad(conv_x, (0, 0, 0, pad_len)); L_padded = L + pad_len
        else: L_padded = L
        theta_k = self.theta_k_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        theta_q = self.theta_q_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        v = self.val_proj(conv_x).view(B, L_padded, self.n_heads, self.d_k).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(conv_x)).transpose(1, 2)
        if x_ids is not None:
            beta = beta * compute_kv_mask(x_ids, L_padded)
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

class ChunkwiseRealDeltaNetBlock(nn.Module):
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

    def forward(self, x, x_ids=None):
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
        if x_ids is not None:
            beta = beta * compute_kv_mask(x_ids, L_padded)
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

class ChunkwiseRealDeltaNetRectangularBlock(nn.Module):
    def __init__(self, d_model, n_heads=2, d_k=64, chunk_size=64):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.d_key, self.d_val, self.chunk_size = 2 * d_k, d_k, chunk_size
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_key)
        self.q_proj = nn.Linear(d_model, n_heads * self.d_key)
        self.val_proj = nn.Linear(d_model, n_heads * self.d_val)
        self.beta_proj = nn.Linear(d_model, n_heads)
        self.out_proj = nn.Linear(n_heads * self.d_val, d_model)
        self.ffn = FFN(d_model)

    def forward(self, x, x_ids=None):
        res = x; conv_x = self.causal_conv(self.norm1(x))
        B, L, D = conv_x.shape; C = self.chunk_size
        pad_len = (C - (L % C)) % C
        if pad_len > 0:
            conv_x = F.pad(conv_x, (0, 0, 0, pad_len)); L_padded = L + pad_len
        else: L_padded = L
        k_raw = self.k_proj(conv_x).view(B, L_padded, self.n_heads, self.d_key).transpose(1, 2)
        q_raw = self.q_proj(conv_x).view(B, L_padded, self.n_heads, self.d_key).transpose(1, 2)
        v = self.val_proj(conv_x).view(B, L_padded, self.n_heads, self.d_val).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(conv_x)).transpose(1, 2)
        if x_ids is not None:
            beta = beta * compute_kv_mask(x_ids, L_padded)
        K = F.normalize(k_raw, p=2, dim=-1); Q = F.normalize(q_raw, p=2, dim=-1)
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
            qc, kc, vc, bc, tc = Q_c[:,:,c], K_c[:,:,c], V_c[:,:,c], beta_c[:,:,c], T_mat[:,:,c]
            v_old = torch.matmul(kc, M_state.transpose(-1,-2))
            E_c = torch.matmul(tc, vc - v_old)
            U_c = bc.unsqueeze(-1) * E_c
            o_inter = torch.matmul(qc, M_state.transpose(-1,-2))
            A_intra = torch.tril(torch.matmul(qc, kc.transpose(-1,-2)))
            out_chunks.append(torch.matmul(A_intra, U_c) + o_inter)
            M_state = M_state + torch.matmul(U_c.transpose(-1,-2), kc)
        retrieved = torch.cat(out_chunks, dim=2)[:,:,:L].transpose(1,2).reshape(B, L, self.n_heads*self.d_val)
        out = res + self.out_proj(retrieved)
        return out + self.ffn(self.norm2(out))

class CausalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=2):
        super().__init__()
        self.causal_conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)
    def forward(self, x, x_ids=None):
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
        self.layers = nn.ModuleList([block_cls(d_model=d_model, **(block_kwargs or {}))
                                     for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        x_ids = x
        h = self.pe(self.emb(x))
        for layer in self.layers:
            h = layer(h, x_ids=x_ids)
        return self.head(h)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def print_model_topology(model, model_name):
    print(f"\n{ts()} --- TOPOLOGY & PARAMETERS: {model_name} ---")
    emb_p = sum(p.numel() for p in model.emb.parameters())
    head_p = sum(p.numel() for p in model.head.parameters())
    layer_p = sum(p.numel() for p in model.layers[0].parameters())
    total_layers_p = sum(p.numel() for p in model.layers.parameters())
    total_p = sum(p.numel() for p in model.parameters())
    
    print(f"{ts()}   [Embedding]  nn.Embedding(vocab={VOCAB_SIZE}, d_model={model.emb.embedding_dim}) -> {emb_p:,} params")
    print(f"{ts()}   [PosEnc]     SinCosPE(d_model={model.emb.embedding_dim}) -> 0 params (fixed buffer)")
    print(f"{ts()}   [Layers]     {len(model.layers)} x {model.layers[0].__class__.__name__} -> {layer_p:,} params/layer (Total Layers: {total_layers_p:,} params)")
    
    block = model.layers[0]
    for sub_name, sub_module in block.named_children():
        sub_p = sum(p.numel() for p in sub_module.parameters())
        print(f"{ts()}      |- {sub_name:15s}: {sub_module.__class__.__name__:25s} -> {sub_p:>8,} params")
        
    print(f"{ts()}   [LM Head]    nn.Linear(d_model={model.emb.embedding_dim}, vocab={VOCAB_SIZE}) -> {head_p:,} params")
    print(f"{ts()}   [TOTAL]      {total_p:,} trainable parameters")
    print(f"{ts()} ---------------------------------------------------------", flush=True)

# ── 3. Model Registry ───────────────────────────────────────────────────
def get_model_specs(d_k_key, chunk_size):
    """Returns (name, block_cls, block_kwargs) for all models, keyed by name."""
    iso = ISO_FLOATS[d_k_key]
    dk_c, dk_r = iso["dk_complex"], iso["dk_real"]
    return {
        "ChunkwiseComplexDeltaPhase": (
            ChunkwiseComplexDeltaPhaseBlock,
            {"d_k": dk_c, "chunk_size": chunk_size}
        ),
        "ChunkwiseRealDeltaNetSquare": (
            ChunkwiseRealDeltaNetBlock,
            {"d_k_real": dk_r, "chunk_size": chunk_size}
        ),
        "ChunkwiseRealDeltaNetRectangular": (
            ChunkwiseRealDeltaNetRectangularBlock,
            {"d_k": dk_c, "chunk_size": chunk_size}
        ),
        "CausalAttentionMHA": (
            CausalAttentionBlock,
            {}
        ),
    }

# ── 4. Training Loop ────────────────────────────────────────────────────
def train_and_eval(model, model_name, train_x, train_y, eval_x, eval_y,
                   lr, n_params=0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    max_epochs = CFG["epochs"]
    steps_per_epoch = CFG["steps_per_epoch"]
    target_batch = CFG["batch_size"]

    # Determine micro-batch from the pre-generated data shape
    actual_micro = train_x.shape[1]  # second dim after stacking
    seq_len = train_x.shape[2]
    accum_steps = max(1, target_batch // actual_micro)

    epoch_times = []
    start_time = time.time()
    for ep in range(max_epochs):
        ep_start = time.time()
        model.train()
        epoch_loss_sum = 0.0
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
        ep_time = time.time() - ep_start
        epoch_times.append(ep_time)
        avg_loss = epoch_loss_sum / steps_per_epoch
        print(f"  {ts()} [{model_name:38s} | lr={lr:.4f} | {n_params:>8,}p] "
              f"Epoch {ep+1:2d}/{max_epochs:2d} | AvgLoss = {avg_loss:.4f} | "
              f"EpTime = {ep_time:.2f}s", flush=True)
    train_time = time.time() - start_time

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(len(eval_x)):
            logits = model(eval_x[i])
            preds = logits.argmax(dim=-1)
            mask = (eval_y[i] != -100)
            correct += (preds[mask] == eval_y[i][mask]).sum().item()
            total += mask.sum().item()
    acc = (correct / total) * 100.0 if total > 0 else 0.0
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    avg_ep = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    return acc, train_time, avg_ep

# ── 5. Main ─────────────────────────────────────────────────────────────
print("=" * 85)
print(f"{ts()} EXPERIMENT: V302 MULTI-HOP MQAR BENCHMARK ({args.mode.upper()})")
print("=" * 85)
print(f"{ts()}   * Exp ID:          {CFG['exp_id']}")
print(f"{ts()}   * Hardware Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"{ts()}   * PyTorch Version: {torch.__version__}")
print(f"{ts()}   * CUDA Available:  {torch.cuda.is_available()}")
print(f"{ts()}   * Seed:            {CFG['seed']}")
print(f"{ts()}   * Layers:          {CFG['n_layers']} (n_layers > max_hops for chain capacity)")
print(f"{ts()}   * Chunk Size:      {CFG['chunk_size']}")
print(f"{ts()}   * Batch Size:      {CFG['batch_size']}")
print(f"{ts()}   * Epochs/Steps:    {CFG['epochs']}ep x {CFG['steps_per_epoch']}steps")
print(f"{ts()}   * LR Grid:         {CFG['lr_grid']}")
print(f"{ts()}   * Eval Batches:    {CFG['eval_batches']}")
print(f"{ts()}   * d_k Sweep:       {CFG['d_k_list']}")
print(f"{ts()}   * Hops Sweep:      {CFG['n_hops_list']}")
print(f"{ts()}   * Chains Sweep:    {CFG['n_chains_list']}")
print(f"{ts()}   * Models:          {CFG['model_keys']}")
print(f"{ts()}   * Num Tokens:      {NUM_TOKENS} (shared key/value space)")
total_runs = (len(CFG['d_k_list']) * len(CFG['n_hops_list']) *
              len(CFG['n_chains_list']) * len(CFG['model_keys']) * len(CFG['lr_grid']))
print(f"{ts()}   * Total Runs:      {total_runs}")
print("=" * 85, flush=True)

results = {}
run_counter = 0

for d_k in CFG["d_k_list"]:
    d_k_key = int(d_k)
    iso = ISO_FLOATS[d_k_key]
    dk_c = iso["dk_complex"]
    d_model = 2 * dk_c
    all_specs = get_model_specs(d_k_key, CFG["chunk_size"])
    
    print(f"\n{ts()} === SWEEP d_k={dk_c} (d_model={d_model}) ===", flush=True)
    results[d_k_key] = {}

    for n_hops in CFG["n_hops_list"]:
        print(f"\n{ts()}   --- HOPS = {n_hops} ---", flush=True)
        
        for n_chains in CFG["n_chains_list"]:
            tokens_needed = n_chains * (n_hops + 1)
            if tokens_needed > NUM_TOKENS:
                print(f"{ts()}   SKIP: {n_chains} chains x {n_hops} hops needs "
                      f"{tokens_needed} tokens > {NUM_TOKENS}", flush=True)
                continue
            
            seq_len = compute_seq_len(n_chains, n_hops, CFG["chunk_size"])
            total_pairs = n_chains * n_hops
            
            print(f"\n{ts()}   >>> Generating data: {n_chains} chains x {n_hops} hops "
                  f"= {total_pairs} pairs, L={seq_len} <<<", flush=True)
            
            target_batch = CFG["batch_size"]
            micro_batch = 2 if seq_len >= 2048 else (4 if seq_len >= 1024 else
                          (16 if seq_len >= 512 else target_batch))
            accum_steps = max(1, target_batch // micro_batch)
            
            train_x, train_y = generate_multihop_dataset(
                CFG["steps_per_epoch"] * accum_steps, micro_batch,
                n_chains, n_hops, seq_len, seed=100, device=device)
            eval_x, eval_y = generate_multihop_dataset(
                CFG["eval_batches"], micro_batch,
                n_chains, n_hops, seq_len, seed=200, device=device)
            
            n_eval_seqs = CFG["eval_batches"] * micro_batch
            print(f"{ts()}   Eval: {n_eval_seqs} sequences", flush=True)
            
            cell_key = f"hops{n_hops}_chains{n_chains}"
            
            for model_name in CFG["model_keys"]:
                block_cls, block_kwargs = all_specs[model_name]
                best_acc, best_lr, best_time, best_avg_ep = -1.0, None, 0.0, 0.0
                lr_results = []
                
                for lr_idx, lr in enumerate(CFG["lr_grid"]):
                    run_counter += 1
                    torch.manual_seed(CFG["seed"])
                    model = SequenceModel(block_cls, VOCAB_SIZE, d_model,
                                          CFG["n_layers"], block_kwargs).to(device)
                    n_params = count_params(model)
                    if lr_idx == 0:
                        print_model_topology(model, model_name)
                    
                    print(f"\n{ts()}   [{run_counter}/{total_runs}] "
                          f"{model_name} | lr={lr} | Params: {n_params:,}", flush=True)
                    
                    acc, train_time, avg_ep = train_and_eval(
                        model, model_name, train_x, train_y,
                        eval_x, eval_y, lr, n_params=n_params)
                    
                    lr_results.append({
                        "lr": lr, "acc": round(acc, 2),
                        "train_time_s": round(train_time, 2),
                        "avg_epoch_s": round(avg_ep, 2)
                    })
                    
                    print(f"{ts()}   >> Acc: {acc:.2f}% | Time: {train_time:.2f}s | "
                          f"AvgEp: {avg_ep:.2f}s", flush=True)
                    
                    if acc > best_acc:
                        best_acc, best_lr = acc, lr
                        best_time, best_avg_ep = train_time, avg_ep
                    
                    del model
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                
                result_key = f"{model_name}_dk{d_k_key}"
                if result_key not in results[d_k_key]:
                    results[d_k_key][result_key] = {}
                results[d_k_key][result_key][cell_key] = {
                    "best_acc": round(best_acc, 2),
                    "best_lr": best_lr,
                    "n_params": n_params,
                    "n_hops": n_hops,
                    "n_chains": n_chains,
                    "total_pairs": total_pairs,
                    "seq_len": seq_len,
                    "best_train_time_s": round(best_time, 2),
                    "best_avg_epoch_s": round(best_avg_ep, 2),
                    "all_lr_results": lr_results,
                }
                
                print(f"\n{ts()} *** RESULT: [{model_name:38s}] d_k={d_k_key} "
                      f"hops={n_hops} chains={n_chains} -> "
                      f"Acc: {best_acc:.2f}% (lr={best_lr}) ***", flush=True)
            
            # Free datasets
            del train_x, train_y, eval_x, eval_y
            if torch.cuda.is_available(): torch.cuda.empty_cache()

# ── 6. Summary Table ────────────────────────────────────────────────────
print(f"\n{'='*85}")
print(f"{ts()} SUMMARY TABLE")
print(f"{'='*85}")
for d_k_key in results:
    print(f"\n  d_k = {d_k_key}:")
    # Collect all models and cells
    all_cells = set()
    for model_key in results[d_k_key]:
        all_cells.update(results[d_k_key][model_key].keys())
    all_cells = sorted(all_cells)
    
    # Header
    header = f"  {'Model':40s}"
    for cell in all_cells:
        header += f" | {cell:>16s}"
    print(header)
    print("  " + "-" * len(header))
    
    for model_key in sorted(results[d_k_key].keys()):
        row = f"  {model_key:40s}"
        for cell in all_cells:
            if cell in results[d_k_key][model_key]:
                acc = results[d_k_key][model_key][cell]["best_acc"]
                row += f" | {acc:>14.2f}%"
            else:
                row += f" | {'--':>15s}"
        print(row)

# ── 7. Save ─────────────────────────────────────────────────────────────
output_file = f"v302_multihop_mqar_{args.mode}_results.json"
with open(output_file, "w") as f:
    json.dump({"config": CFG, "results": results}, f, indent=2, default=str)
print(f"\n{ts()} saved: {output_file}")
print(f"{ts()} Total elapsed: {time.time() - T0:.2f}s", flush=True)
