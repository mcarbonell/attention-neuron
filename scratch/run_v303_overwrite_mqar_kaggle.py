# ============================================================================
# V303 OVERWRITE MQAR BENCHMARK — Kaggle/Colab Single-Cell Version
# ============================================================================
# CONFIGURE HERE: "lite" (~10-15 min on T4) or "normal" (~1-2h on T4)
MODE = "lite"
SEED = 42
# ============================================================================

import math, time, os, json, sys, torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()

def ts():
    elapsed = time.time() - T0
    h = int(elapsed // 3600)
    m = int(elapsed % 3600 // 60)
    s = elapsed % 60
    return f"[{h:02d}:{m:02d}:{s:05.2f}]"

_device = "cuda" if torch.cuda.is_available() else "cpu"

CONFIGS = {
    "lite": {
        "exp_id": "v303_overwrite_mqar_lite",
        "d_k_list": [32],
        "n_keys_list": [32],
        "overwrite_ratios": [0.0, 0.3, 0.6],
        "num_tokens": 512,
        "batch_size": 32,
        "chunk_size": 64,
        "n_layers": 4,
        "epochs": 20,
        "steps_per_epoch": 50,
        "lr_grid": [4e-3],
        "eval_batches": 10,
        "model_keys": [
            "ChunkwiseComplexDeltaPhase",
            "ChunkwiseRealDeltaNetRectangular",
            "CausalAttentionMHA",
        ],
        "seed": SEED,
        "device": _device,
    },
    "normal": {
        "exp_id": "v303_overwrite_mqar_normal",
        "d_k_list": [32, 64],
        "n_keys_list": [32, 64],
        "overwrite_ratios": [0.0, 0.3, 0.6],
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
        "seed": SEED,
        "device": _device,
    },
}

CFG = CONFIGS[MODE]
device = torch.device(CFG["device"])

ISO_FLOATS = {
    32:  {"dk_complex": 32,  "dk_real": 45,  "floats_c": 2048,   "floats_r": 2025},
    64:  {"dk_complex": 64,  "dk_real": 90,  "floats_c": 8192,   "floats_r": 8100},
    128: {"dk_complex": 128, "dk_real": 181, "floats_c": 32768,  "floats_r": 32761},
}

PAD_ID = 0
TOKEN_OFFSET = 1
NUM_TOKENS = CFG["num_tokens"]
QUERY_MARKER = TOKEN_OFFSET + NUM_TOKENS
VOCAB_SIZE = QUERY_MARKER + 1

# ── 1. Overwrite MQAR Data Generator ───────────────────────────────────

def compute_seq_len(n_keys, overwrite_ratio, chunk_size=64):
    n_overwrites = int(round(n_keys * overwrite_ratio))
    total_pairs = n_keys + n_overwrites
    min_len = 2 * total_pairs + 2 + 2 * n_keys
    return ((min_len + chunk_size - 1) // chunk_size) * chunk_size

def generate_overwrite_batch(batch_size, n_keys, overwrite_ratio, seq_len,
                             num_tokens=512, device=device):
    n_overwrites = int(round(n_keys * overwrite_ratio))
    total_pairs = n_keys + n_overwrites
    tokens_needed = n_keys + n_keys + n_overwrites
    
    assert tokens_needed <= num_tokens, f"Need {tokens_needed} tokens, but num_tokens={num_tokens}"
    
    rand_t = torch.rand(batch_size, num_tokens, device=device)
    sampled = torch.argsort(rand_t, dim=-1)[:, :tokens_needed] + TOKEN_OFFSET
    
    keys = sampled[:, :n_keys]
    vals_old = sampled[:, n_keys:2*n_keys]
    vals_new = sampled[:, 2*n_keys:]
    
    init_k = keys
    init_v = vals_old
    
    perm_init = torch.argsort(torch.rand(batch_size, n_keys, device=device), dim=-1)
    init_k = torch.gather(init_k, 1, perm_init)
    init_v = torch.gather(init_v, 1, perm_init)
    
    targets = init_v.clone()
    if n_overwrites > 0:
        ow_k = init_k[:, :n_overwrites]
        ow_v = vals_new[:, :n_overwrites]
        
        perm_ow = torch.argsort(torch.rand(batch_size, n_overwrites, device=device), dim=-1)
        ow_k = torch.gather(ow_k, 1, perm_ow)
        ow_v = torch.gather(ow_v, 1, perm_ow)
        
        targets[:, :n_overwrites] = ow_v
    
    x = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long, device=device)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    
    if n_overwrites > 0:
        all_k = torch.cat([init_k, ow_k], dim=1)
        all_v = torch.cat([init_v, ow_v], dim=1)
    else:
        all_k = init_k
        all_v = init_v
        
    kv = torch.stack([all_k, all_v], dim=2).view(batch_size, 2 * total_pairs)
    x[:, :2 * total_pairs] = kv
    
    q_perm = torch.argsort(torch.rand(batch_size, n_keys, device=device), dim=-1)
    query_starts = torch.gather(init_k, 1, q_perm)
    query_answers = torch.gather(targets, 1, q_perm)
    
    gap = 2
    pos_q = (2 * total_pairs + gap + 2 * torch.arange(n_keys, device=device)
             ).unsqueeze(0).expand(batch_size, -1)
    
    x.scatter_(1, pos_q, QUERY_MARKER)
    x.scatter_(1, pos_q + 1, query_starts)
    y.scatter_(1, pos_q + 1, query_answers)
    
    return x, y

def generate_overwrite_dataset(num_batches, batch_size, n_keys, overwrite_ratio,
                               seq_len, seed=42, device=device):
    torch.manual_seed(seed)
    x_list, y_list = [], []
    for _ in range(num_batches):
        x, y = generate_overwrite_batch(batch_size, n_keys, overwrite_ratio,
                                        seq_len, NUM_TOKENS, device)
        x_list.append(x)
        y_list.append(y)
    return torch.stack(x_list), torch.stack(y_list)

# ── 2. Building Blocks with Memory Mask ─────────────────────────────────

def compute_kv_mask(x_ids, L_padded):
    B, L = x_ids.shape
    kv_mask = torch.zeros(B, 1, L_padded, device=x_ids.device)
    for b in range(B):
        q_pos = (x_ids[b] == QUERY_MARKER).nonzero(as_tuple=False)
        kv_end = q_pos[0].item() if len(q_pos) > 0 else L
        kv_mask[b, 0, 1:kv_end:2] = 1.0
    return kv_mask

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
    iso = ISO_FLOATS[d_k_key]
    dk_c, dk_r = iso["dk_complex"], iso["dk_real"]
    return {
        "ChunkwiseComplexDeltaPhase": (
            ChunkwiseComplexDeltaPhaseBlock, {"d_k": dk_c, "chunk_size": chunk_size}),
        "ChunkwiseRealDeltaNetSquare": (
            ChunkwiseRealDeltaNetBlock, {"d_k_real": dk_r, "chunk_size": chunk_size}),
        "ChunkwiseRealDeltaNetRectangular": (
            ChunkwiseRealDeltaNetRectangularBlock, {"d_k": dk_c, "chunk_size": chunk_size}),
        "CausalAttentionMHA": (
            CausalAttentionBlock, {}),
    }

# ── 4. Training Loop ────────────────────────────────────────────────────

def train_and_eval(model, model_name, train_x, train_y, eval_x, eval_y,
                   lr, n_params=0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    max_epochs = CFG["epochs"]
    steps_per_epoch = CFG["steps_per_epoch"]
    target_batch = CFG["batch_size"]
    actual_micro = train_x.shape[1]
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

# ── 5. Main Execution ───────────────────────────────────────────────────

print("=" * 85)
print(f"{ts()} EXPERIMENT: V303 OVERWRITE MQAR BENCHMARK ({MODE.upper()})")
print("=" * 85)
print(f"{ts()}   * Exp ID:           {CFG['exp_id']}")
print(f"{ts()}   * Hardware Device:  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"{ts()}   * PyTorch Version:  {torch.__version__}")
print(f"{ts()}   * CUDA Available:   {torch.cuda.is_available()}")
print(f"{ts()}   * Seed:             {CFG['seed']}")
print(f"{ts()}   * Layers:           {CFG['n_layers']}")
print(f"{ts()}   * Chunk Size:       {CFG['chunk_size']}")
print(f"{ts()}   * Batch Size:       {CFG['batch_size']}")
print(f"{ts()}   * Epochs/Steps:     {CFG['epochs']}ep x {CFG['steps_per_epoch']}steps")
print(f"{ts()}   * LR Grid:          {CFG['lr_grid']}")
print(f"{ts()}   * Eval Batches:     {CFG['eval_batches']}")
print(f"{ts()}   * d_k Sweep:        {CFG['d_k_list']}")
print(f"{ts()}   * n_keys Sweep:     {CFG['n_keys_list']}")
print(f"{ts()}   * Overwrite Ratios: {CFG['overwrite_ratios']}")
print(f"{ts()}   * Models:           {CFG['model_keys']}")
print(f"{ts()}   * Num Tokens:       {NUM_TOKENS}")
total_runs = (len(CFG['d_k_list']) * len(CFG['n_keys_list']) *
              len(CFG['overwrite_ratios']) * len(CFG['model_keys']) * len(CFG['lr_grid']))
print(f"{ts()}   * Total Runs:       {total_runs}")
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
    
    for ow_ratio in CFG["overwrite_ratios"]:
        print(f"\n{ts()}   --- OVERWRITE RATIO = {ow_ratio:.1f} ({int(ow_ratio*100)}% keys rewritten) ---", flush=True)
        
        for n_keys in CFG["n_keys_list"]:
            seq_len = compute_seq_len(n_keys, ow_ratio, CFG["chunk_size"])
            n_overwrites = int(round(n_keys * ow_ratio))
            total_pairs = n_keys + n_overwrites
            
            print(f"\n{ts()}   >>> Generating data: {n_keys} unique keys, {n_overwrites} overwrites "
                  f"= {total_pairs} pairs, L={seq_len} <<<", flush=True)
            
            target_batch = CFG["batch_size"]
            micro_batch = 2 if seq_len >= 2048 else (4 if seq_len >= 1024 else
                          (16 if seq_len >= 512 else target_batch))
            accum_steps = max(1, target_batch // micro_batch)
            
            train_x, train_y = generate_overwrite_dataset(
                CFG["steps_per_epoch"] * accum_steps, micro_batch,
                n_keys, ow_ratio, seq_len, seed=100, device=device)
            eval_x, eval_y = generate_overwrite_dataset(
                CFG["eval_batches"], micro_batch,
                n_keys, ow_ratio, seq_len, seed=200, device=device)
            
            n_eval_seqs = CFG["eval_batches"] * micro_batch
            print(f"{ts()}   Eval: {n_eval_seqs} sequences", flush=True)
            cell_key = f"ow{int(ow_ratio*100)}_k{n_keys}"
            
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
                    "overwrite_ratio": ow_ratio,
                    "n_keys": n_keys,
                    "n_overwrites": n_overwrites,
                    "total_pairs": total_pairs,
                    "seq_len": seq_len,
                    "best_train_time_s": round(best_time, 2),
                    "best_avg_epoch_s": round(best_avg_ep, 2),
                    "all_lr_results": lr_results,
                }
                print(f"\n{ts()} *** RESULT: [{model_name:38s}] d_k={d_k_key} "
                      f"ow_ratio={ow_ratio:.1f} n_keys={n_keys} -> "
                      f"Acc: {best_acc:.2f}% (lr={best_lr}) ***", flush=True)
            
            del train_x, train_y, eval_x, eval_y
            if torch.cuda.is_available(): torch.cuda.empty_cache()

# ── 6. Summary Table ────────────────────────────────────────────────────

print(f"\n{'='*85}")
print(f"{ts()} SUMMARY TABLE — OVERWRITE MQAR")
print(f"{'='*85}")
for d_k_key in results:
    print(f"\n  d_k = {d_k_key}:")
    all_cells = set()
    for model_key in results[d_k_key]:
        all_cells.update(results[d_k_key][model_key].keys())
    all_cells = sorted(all_cells)
    header = f"  {'Model':40s}"
    for cell in all_cells:
        header += f" | {cell:>12s}"
    print(header)
    print("  " + "-" * len(header))
    for model_key in sorted(results[d_k_key].keys()):
        row = f"  {model_key:40s}"
        for cell in all_cells:
            if cell in results[d_k_key][model_key]:
                acc = results[d_k_key][model_key][cell]["best_acc"]
                row += f" | {acc:>10.2f}%"
            else:
                row += f" | {'--':>11s}"
        print(row)

# ── 7. Save ─────────────────────────────────────────────────────────────

output_file = f"v303_overwrite_mqar_{MODE}_results.json"
with open(output_file, "w") as f:
    json.dump({"config": CFG, "results": results}, f, indent=2, default=str)
print(f"\n{ts()} saved: {output_file}")
print(f"{ts()} Total elapsed: {time.time() - T0:.2f}s", flush=True)
