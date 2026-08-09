"""
run_v304_tiny_lm.py
===================
V304 Tiny Language Modeling Benchmark (Next-Token Prediction & Perplexity).

Evaluates ChunkwiseComplexDeltaPhase vs Real-valued DeltaNet controls & Softmax MHA
on a real/structured text corpus (Tiny Shakespeare / Synthetic Language Task).

Measures:
  - Cross-Entropy Loss
  - Validation Perplexity (PPL = exp(loss))
  - Training speed & Memory Efficiency

Modes:
  --mode lite   : Fast feedback (~10-15 min on T4).
  --mode normal : Comprehensive sweep (~1-2h on T4).

Usage:
  python scratch/run_v304_tiny_lm.py --mode lite
  python scratch/run_v304_tiny_lm.py --mode normal --device cuda --seed 42
"""

import argparse, math, time, os, json, sys, urllib.request, torch
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
parser = argparse.ArgumentParser(description="Tiny Language Modeling Benchmark for DeltaPhase")
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
        "exp_id": "v304_tiny_lm_lite",
        "d_k_list": [32],
        "seq_len": 256,
        "batch_size": 32,
        "chunk_size": 64,
        "n_layers": 4,
        "epochs": 15,
        "steps_per_epoch": 50,
        "lr_grid": [4e-3],
        "eval_batches": 15,
        "model_keys": [
            "ChunkwiseComplexDeltaPhase",
            "ChunkwiseRealDeltaNetRectangular",
            "CausalAttentionMHA",
        ],
        "seed": args.seed,
        "device": _device,
    },
    "normal": {
        "exp_id": "v304_tiny_lm_normal",
        "d_k_list": [32, 64],
        "seq_len": 384,
        "batch_size": 32,
        "chunk_size": 64,
        "n_layers": 4,
        "epochs": 20,
        "steps_per_epoch": 80,
        "lr_grid": [2e-3, 4e-3, 8e-3],
        "eval_batches": 25,
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

# ISO-Floats Map
ISO_FLOATS = {
    32:  {"dk_complex": 32,  "dk_real": 45,  "floats_c": 2048,   "floats_r": 2025},
    64:  {"dk_complex": 64,  "dk_real": 90,  "floats_c": 8192,   "floats_r": 8100},
}

# ── 1. Text Corpus & Tokenizer ──────────────────────────────────────────

def get_text_corpus():
    """Fetches Tiny Shakespeare or generates fallback synthetic structured corpus."""
    file_path = "data_tiny_shakespeare.txt"
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    try:
        print(f"{ts()} Fetching Tiny Shakespeare corpus...", flush=True)
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        urllib.request.urlretrieve(url, file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"{ts()} Downloaded {len(text):,} characters.", flush=True)
        return text
    except Exception as e:
        print(f"{ts()} Could not download text ({e}). Using synthetic English corpus.", flush=True)
        # Fallback synthetic text
        words = ["the", "king", "said", "to", "the", "queen", "that", "courage", "and", "honor",
                 "belong", "to", "the", "brave", "knights", "of", "the", "realm", "where", "the",
                 "sun", "shines", "brightly", "over", "the", "castle", "and", "the", "mountains"]
        import random
        random.seed(42)
        text = " ".join([random.choice(words) for _ in range(50000)])
        return text

corpus_text = get_text_corpus()

# Character-level Tokenizer
chars = sorted(list(set(corpus_text)))
PAD_ID = 0
TOKEN_OFFSET = 1
char2idx = {ch: i + TOKEN_OFFSET for i, ch in enumerate(chars)}
idx2char = {i + TOKEN_OFFSET: ch for i, ch in enumerate(chars)}
VOCAB_SIZE = len(chars) + TOKEN_OFFSET + 1  # +1 for safety / markers

encoded_tensor = torch.tensor([char2idx[ch] for ch in corpus_text], dtype=torch.long)
n_tokens_total = len(encoded_tensor)

# Train / Val Split (90% / 10%)
n_train = int(n_tokens_total * 0.9)
train_data = encoded_tensor[:n_train]
val_data = encoded_tensor[n_train:]

def sample_lm_batch(dataset, batch_size, seq_len, device=device):
    max_idx = len(dataset) - seq_len - 1
    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([dataset[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([dataset[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y

def generate_lm_dataset(num_batches, dataset, batch_size, seq_len, seed=42, device=device):
    torch.manual_seed(seed)
    x_list, y_list = [], []
    for _ in range(num_batches):
        x, y = sample_lm_batch(dataset, batch_size, seq_len, device)
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
        h = self.pe(self.emb(x))
        for layer in self.layers:
            h = layer(h)
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

def train_and_eval_lm(model, model_name, train_data, val_data, lr, n_params=0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    max_epochs = CFG["epochs"]
    steps_per_epoch = CFG["steps_per_epoch"]
    batch_size = CFG["batch_size"]
    seq_len = CFG["seq_len"]
    
    epoch_times = []
    start_time = time.time()
    
    for ep in range(max_epochs):
        ep_start = time.time()
        model.train()
        epoch_loss_sum = 0.0
        
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            x, y = sample_lm_batch(train_data, batch_size, seq_len, device=device)
            logits = model(x)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss_sum += loss.item()
            
        ep_time = time.time() - ep_start
        epoch_times.append(ep_time)
        avg_train_loss = epoch_loss_sum / steps_per_epoch
        train_ppl = math.exp(min(avg_train_loss, 20.0))
        
        print(f"  {ts()} [{model_name:38s} | lr={lr:.4f} | {n_params:>8,}p] "
              f"Epoch {ep+1:2d}/{max_epochs:2d} | TrainLoss = {avg_train_loss:.4f} | "
              f"TrainPPL = {train_ppl:6.2f} | EpTime = {ep_time:.2f}s", flush=True)
              
    train_time = time.time() - start_time
    
    # Validation Evaluation
    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for _ in range(CFG["eval_batches"]):
            x_v, y_v = sample_lm_batch(val_data, batch_size, seq_len, device=device)
            logits_v = model(x_v)
            val_loss_sum += criterion(logits_v.view(-1, VOCAB_SIZE), y_v.view(-1)).item()
            
    val_loss = val_loss_sum / CFG["eval_batches"]
    val_ppl = math.exp(min(val_loss, 20.0))
    avg_ep = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return val_loss, val_ppl, train_time, avg_ep

# ── 5. Main Execution ───────────────────────────────────────────────────

print("=" * 85)
print(f"{ts()} EXPERIMENT: V304 TINY LANGUAGE MODELING BENCHMARK ({args.mode.upper()})")
print("=" * 85)
print(f"{ts()}   * Exp ID:           {CFG['exp_id']}")
print(f"{ts()}   * Hardware Device:  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"{ts()}   * PyTorch Version:  {torch.__version__}")
print(f"{ts()}   * CUDA Available:   {torch.cuda.is_available()}")
print(f"{ts()}   * Seed:             {CFG['seed']}")
print(f"{ts()}   * Layers:           {CFG['n_layers']}")
print(f"{ts()}   * Chunk Size:       {CFG['chunk_size']}")
print(f"{ts()}   * Batch Size:       {CFG['batch_size']}")
print(f"{ts()}   * Seq Length:       {CFG['seq_len']}")
print(f"{ts()}   * Epochs/Steps:     {CFG['epochs']}ep x {CFG['steps_per_epoch']}steps")
print(f"{ts()}   * LR Grid:          {CFG['lr_grid']}")
print(f"{ts()}   * Eval Batches:     {CFG['eval_batches']}")
print(f"{ts()}   * d_k Sweep:        {CFG['d_k_list']}")
print(f"{ts()}   * Models:           {CFG['model_keys']}")
print(f"{ts()}   * Vocab Size:       {VOCAB_SIZE} (char-level)")
print(f"{ts()}   * Train/Val Tokens: {len(train_data):,} / {len(val_data):,}")
total_runs = len(CFG['d_k_list']) * len(CFG['model_keys']) * len(CFG['lr_grid'])
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
    
    for model_name in CFG["model_keys"]:
        block_cls, block_kwargs = all_specs[model_name]
        best_val_loss, best_val_ppl = float("inf"), float("inf")
        best_lr, best_time, best_avg_ep = None, 0.0, 0.0
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
                  
            val_loss, val_ppl, train_time, avg_ep = train_and_eval_lm(
                model, model_name, train_data, val_data, lr, n_params=n_params)
                
            lr_results.append({
                "lr": lr,
                "val_loss": round(val_loss, 4),
                "val_ppl": round(val_ppl, 2),
                "train_time_s": round(train_time, 2),
                "avg_epoch_s": round(avg_ep, 2)
            })
            
            print(f"{ts()}   >> ValLoss: {val_loss:.4f} | ValPPL: {val_ppl:.2f} | "
                  f"Time: {train_time:.2f}s | AvgEp: {avg_ep:.2f}s", flush=True)
                  
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_ppl = val_ppl
                best_lr = lr
                best_time = train_time
                best_avg_ep = avg_ep
                
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        results[d_k_key][model_name] = {
            "best_val_loss": round(best_val_loss, 4),
            "best_val_ppl": round(best_val_ppl, 2),
            "best_lr": best_lr,
            "n_params": n_params,
            "best_train_time_s": round(best_time, 2),
            "best_avg_epoch_s": round(best_avg_ep, 2),
            "all_lr_results": lr_results,
        }
        
        print(f"\n{ts()} *** RESULT: [{model_name:38s}] d_k={d_k_key} -> "
              f"ValLoss: {best_val_loss:.4f} | ValPPL: {best_val_ppl:.2f} (lr={best_lr}) ***", flush=True)

# ── 6. Summary Table ────────────────────────────────────────────────────

print(f"\n{'='*85}")
print(f"{ts()} SUMMARY TABLE — TINY LANGUAGE MODELING (PERPLEXITY)")
print(f"{'='*85}")
for d_k_key in results:
    print(f"\n  d_k = {d_k_key}:")
    header = f"  {'Model':40s} | {'Params':>10s} | {'Best ValLoss':>12s} | {'Best ValPPL':>12s} | {'Best LR':>8s}"
    print(header)
    print("  " + "-" * len(header))
    for model_key in sorted(results[d_k_key].keys()):
        info = results[d_k_key][model_key]
        row = (f"  {model_key:40s} | {info['n_params']:>10,} | "
               f"{info['best_val_loss']:>12.4f} | {info['best_val_ppl']:>12.2f} | {info['best_lr']:>8}")
        print(row)

# ── 7. Save ─────────────────────────────────────────────────────────────

output_file = f"v304_tiny_lm_{args.mode}_results.json"
with open(output_file, "w") as f:
    json.dump({"config": CFG, "results": results}, f, indent=2, default=str)
print(f"\n{ts()} saved: {output_file}")
print(f"{ts()} Total elapsed: {time.time() - T0:.2f}s", flush=True)
