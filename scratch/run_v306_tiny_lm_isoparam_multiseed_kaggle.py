# ============================================================================
# V306 TINY LM ISO-PARAMETRIC MULTI-SEED — Kaggle/Colab Single-Cell Version
# ============================================================================
MODE = "lite"
SEEDS = [10, 20, 30, 42, 100]
# ============================================================================

import math, time, os, json, sys, urllib.request, torch
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

_device = "cuda" if torch.cuda.is_available() else "cpu"

CONFIGS = {
    "lite": {
        "exp_id": "v306_tiny_lm_isoparam_lite",
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
        "model_keys": [
            "ChunkwiseComplexDeltaPhase",
            "ChunkwiseRealDeltaNetIsoParam",
            "CausalAttentionMHA",
        ],
        "device": _device,
    },
}

CFG = CONFIGS[MODE]
device = torch.device(CFG["device"])

# ── 1. Text Corpus & Tokenizer ──────────────────────────────────────────

def get_text_corpus():
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
        return text
    except Exception as e:
        print(f"{ts()} Synthetic English fallback...", flush=True)
        words = ["the", "king", "said", "to", "the", "queen", "that", "courage", "and", "honor",
                 "belong", "to", "the", "brave", "knights", "of", "the", "realm"]
        import random
        random.seed(42)
        return " ".join([random.choice(words) for _ in range(50000)])

corpus_text = get_text_corpus()
chars = sorted(list(set(corpus_text)))
TOKEN_OFFSET = 1
char2idx = {ch: i + TOKEN_OFFSET for i, ch in enumerate(chars)}
VOCAB_SIZE = len(chars) + TOKEN_OFFSET + 1

encoded_tensor = torch.tensor([char2idx[ch] for ch in corpus_text], dtype=torch.long)
n_train = int(len(encoded_tensor) * 0.9)
train_data = encoded_tensor[:n_train]
val_data = encoded_tensor[n_train:]

def sample_lm_batch(dataset, batch_size, seq_len, device=device):
    max_idx = len(dataset) - seq_len - 1
    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([dataset[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([dataset[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y

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
    """
    Real-valued DeltaNet configured for EXACT ISO-PARAMETERS with ComplexDeltaPhase:
    d_model=64, n_heads=2, d_k=32 (key_dim=32, val_dim=32).
    Yields EXACTLY 144,331 trainable parameters.
    """
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
    def __init__(self, block_cls, vocab_size, d_model=64, n_layers=4, block_kwargs=None):
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

def train_and_eval_seed(model, model_name, train_data, val_data, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    max_epochs = CFG["epochs"]
    steps_per_epoch = CFG["steps_per_epoch"]
    batch_size = CFG["batch_size"]
    seq_len = CFG["seq_len"]
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
            x, y = sample_lm_batch(train_data, batch_size, seq_len, device=device)
            logits = model(x)
            loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
    train_time = time.time() - start_time
    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for _ in range(CFG["eval_batches"]):
            x_v, y_v = sample_lm_batch(val_data, batch_size, seq_len, device=device)
            logits_v = model(x_v)
            val_loss_sum += criterion(logits_v.view(-1, VOCAB_SIZE), y_v.view(-1)).item()
            
    val_loss = val_loss_sum / CFG["eval_batches"]
    val_ppl = math.exp(min(val_loss, 20.0))
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return val_loss, val_ppl, train_time

print("=" * 85)
print(f"{ts()} EXPERIMENT: V306 TINY LM ISO-PARAMETRIC MULTI-SEED ({MODE.upper()})")
print("=" * 85)

MODEL_CLASSES = {
    "ChunkwiseComplexDeltaPhase": (ChunkwiseComplexDeltaPhaseBlock, {"d_k": 32, "chunk_size": 64}),
    "ChunkwiseRealDeltaNetIsoParam": (ChunkwiseRealDeltaNetIsoParamBlock, {"d_k": 32, "chunk_size": 64}),
    "CausalAttentionMHA": (CausalAttentionBlock, {}),
}

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
            model, model_name, train_data, val_data, CFG["lr"])
            
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
        "seed_losses": [round(l, 4) for l in seed_losses],
        "seed_ppls": [round(p, 2) for p in seed_ppls],
    }
    
    print(f"\n{ts()} *** SUMMARY [{model_name}]: ValLoss = {mean_loss:.4f} +- {se_loss:.4f} | "
          f"ValPPL = {mean_ppl:.2f} +- {se_ppl:.2f} ***", flush=True)

print(f"\n{'='*85}")
print(f"{ts()} SUMMARY TABLE — V306 ISO-PARAMETRIC MULTI-SEED (LEVEL 2 ANCLA)")
print(f"{'='*85}")
header = f"  {'Model':35s} | {'Params':>10s} | {'Mean ValLoss +- SE':>22s} | {'Mean ValPPL +- SE':>22s}"
print(header)
print("  " + "-" * len(header))

best_model = min(results.keys(), key=lambda m: results[m]["mean_val_loss"])

for m in sorted(results.keys()):
    info = results[m]
    star = " 🌟" if m == best_model else ""
    loss_str = f"{info['mean_val_loss']:.4f} +- {info['se_val_loss']:.4f}"
    ppl_str = f"{info['mean_val_ppl']:.2f} +- {info['se_val_ppl']:.2f}{star}"
    row = f"  {m:35s} | {info['n_params']:>10,} | {loss_str:>22s} | {ppl_str:>22s}"
    print(row)

output_file = f"v306_tiny_lm_isoparam_{MODE}_results.json"
with open(output_file, "w") as f:
    json.dump({"config": CFG, "results": results}, f, indent=2, default=str)
print(f"\n{ts()} saved: {output_file}")
