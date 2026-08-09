# ============================================================================
# V305 MQAR HARNESS DEBUG & CERTIFICATION — Kaggle/Colab Single-Cell Version
# ============================================================================
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
        "exp_id": "v305_debug_mqar_lite",
        "d_k_list": [32],
        "seq_len_list": [128, 256, 512, 1024],
        "batch_size": 32,
        "chunk_size": 64,
        "n_layers": 4,
        "epochs": 20,
        "steps_per_epoch": 50,
        "lr": 4e-3,
        "warmup_pct": 0.05,
        "eval_batches": 10,
        "model_keys": [
            "ChunkwiseComplexDeltaPhase",
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
    32: {"dk_complex": 32, "dk_real": 45, "floats_c": 2048, "floats_r": 2025},
}

PAD_ID = 0
TOKEN_OFFSET = 1
NUM_TOKENS = 512
QUERY_MARKER = TOKEN_OFFSET + NUM_TOKENS
VOCAB_SIZE = QUERY_MARKER + 1

# ── 1. Synthetic MQAR Data Generator ────────────────────────────────────

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

def train_and_eval(model, model_name, train_x, train_y, eval_x, eval_y, lr,
                   use_warmup=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    max_epochs = CFG["epochs"]
    steps_per_epoch = CFG["steps_per_epoch"]
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = int(total_steps * CFG["warmup_pct"]) if use_warmup else 0
    
    step_count = 0
    start_time = time.time()
    
    for ep in range(max_epochs):
        model.train()
        for step in range(steps_per_epoch):
            step_count += 1
            if use_warmup and step_count <= warmup_steps:
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
    return acc, train_time

print("=" * 85)
print(f"{ts()} EXPERIMENT: V305 MQAR HARNESS CERTIFICATION ({MODE.upper()})")
print("=" * 85)

results = {}

for seq_len in CFG["seq_len_list"]:
    n_pairs = min(64, (seq_len - 10) // 4)
    print(f"\n{ts()} === BISECT L={seq_len} (n_pairs={n_pairs}) ===", flush=True)
    results[seq_len] = {}
    
    train_x, train_y = generate_mqar_dataset(
        CFG["steps_per_epoch"], CFG["batch_size"], n_pairs, seq_len, seed=100, device=device)
    eval_x, eval_y = generate_mqar_dataset(
        CFG["eval_batches"], CFG["batch_size"], n_pairs, seq_len, seed=200, device=device)
        
    for use_warmup in [True, False]:
        mode_str = "WITH_Warmup" if use_warmup else "NO_Warmup"
        print(f"\n{ts()}   --- Mode: {mode_str} ---", flush=True)
        
        for model_name in CFG["model_keys"]:
            d_model = 64
            if model_name == "ChunkwiseComplexDeltaPhase":
                block_cls, kwargs = ChunkwiseComplexDeltaPhaseBlock, {"d_k": 32, "chunk_size": 64}
            elif model_name == "ChunkwiseRealDeltaNetRectangular":
                block_cls, kwargs = ChunkwiseRealDeltaNetRectangularBlock, {"d_k": 32, "chunk_size": 64}
            else:
                block_cls, kwargs = CausalAttentionBlock, {}
                
            torch.manual_seed(CFG["seed"])
            model = SequenceModel(block_cls, VOCAB_SIZE, d_model, CFG["n_layers"], kwargs).to(device)
            acc, train_time = train_and_eval(model, model_name, train_x, train_y,
                                             eval_x, eval_y, CFG["lr"], use_warmup=use_warmup)
                                             
            cell_key = f"{model_name}_{mode_str}"
            results[seq_len][cell_key] = round(acc, 2)
            print(f"{ts()}   [{model_name:38s} | {mode_str}] L={seq_len} -> Acc: {acc:6.2f}%", flush=True)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

print(f"\n{'='*85}")
print(f"{ts()} SUMMARY TABLE — V305 HARNESS CERTIFICATION")
print(f"{'='*85}")
header = f"  {'Model & Mode':50s}" + "".join(f" | L={l:<4d}" for l in CFG["seq_len_list"])
print(header)
print("  " + "-" * len(header))

all_keys = sorted(list(results[CFG["seq_len_list"][0]].keys()))
for key in all_keys:
    row = f"  {key:50s}"
    for l in CFG["seq_len_list"]:
        acc = results[l].get(key, 0.0)
        row += f" | {acc:6.2f}%"
    print(row)

output_file = f"v305_debug_mqar_{MODE}_results.json"
with open(output_file, "w") as f:
    json.dump({"config": CFG, "results": results}, f, indent=2, default=str)
print(f"\n{ts()} saved: {output_file}")
