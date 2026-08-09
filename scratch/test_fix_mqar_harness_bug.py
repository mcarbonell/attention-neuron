"""
test_fix_mqar_harness_bug.py
============================
Tests fixing the compute_kv_mask bug in MQAR harness.
"""

import math, time, os, torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

PAD_ID = 0
TOKEN_OFFSET = 1
NUM_TOKENS = 512
QUERY_MARKER = TOKEN_OFFSET + NUM_TOKENS
VOCAB_SIZE = QUERY_MARKER + 1

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

# FIXED compute_kv_mask: allow beta > 0 for all KV prefix positions (0 to kv_end)
def compute_kv_mask_FIXED(x_ids, L_padded):
    B, L = x_ids.shape
    kv_mask = torch.zeros(B, 1, L_padded, device=x_ids.device)
    for b in range(B):
        q_pos = (x_ids[b] == QUERY_MARKER).nonzero(as_tuple=False)
        kv_end = q_pos[0].item() if len(q_pos) > 0 else L
        kv_mask[b, 0, :kv_end] = 1.0  # ALL KV tokens active!
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

class ChunkwiseRealDeltaNetRectangularBlockFixed(nn.Module):
    def __init__(self, d_model, n_heads=2, d_k=32, chunk_size=64):
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
            beta = beta * compute_kv_mask_FIXED(x_ids, L_padded)
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

class SequenceModel(nn.Module):
    def __init__(self, block_cls, vocab_size, d_model=64, n_layers=4, block_kwargs=None):
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

def run_test():
    seq_len = 256
    n_pairs = 61
    batch_size = 32
    train_x, train_y = generate_mqar_dataset(50, batch_size, n_pairs, seq_len, seed=42, device=device)
    eval_x, eval_y = generate_mqar_dataset(10, batch_size, n_pairs, seq_len, seed=100, device=device)

    model = SequenceModel(ChunkwiseRealDeltaNetRectangularBlockFixed, VOCAB_SIZE, d_model=64, n_layers=4, block_kwargs={"d_k": 32}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    print("Running test with FIXED compute_kv_mask on RealRectangular at L=256...")
    for ep in range(15):
        model.train()
        for step in range(50):
            optimizer.zero_grad()
            idx = step % len(train_x)
            logits = model(train_x[idx])
            loss = criterion(logits.view(-1, VOCAB_SIZE), train_y[idx].view(-1))
            loss.backward()
            optimizer.step()
        print(f"  Epoch {ep+1:2d} -> Loss: {loss.item():.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(len(eval_x)):
            logits = model(eval_x[i])
            preds = logits.argmax(dim=-1)
            mask = (eval_y[i] != -100)
            correct += (preds[mask] == eval_y[i][mask]).sum().item()
            total += mask.sum().item()

    acc = (correct / total) * 100.0
    print(f"\nResult with FIXED mask: Acc = {acc:.2f}%")

if __name__ == "__main__":
    run_test()
