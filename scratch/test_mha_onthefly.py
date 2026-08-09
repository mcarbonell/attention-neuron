"""
scratch/test_mha_onthefly.py
============================
Testing MHA with ON-THE-FLY MQAR batch generation (infinite random data) to prevent overfitting.
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

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
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

class CausalMHABlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.conv = ShortCausalConv1D(d_model, kernel_size=4)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
    def forward(self, x):
        x = self.conv(x)
        res = x; norm_x = self.norm1(x)
        L = x.shape[1]
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x, attn_mask=causal_mask, is_causal=False)
        x = res + attn_out
        return x + self.ffn(self.norm2(x))

class MHAModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, max_len=4096, d_model=128, n_layers=2, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = AbsolutePositionalEmbedding(max_len, d_model)
        self.layers = nn.ModuleList([CausalMHABlock(d_model=d_model, n_heads=n_heads) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        h = self.emb(x) + self.pe(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)

def test_mha_onthefly(seq_len, n_pairs, lr=3e-3, target_steps=1000):
    batch_size = 32
    eval_x, eval_y = generate_mqar_dataset(10, batch_size, n_pairs, seq_len, seed=999, device=device)
    
    model = MHAModel(vocab_size=VOCAB_SIZE, max_len=4096, d_model=128, n_layers=2, n_heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    start_t = time.time()
    print(f"\n--- Testing MHA ON-THE-FLY at L={seq_len} (n_pairs={n_pairs}) ---")
    
    for step in range(1, target_steps + 1):
        model.train()
        optimizer.zero_grad()
        # GENERATE FRESH BATCH ON-THE-FLY!
        x, y = generate_mqar_batch(batch_size, n_pairs, seq_len, NUM_TOKENS, device)
        logits = model(x)
        loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 50 == 0:
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
            print(f"  Step {step:4d} | Train Loss: {loss.item():.4f} | Eval Acc: {acc:6.2f}%")
            if acc >= 99.9:
                print(f"  ==> PERFECT AT STEP {step} ({time.time()-start_t:.2f}s) <==")
                return acc, step
    return acc, target_steps

if __name__ == "__main__":
    test_mha_onthefly(256, 61)
