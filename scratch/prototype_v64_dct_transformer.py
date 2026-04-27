import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import os
import numpy as np

# =============================================================================
# DCT COMPRESSION KERNEL
# =============================================================================
def get_dct_matrix_1d(N, device='cpu'):
    dct_mat = torch.zeros(N, N, device=device)
    for i in range(N):
        for j in range(N):
            if i == 0:
                dct_mat[i, j] = math.sqrt(1/N)
            else:
                dct_mat[i, j] = math.sqrt(2/N) * math.cos((math.pi * i * (2*j + 1)) / (2*N))
    return dct_mat

class DCTLinear(nn.Module):
    """
    A Linear layer synthesized from a tiny DCT core.
    """
    def __init__(self, in_features, out_features, k_in, k_out, bias=False, device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k_in = min(k_in, in_features)
        self.k_out = min(k_out, out_features)
        
        self.register_buffer('D_in', get_dct_matrix_1d(in_features, device=device))
        self.register_buffer('D_out', get_dct_matrix_1d(out_features, device=device))
        
        # The learnable core
        self.dct_coeffs = nn.Parameter(torch.randn(self.k_out, self.k_in) * (1.0 / math.sqrt(self.k_in)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        # We can optimize this by doing the matmul in a way that broadcasts over batch & seq
        C_padded = torch.zeros(self.out_features, self.in_features, device=x.device)
        C_padded[:self.k_out, :self.k_in] = self.dct_coeffs
        
        W = torch.matmul(self.D_out.t(), torch.matmul(C_padded, self.D_in))
        
        return F.linear(x, W, self.bias)

# =============================================================================
# DCT-TRANSFORMER ARCHITECTURE
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = (x * x).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class DCTFeedForward(nn.Module):
    """
    SwiGLU FeedForward but completely compressed using DCTLinear.
    """
    def __init__(self, dim, hidden_dim, k_dim, k_hidden):
        super().__init__()
        # In a standard LLM: w1 and w3 go from dim -> hidden_dim
        # w2 goes from hidden_dim -> dim
        
        # We replace them with DCT Linear layers!
        self.w1 = DCTLinear(dim, hidden_dim, k_in=k_dim, k_out=k_hidden, bias=False)
        self.w2 = DCTLinear(hidden_dim, dim, k_in=k_hidden, k_out=k_dim, bias=False)
        self.w3 = DCTLinear(dim, hidden_dim, k_in=k_dim, k_out=k_hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
    """Standard multi-head causal attention for baseline comparison."""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.wo(output)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, k_dim, k_hidden):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = DCTFeedForward(dim, hidden_dim, k_dim, k_hidden)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class DCTTransformer(nn.Module):
    def __init__(self, vocab_size, dim=128, n_layers=4, n_heads=4, hidden_dim=512, k_dim=32, k_hidden=64):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        # Absolute positional embeddings (simple alternative to RoPE for this PoC)
        self.pos_embeddings = nn.Parameter(torch.zeros(1, 1024, dim))
        
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, hidden_dim, k_dim, k_hidden) 
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights
        self.tok_embeddings.weight = self.output.weight

    def forward(self, tokens, targets=None):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens) + self.pos_embeddings[:, :seqlen, :]
        
        for layer in self.layers:
            h = layer(h)
            
        logits = self.output(self.norm(h))
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits

# =============================================================================
# DATA LOADER
# =============================================================================
def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_len]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# =============================================================================
# TRAINING LOOP
# =============================================================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V64: DCT-TRANSFORMER ---")
    print(f"Device: {device}")
    
    # Load dataset from tiny-thinker
    # Let's try to load a chunk of synthetic_logic.bin
    data_path = r"C:\Users\mrcm_\Local\proj\tiny-thinker\data\train_v1.bin"
    if not os.path.exists(data_path):
        data_path = r"C:\Users\mrcm_\Local\proj\tiny-thinker\data\synthetic_logic.bin"
        
    print(f"Loading data from {data_path}...")
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    
    # Use just a 5M token slice to be fast
    train_data = data[:5_000_000]
    val_data = data[5_000_000:5_100_000]
    print(f"Loaded {len(train_data)} training tokens.")
    
    # Hyperparameters
    VOCAB_SIZE = 16384 # Tiny-thinker's tokenizer size
    DIM = 128
    HIDDEN_DIM = 512
    N_LAYERS = 4
    N_HEADS = 4
    SEQ_LEN = 256
    BATCH_SIZE = 64
    
    # DCT Compression settings
    # Normal FFN: dim -> hidden_dim (128 -> 512)
    # DCT Core: k_dim -> k_hidden
    K_DIM = 32      # Out of 128
    K_HIDDEN = 64   # Out of 512
    
    model = DCTTransformer(
        vocab_size=VOCAB_SIZE, 
        dim=DIM, 
        n_layers=N_LAYERS, 
        n_heads=N_HEADS, 
        hidden_dim=HIDDEN_DIM,
        k_dim=K_DIM,
        k_hidden=K_HIDDEN
    ).to(device)
    
    # Let's calculate compression
    dense_ffn_params = 4 * (128*512 + 512*128 + 128*512) # 4 layers, 3 weights per FFN
    dct_ffn_params = 4 * (K_DIM*K_HIDDEN * 3)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Learnable Params: {total_params:,}")
    print(f"Dense FFN Params would be: {dense_ffn_params:,}")
    print(f"DCT FFN Params are: {dct_ffn_params:,}")
    print(f"FFN Compression Ratio: {dense_ffn_params/dct_ffn_params:.1f}x")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    # Quick Training Loop
    ITERATIONS = 500
    t0 = time.time()
    
    model.train()
    for i in range(ITERATIONS):
        x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, device)
        
        logits, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if i % 50 == 0 or i == ITERATIONS - 1:
            # Eval
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_data, BATCH_SIZE, SEQ_LEN, device)
                _, val_loss = model(xv, yv)
            model.train()
            
            dt = time.time() - t0
            print(f"Iter {i:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Time: {dt:.1f}s")
            t0 = time.time()

if __name__ == "__main__":
    train()
