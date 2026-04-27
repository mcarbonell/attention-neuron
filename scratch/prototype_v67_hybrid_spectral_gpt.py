import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import os
import numpy as np

# =============================================================================
# 1. DCT BASIS (Smooth, Analog, Semantic)
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
    """Used for Attention: Smooth routing of semantic concepts."""
    def __init__(self, in_features, out_features, k_in, k_out, bias=False, device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k_in = min(k_in, in_features)
        self.k_out = min(k_out, out_features)
        
        self.register_buffer('D_in', get_dct_matrix_1d(in_features, device=device))
        self.register_buffer('D_out', get_dct_matrix_1d(out_features, device=device))
        
        self.dct_coeffs = nn.Parameter(torch.randn(self.k_out, self.k_in) * (1.0 / math.sqrt(self.k_in)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        C_padded = torch.zeros(self.out_features, self.in_features, device=x.device)
        C_padded[:self.k_out, :self.k_in] = self.dct_coeffs
        W = torch.matmul(self.D_out.t(), torch.matmul(C_padded, self.D_in))
        return F.linear(x, W, self.bias)

# =============================================================================
# 2. WALSH BASIS (Sharp, Digital, Logical)
# =============================================================================
def get_walsh_matrix_1d(N, device='cpu'):
    """Generates a 1D Walsh-Hadamard matrix of size N x N. N must be a power of 2."""
    # Start with H_1
    H = torch.tensor([[1.0]], device=device)
    # Sylvester's construction
    while H.shape[0] < N:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(N) # Orthogonal normalize

class WalshLinear(nn.Module):
    """Used for FFN: Sharp, rule-based binary logic routing."""
    def __init__(self, in_features, out_features, k_in, k_out, bias=False, device='cpu'):
        super().__init__()
        # Ensure dimensions are powers of 2 for pure Walsh
        assert (in_features & (in_features - 1)) == 0, "Walsh requires power of 2 dims"
        assert (out_features & (out_features - 1)) == 0, "Walsh requires power of 2 dims"
        
        self.in_features = in_features
        self.out_features = out_features
        self.k_in = min(k_in, in_features)
        self.k_out = min(k_out, out_features)
        
        self.register_buffer('W_in', get_walsh_matrix_1d(in_features, device=device))
        self.register_buffer('W_out', get_walsh_matrix_1d(out_features, device=device))
        
        # The learnable core
        self.walsh_coeffs = nn.Parameter(torch.randn(self.k_out, self.k_in) * (1.0 / math.sqrt(self.k_in)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        C_padded = torch.zeros(self.out_features, self.in_features, device=x.device)
        C_padded[:self.k_out, :self.k_in] = self.walsh_coeffs
        
        # W = W_out^T @ C @ W_in (Walsh is symmetric, so W^T == W, but we keep standard form)
        W = torch.matmul(self.W_out.t(), torch.matmul(C_padded, self.W_in))
        return F.linear(x, W, self.bias)

# =============================================================================
# 3. HYBRID SPECTRAL ARCHITECTURE (V67)
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = (x * x).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class WalshFeedForward(nn.Module):
    """FFN using Walsh matrices for sharp logical reasoning."""
    def __init__(self, dim, hidden_dim, k_dim, k_hidden):
        super().__init__()
        self.w1 = WalshLinear(dim, hidden_dim, k_in=k_dim, k_out=k_hidden, bias=False)
        self.w2 = WalshLinear(hidden_dim, dim, k_in=k_hidden, k_out=k_dim, bias=False)
        self.w3 = WalshLinear(dim, hidden_dim, k_in=k_dim, k_out=k_hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class DCTAttention(nn.Module):
    """Attention using DCT matrices for smooth semantic context."""
    def __init__(self, dim, n_heads, k_dim_attn):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.wq = DCTLinear(dim, dim, k_in=k_dim_attn, k_out=k_dim_attn, bias=False)
        self.wk = DCTLinear(dim, dim, k_in=k_dim_attn, k_out=k_dim_attn, bias=False)
        self.wv = DCTLinear(dim, dim, k_in=k_dim_attn, k_out=k_dim_attn, bias=False)
        self.wo = DCTLinear(dim, dim, k_in=k_dim_attn, k_out=k_dim_attn, bias=False)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.wo(output)

class HybridSpectralTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, k_dim_ffn, k_hidden_ffn, k_dim_attn):
        super().__init__()
        self.attention = DCTAttention(dim, n_heads, k_dim_attn)
        self.feed_forward = WalshFeedForward(dim, hidden_dim, k_dim_ffn, k_hidden_ffn)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class HybridSpectralGPT(nn.Module):
    def __init__(self, vocab_size, dim=128, n_layers=4, n_heads=4, hidden_dim=512, 
                 k_dim_ffn=32, k_hidden_ffn=64, k_dim_attn=32):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.pos_embeddings = nn.Parameter(torch.zeros(1, 1024, dim))
        
        self.layers = nn.ModuleList([
            HybridSpectralTransformerBlock(dim, n_heads, hidden_dim, k_dim_ffn, k_hidden_ffn, k_dim_attn) 
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
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

def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_len]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V67: HYBRID SPECTRAL GPT (DCT Attention + Walsh FFN) ---")
    print(f"Device: {device}")
    
    data_path = r"C:\Users\mrcm_\Local\proj\tiny-thinker\data\train_v1.bin"
    if not os.path.exists(data_path):
        data_path = r"C:\Users\mrcm_\Local\proj\tiny-thinker\data\synthetic_logic.bin"
        
    print(f"Loading data from {data_path}...")
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    train_data = data[:5_000_000]
    val_data = data[5_000_000:5_100_000]
    
    VOCAB_SIZE = 16384 
    DIM = 128
    HIDDEN_DIM = 512
    N_LAYERS = 4
    N_HEADS = 4
    SEQ_LEN = 256
    BATCH_SIZE = 64
    
    # Compress 128x512 to 32x64 (32x)
    K_DIM_FFN = 32      
    K_HIDDEN_FFN = 64   
    # Compress 128x128 to 32x32 (16x)
    K_DIM_ATTN = 32
    
    model = HybridSpectralGPT(
        vocab_size=VOCAB_SIZE, dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS, hidden_dim=HIDDEN_DIM,
        k_dim_ffn=K_DIM_FFN, k_hidden_ffn=K_HIDDEN_FFN, k_dim_attn=K_DIM_ATTN
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Hybrid Learnable Params: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
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
