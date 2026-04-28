import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Add the parent directory to sys.path to import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention_neuron import DCTLinear, WalshLinear

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = (x * x).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class HybridSpectralBlock(nn.Module):
    """Transformer block using DCT for Attention and Walsh for FFN."""
    def __init__(self, dim, n_heads, hidden_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Attention using DCT (Smooth semantic routing)
        self.wq = DCTLinear(dim, dim, k_in=32, k_out=32, bias=False)
        self.wk = DCTLinear(dim, dim, k_in=32, k_out=32, bias=False)
        self.wv = DCTLinear(dim, dim, k_in=32, k_out=32, bias=False)
        self.wo = DCTLinear(dim, dim, k_in=32, k_out=32, bias=False)
        
        # FFN using Walsh (Sharp logical routing)
        self.w1 = WalshLinear(dim, hidden_dim, k_in=32, k_out=64, bias=False)
        self.w2 = WalshLinear(hidden_dim, dim, k_in=64, k_out=32, bias=False)
        self.w3 = WalshLinear(dim, hidden_dim, k_in=32, k_out=64, bias=False)
        
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x):
        # Attention
        h = self.norm1(x)
        bsz, seqlen, dim = h.shape
        xq, xk, xv = self.wq(h), self.wk(h), self.wv(h)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        x = x + self.wo(attn)
        
        # FFN
        h = self.norm2(x)
        ffn = self.w2(F.silu(self.w1(h)) * self.w3(h))
        x = x + ffn
        return x

class SpectralGPT(nn.Module):
    def __init__(self, vocab_size, dim=128, n_layers=2, n_heads=4, hidden_dim=512):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([HybridSpectralBlock(dim, n_heads, hidden_dim) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.tok_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Attention Neuron: Spectral GPT Demo ---")
    
    VOCAB_SIZE = 1000
    SEQ_LEN = 32
    BATCH_SIZE = 4
    
    model = SpectralGPT(vocab_size=VOCAB_SIZE).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
    
    # Toy training step
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    print("Forward pass successful!")

if __name__ == "__main__":
    main()
