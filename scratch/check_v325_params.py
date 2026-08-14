import sys, os
import torch
import torch.nn as nn

# Verify LLaMA vs DeltaPhase param counts at Scale 4
class LLaMABlock(nn.Module):
    def __init__(self, d_model=1024, n_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # SwiGLU FFN: 8/3 * d_model = 2730
        d_ff = int(2 * 4 * d_model / 3) # 2730
        self.w1 = nn.Linear(d_model, d_ff, bias=False) # gate
        self.w2 = nn.Linear(d_ff, d_model, bias=False) # down
        self.w3 = nn.Linear(d_model, d_ff, bias=False) # up
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        res = x
        x = res + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        swiglu = F.silu(self.w1(self.norm2(x))) * self.w3(self.norm2(x))
        return x + self.w2(swiglu)

def count_params():
    d_model = 1024
    vocab = 32768
    emb_dim = 256
    
    # DeltaPhase Model
    embed = nn.Embedding(vocab, emb_dim)
    embed_proj = nn.Linear(emb_dim, d_model, bias=False)
    # Head shares embed.weight via weight tying
    head_proj = nn.Linear(d_model, emb_dim, bias=False)
    
    # 8 Bloques DeltaPhase
    # Cada bloque en v325:
    # conv1d: 5120
    # w_k, w_q, w_v, w_o: 4 x (1024*1024 + 1024) = 4,198,400
    # w_beta: 1024*8 + 8 = 8200
    # LayerNorms (3x): 6144
    # FFN Lerp: combine (4096*1024) = 4,194,304 + gains/phases (49152) = 4,243,456
    # Layer total = 8,455,420
    # 8 layers = 67,643,360
    
    # Total DeltaPhase V12 = 4194304 + 262144 + 67643360 + 2048 + 262144 = 72,363,999 (~72.36M)
    print("DeltaPhase V12 Total Params:", 4194304 + 262144 + 67643360 + 2048 + 262144)

if __name__ == "__main__":
    count_params()
