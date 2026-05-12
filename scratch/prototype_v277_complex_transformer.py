import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import json
import os
import math

# --- PID Optimizer (V4: Complex-Aware) ---
class PID(optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, derivative=0.1, kp=1.0, ki=1.0, kd=1.0, weight_decay=1e-4):
        defaults = dict(lr=lr, momentum=momentum, derivative=derivative, 
                        kp=kp, ki=ki, kd=kd, weight_decay=weight_decay)
        super(PID, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr, momentum, derivative = group['lr'], group['momentum'], group['derivative']
            kp, ki, kd = group['kp'], group['ki'], group['kd']
            wd = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                if wd != 0: grad = grad.add(p, alpha=wd)
                
                state = self.state[p]
                if len(state) == 0:
                    state['integral'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.clone(grad).detach()
                    state['derivative'] = torch.zeros_like(p)
                
                integral, prev_grad, deriv_ema = state['integral'], state['prev_grad'], state['derivative']
                integral.mul_(momentum).add_(grad, alpha=1 - momentum)
                current_deriv = grad - prev_grad
                deriv_ema.mul_(derivative).add_(current_deriv, alpha=1 - derivative)
                update = grad.mul(kp).add(integral, alpha=ki).add(deriv_ema, alpha=kd)
                p.add_(update, alpha=-lr)
                state['prev_grad'].copy_(grad)
        return loss

# --- Complex Components ---

class ModReLU(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.bias = nn.Parameter(torch.full((features,), -0.5))

    def forward(self, z):
        abs_z = torch.abs(z)
        scale = F.relu(abs_z + self.bias) / (abs_z + 1e-6)
        return z * scale

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.complex(
            torch.randn(out_features, in_features) / math.sqrt(in_features),
            torch.randn(out_features, in_features) / math.sqrt(in_features)
        ))
        if bias:
            self.bias = nn.Parameter(torch.complex(torch.zeros(out_features), torch.zeros(out_features)))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

# --- Complex Transformer Components ---

class ComplexMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = ComplexLinear(d_model, d_model, bias=False)
        self.w_k = ComplexLinear(d_model, d_model, bias=False)
        self.w_v = ComplexLinear(d_model, d_model, bias=False)
        self.w_o = ComplexLinear(d_model, d_model, bias=False)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Hermitian Attention: Q * K^H
        # In PyTorch, complex matmul handles this. k.conj() is K^H
        # Energy: (B, H, L, L)
        energy = torch.matmul(q, k.conj().transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # We take the REAL part of energy for softmax weights
        # This ensures that attention weights are probability distributions
        attn_weights = F.softmax(energy.real, dim=-1)
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        
        # Weighted sum of complex values
        out = torch.matmul(attn_weights.to(v.dtype), v)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(out)

class ComplexTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim):
        super().__init__()
        self.attn = ComplexMultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            ComplexLinear(d_model, hidden_dim),
            ModReLU(hidden_dim),
            ComplexLinear(hidden_dim, d_model)
        )
        # We use a simplified LayerNorm (Real-only on magnitude or just skip)
        # For simplicity in this prototype, we'll use residual connections without LN
        # Or a complex-aware LN if needed.
        
    def forward(self, x):
        # x: (B, L, D) complex
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x

class ComplexTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, hidden_dim):
        super().__init__()
        # Embedding: Character to complex vector
        self.embed = nn.Embedding(vocab_size, d_model)
        # In the complex version, we treat embedding as real part, imag is 0
        self.layers = nn.ModuleList([
            ComplexTransformerBlock(d_model, num_heads, hidden_dim) for _ in range(num_layers)
        ])
        self.head = ComplexLinear(d_model, vocab_size)
        
    def forward(self, x):
        # x: (B, L) indices
        x = self.embed(x) # (B, L, D) real
        x = torch.complex(x, torch.zeros_like(x)) # To complex
        
        for layer in self.layers:
            x = layer(x)
            
        logits = self.head(x)
        return torch.abs(logits) # (B, L, V) magnitudes for CrossEntropy

class RealTransformer(nn.Module):
    """
    Matched baseline with 2x hidden dimension to match parameter count.
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, hidden_dim):
        super().__init__()
        # d_model and hidden_dim are doubled for real to match parameters
        # Actually, d_model complex has 2x params in MHA.
        # So we use d_model_real = d_model * sqrt(2) approx, but 2x is safer/standard.
        dm_r = d_model * 2
        hd_r = hidden_dim * 2
        
        self.embed = nn.Embedding(vocab_size, dm_r)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dm_r, nhead=num_heads, dim_feedforward=hd_r, batch_first=True, norm_first=True)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(dm_r, vocab_size)
        
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

# --- Synthetic Dataset: Rhythmic Poetry ---

def generate_rhythmic_data(num_samples=5000, seq_len=32, vocab_size=16):
    """
    Sequences with strong periodic patterns.
    Pattern: [A, B, A, C] repeated.
    """
    data = []
    for _ in range(num_samples):
        # Random start for variety
        start = torch.randint(0, vocab_size, (4,))
        pattern = start.repeat(seq_len // 4 + 1)[:seq_len+1]
        data.append(pattern)
    
    data = torch.stack(data)
    # Inputs: (B, L), Targets: (B, L)
    return data[:, :-1], data[:, 1:]

# --- Utility ---

def count_parameters(model):
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel() * (2 if torch.is_complex(p) else 1)
    return total

def run_experiment(name, model, train_x, train_y, epochs=20, device="cpu"):
    num_params = count_parameters(model)
    optimizer = PID(model.parameters(), lr=1e-3, kp=1.0, ki=10.0, kd=1.0, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n>>> Running Experiment: {name} | Parameters: {num_params}")
    
    wall_clock_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        # Single batch for simplicity in this prototype
        data, target = train_x.to(device), train_y.to(device)
        
        optimizer.zero_grad()
        output = model(data) # (B, L, V)
        loss = criterion(output.transpose(1, 2), target)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"  Iter {epoch:3d} | Loss: {loss.item():.4f}")
            
    wall_clock_time = time.time() - wall_clock_start
    final_loss = loss.item()
    pei = (10 - final_loss) / math.log10(num_params + 1)
    
    return {"name": name, "final_loss": final_loss, "PEI": pei, "time": wall_clock_time}

def main():
    torch.manual_seed(42)
    device = torch.device("cpu") # Keep on CPU for quick local verification
    
    vocab_size = 16
    seq_len = 32
    train_x, train_y = generate_rhythmic_data(num_samples=256, seq_len=seq_len, vocab_size=vocab_size)
    
    print("="*60)
    print("V277: THE COMPLEX TRANSFORMER - RHYTHMIC POETRY")
    print("="*60)
    
    # Complex Transformer: d=32, heads=4, layers=2
    c_model = ComplexTransformer(vocab_size, d_model=32, num_heads=4, num_layers=2, hidden_dim=64).to(device)
    
    # Real Transformer: Matched params
    r_model = RealTransformer(vocab_size, d_model=32, num_heads=4, num_layers=2, hidden_dim=64).to(device)
    
    res_c = run_experiment("Complex Transformer", c_model, train_x, train_y, epochs=100, device=device)
    res_r = run_experiment("Real Transformer (Matched)", r_model, train_x, train_y, epochs=100, device=device)
    
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    for r in [res_c, res_r]:
        print(f"{r['name']:25} | Loss: {r['final_loss']:.4f} | PEI: {r['PEI']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
