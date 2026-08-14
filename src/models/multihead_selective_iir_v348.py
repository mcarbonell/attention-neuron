import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadSelectiveConv1DIIRLayerV348(nn.Module):
    """
    Capa Multi-Head Selective Conv1D IIR (v348: MHS-IIR).
    Dividir la memoria de estado en H cabezas independientes elimina la interferencia entre pares clave-valor.
    """
    def __init__(self, d_model: int = 256, num_heads: int = 4, d_state: int = 32, kernel_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_state = d_state
        
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model
        )
        self.conv_act = nn.GELU()
        
        self.proj_gate  = nn.Linear(d_model, num_heads)
        self.proj_alpha = nn.Linear(d_model, num_heads)
        self.proj_beta  = nn.Linear(d_model, num_heads)
        
        self.proj_key   = nn.Linear(d_model, num_heads * d_state)
        self.proj_val   = nn.Linear(d_model, d_model)
        self.proj_query = nn.Linear(d_model, num_heads * d_state)
        self.proj_read_gate = nn.Linear(d_model, d_model)
        
        self.proj_out   = nn.Linear(d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # 1. Causal Conv1D
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        ctx = self.conv_act(x_conv).transpose(1, 2)
        
        # 2. Factores de compuerta por cabeza [batch, seq_len, num_heads]
        g_t = torch.sigmoid(self.proj_gate(ctx))
        decay_factor = torch.sigmoid(self.proj_alpha(ctx))
        
        alpha_t = 1.0 - g_t * decay_factor                     # [batch, seq_len, H]
        beta_t  = g_t * torch.tanh(self.proj_beta(ctx))       # [batch, seq_len, H]
        
        # Proyecciones K, V, Q por cabeza
        k_t = torch.softmax(self.proj_key(ctx).view(batch_size, seq_len, self.num_heads, self.d_state), dim=-1)
        v_t = self.proj_val(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q_t = self.proj_query(ctx).view(batch_size, seq_len, self.num_heads, self.d_state)
        
        # Estado matricial por cabeza: M_t [batch, H, head_dim, d_state]
        M_t = torch.zeros(batch_size, self.num_heads, self.head_dim, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            a_t = alpha_t[:, t, :].unsqueeze(-1).unsqueeze(-1)  # [batch, H, 1, 1]
            b_t = beta_t[:, t, :].unsqueeze(-1).unsqueeze(-1)   # [batch, H, 1, 1]
            
            vt = v_t[:, t, :, :]  # [batch, H, head_dim]
            kt = k_t[:, t, :, :]  # [batch, H, d_state]
            qt = q_t[:, t, :, :]  # [batch, H, d_state]
            
            # Producto exterior v_t (k_t)^T por cabeza
            outer_kv = torch.matmul(vt.unsqueeze(-1), kt.unsqueeze(-2)) # [batch, H, head_dim, d_state]
            
            M_t = a_t * M_t + b_t * outer_kv
            
            readout = torch.matmul(M_t, qt.unsqueeze(-1)).squeeze(-1)   # [batch, H, head_dim]
            outputs.append(readout.view(batch_size, d_model))
            
        out_stack = torch.stack(outputs, dim=1)
        read_gate = torch.sigmoid(self.proj_read_gate(ctx))
        y_t = self.act(self.proj_out(out_stack * read_gate)) + x
        return y_t

class MultiHeadSelectiveConv1DIIRTransformerV348(nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 256, num_heads: int = 4, d_state: int = 32, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model)
        
        self.layers = nn.ModuleList([
            MultiHeadSelectiveConv1DIIRLayerV348(d_model=d_model, num_heads=num_heads, d_state=d_state) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        out = self.pos_encoder(self.embedding(x))
        for layer in self.layers:
            out = layer(out)
            
        out = self.norm(out)
        last_token_repr = out[:, -1, :]
        logits = self.fc_out(last_token_repr)
        return logits
