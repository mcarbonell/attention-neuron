import torch
import torch.nn as nn
import torch.nn.functional as F
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

class VectorizedSelectiveIIRBlock(nn.Module):
    """
    Bloque Vectorizado IIR de Escaneo Paralelo (v348: Mamba/H3 Style).
    Combina CausalConv1D + Vectorized Log-Cumsum Scan + MLP FFN residual.
    """
    def __init__(self, d_model: int = 256, kernel_size: int = 4, ffn_expansion: int = 2):
        super().__init__()
        self.d_model = d_model
        
        # 1. Causal Conv1D
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model
        )
        self.conv_act = nn.GELU()
        
        # 2. Proyecciones para el escaneo IIR
        self.proj_gate  = nn.Linear(d_model, d_model)
        self.proj_alpha = nn.Linear(d_model, d_model)
        self.proj_beta  = nn.Linear(d_model, d_model)
        self.proj_out   = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        
        # 3. Módulo MLP Feed-Forward (SwiGLU style)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_gate = nn.Linear(d_model, d_model * ffn_expansion)
        self.ffn_up   = nn.Linear(d_model, d_model * ffn_expansion)
        self.ffn_down = nn.Linear(d_model * ffn_expansion, d_model)
        
        self.act = nn.GELU()

    def forward(self, x):
        # --- SUB-BLOQUE 1: MEZCLADOR DE SECUENCIA VECTORIZADO IIR ---
        norm_x = self.norm1(x)
        batch_size, seq_len, d_model = norm_x.shape
        
        # Causal Conv1D
        x_conv = norm_x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        ctx = self.conv_act(x_conv).transpose(1, 2)
        
        # Compuertas vectorizadas
        g_t = torch.sigmoid(self.proj_gate(ctx))
        decay_factor = torch.sigmoid(self.proj_alpha(ctx))
        
        alpha_t = 1.0 - g_t * decay_factor
        beta_t  = g_t * torch.tanh(self.proj_beta(ctx)) * norm_x
        
        # Escaneo Paralelo Log-Cumsum
        log_alpha = torch.log(torch.clamp(alpha_t, min=1e-5, max=1.0 - 1e-5))
        cum_log_alpha = torch.cumsum(log_alpha, dim=1)
        
        Lambda = torch.exp(cum_log_alpha)
        scaled_input = beta_t / (Lambda + 1e-6)
        cum_scaled_input = torch.cumsum(scaled_input, dim=1)
        
        H_t = Lambda * cum_scaled_input
        iir_out = self.act(self.proj_out(H_t))
        
        # Conexión residual 1
        x = x + iir_out
        
        # --- SUB-BLOQUE 2: FEED-FORWARD NETWORK (FFN) ---
        norm_x2 = self.norm2(x)
        ffn_out = self.ffn_down(F.silu(self.ffn_gate(norm_x2)) * self.ffn_up(norm_x2))
        
        # Conexión residual 2
        x = x + ffn_out
        return x

class VectorizedGatedIIRTransformerV348(nn.Module):
    """
    Arquitectura de Capacidad Completa v348: 6 Capas Vectorizadas IIR + FFN + RoPE/Sinusoidal.
    Diseñada para alcanzar convergencia máxima en la tarea MQAR.
    """
    def __init__(self, vocab_size: int = 64, d_model: int = 256, num_layers: int = 6, kernel_size: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model)
        
        self.layers = nn.ModuleList([
            VectorizedSelectiveIIRBlock(d_model=d_model, kernel_size=kernel_size) for _ in range(num_layers)
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
