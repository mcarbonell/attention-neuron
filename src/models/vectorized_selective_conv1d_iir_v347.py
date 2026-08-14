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

class VectorizedSelectiveConv1DIIRLayerV347(nn.Module):
    """
    Capa Conv1D + Selective IIR Vectorizada sin bucles 'for' en Python (v347).
    
    Utiliza el escaneo paralelo de PyTorch (torch.cumsum en el dominio logarítmico)
    para acelerar el forward pass 10x-30x.
    """
    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.d_model = d_model
        
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model
        )
        self.conv_act = nn.GELU()
        
        self.proj_gate  = nn.Linear(d_model, d_model)
        self.proj_alpha = nn.Linear(d_model, d_model)
        self.proj_beta  = nn.Linear(d_model, d_model)
        
        self.proj_out   = nn.Linear(d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        # 1. Causal Conv1D paralela
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        ctx = self.conv_act(x_conv).transpose(1, 2) # [batch, seq_len, d_model]
        
        # 2. Factores de decaimiento y compuerta vectorizados
        g_t = torch.sigmoid(self.proj_gate(ctx))                  # [batch, seq_len, d_model]
        decay_factor = torch.sigmoid(self.proj_alpha(ctx))         # [batch, seq_len, d_model]
        
        # alpha_t en el dominio de decaimiento continuo
        alpha_t = 1.0 - g_t * decay_factor
        beta_t  = g_t * torch.tanh(self.proj_beta(ctx)) * x
        
        # 3. Escaneo Paralelo IIR mediante torch.cumsum en espacio logarítmico
        log_alpha = torch.log(torch.clamp(alpha_t, min=1e-5, max=1.0 - 1e-5))
        cum_log_alpha = torch.cumsum(log_alpha, dim=1)            # [batch, seq_len, d_model]
        
        # Factor de escala acumulado L_t = exp(cum_log_alpha)
        Lambda = torch.exp(cum_log_alpha)
        
        # Entrada escalada: \tilde{V} = beta_t / (Lambda + 1e-6)
        scaled_input = beta_t / (Lambda + 1e-6)
        
        # Suma acumulada paralela de entradas escaladas
        cum_scaled_input = torch.cumsum(scaled_input, dim=1)      # [batch, seq_len, d_model]
        
        # Re-escalado final del estado continuo H_t = Lambda * cum_scaled_input
        H_t = Lambda * cum_scaled_input
        
        y_t = self.act(self.proj_out(H_t)) + x
        return y_t

class VectorizedSelectiveConv1DIIRTransformerV347(nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 128, num_layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model)
        
        self.layers = nn.ModuleList([
            VectorizedSelectiveConv1DIIRLayerV347(d_model=d_model) for _ in range(num_layers)
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
