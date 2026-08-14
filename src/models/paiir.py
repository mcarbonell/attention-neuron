"""
PAIIR: Parallel Adaptive Infinite Impulse Response Architecture
================================================================
Official implementation of the PAIIR (Parallel Adaptive Infinite Impulse Response) model.

PAIIR combines:
1. Causal Conv1D local temporal context (k=4) for noise-signal discrimination.
2. Adaptive Signal Gating g_t in (0, 1) and continuous-time exponential decay alpha_t.
3. Parallel Logarithmic Cumsum Scan (torch.cumsum in log-space) achieving O(N) training speed
   without Python loops, with O(1) memory per token during inference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for token sequence alignment."""
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class PAIIRLayer(nn.Module):
    """
    PAIIR (Parallel Adaptive Infinite Impulse Response) Layer.
    
    Computes continuous linear recurrence in parallel across sequence length L:
        h_t = alpha_t * h_{t-1} + beta_t * x_t
    using parallel log-space cumsum scan without sequential for-loops.
    """
    def __init__(self, d_model: int = 128, kernel_size: int = 4):
        super().__init__()
        self.d_model = d_model
        
        # 1. Local Causal Conv1D (k=4) for temporal context extraction
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model
        )
        self.conv_act = nn.GELU()
        
        # 2. Adaptive Gate & Decay Factor Projections
        self.proj_gate  = nn.Linear(d_model, d_model)
        self.proj_alpha = nn.Linear(d_model, d_model)
        self.proj_beta  = nn.Linear(d_model, d_model)
        self.proj_out   = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PAIIRLayer.
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        norm_x = self.norm(x)
        
        # Step 1: Local Causal Conv1D
        x_conv = norm_x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        ctx = self.conv_act(x_conv).transpose(1, 2)
        
        # Step 2: Adaptive Gating Factors
        g_t = torch.sigmoid(self.proj_gate(ctx))
        decay_factor = torch.sigmoid(self.proj_alpha(ctx))
        
        alpha_t = 1.0 - g_t * decay_factor
        beta_t  = g_t * torch.tanh(self.proj_beta(ctx)) * norm_x
        
        # Step 3: Parallel Logarithmic Cumsum Scan
        log_alpha = torch.log(torch.clamp(alpha_t, min=1e-5, max=1.0 - 1e-5))
        cum_log_alpha = torch.cumsum(log_alpha, dim=1)
        
        Lambda = torch.exp(cum_log_alpha)
        scaled_input = beta_t / (Lambda + 1e-6)
        cum_scaled_input = torch.cumsum(scaled_input, dim=1)
        
        H_t = Lambda * cum_scaled_input
        out = self.act(self.proj_out(H_t))
        
        # Residual Connection
        return x + out

    def step(self, x_t: torch.Tensor, h_prev: torch.Tensor, conv_buffer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        O(1) Step Inference Mode for token-by-token generation without KV-Cache.
        Args:
            x_t: Current token representation [batch_size, 1, d_model]
            h_prev: Previous IIR hidden state [batch_size, d_model]
            conv_buffer: Temporal buffer for Conv1D [batch_size, kernel_size, d_model]
        Returns:
            (out_t, h_next, conv_buffer_next)
        """
        norm_x = self.norm(x_t)
        # Update conv buffer
        conv_buffer_next = torch.cat([conv_buffer[:, 1:, :], norm_x], dim=1)
        
        # Conv1D step
        ctx = self.conv_act(self.conv1d(conv_buffer_next.transpose(1, 2))[:, :, -1:]).transpose(1, 2)
        
        g_t = torch.sigmoid(self.proj_gate(ctx))
        decay_factor = torch.sigmoid(self.proj_alpha(ctx))
        
        alpha_t = (1.0 - g_t * decay_factor).squeeze(1)
        beta_t  = (g_t * torch.tanh(self.proj_beta(ctx)) * norm_x).squeeze(1)
        
        h_next = alpha_t * h_prev + beta_t
        out_t = x_t + self.act(self.proj_out(h_next.unsqueeze(1)))
        return out_t, h_next, conv_buffer_next

class PAIIRModel(nn.Module):
    """
    PAIIR Model: Full Stacked Architecture with Positional Encoding and Classification/LM Head.
    """
    def __init__(self, vocab_size: int = 64, d_model: int = 128, num_layers: int = 4, kernel_size: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model)
        
        self.layers = nn.ModuleList([
            PAIIRLayer(d_model=d_model, kernel_size=kernel_size) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pos_encoder(self.embedding(x))
        for layer in self.layers:
            out = layer(out)
            
        out = self.norm(out)
        last_token = out[:, -1, :]
        logits = self.fc_out(last_token)
        return logits
