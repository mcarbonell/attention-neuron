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

class SelectiveConv1DIIRLayerV346(nn.Module):
    """
    Capa Conv1D + Selective State Space IIR con Dimensión de Estado Escalada d_state=64 (v346).
    Resuelve la interferencia de rango entre claves almacenando las matrices outer-product en 64 slots.
    """
    def __init__(self, d_model: int, d_state: int = 64, kernel_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model
        )
        self.conv_act = nn.GELU()
        
        self.proj_gate  = nn.Linear(d_model, 1)
        self.proj_alpha = nn.Linear(d_model, 1)
        self.proj_beta  = nn.Linear(d_model, 1)
        
        self.proj_key   = nn.Linear(d_model, d_state)
        self.proj_val   = nn.Linear(d_model, d_model)
        self.proj_query = nn.Linear(d_model, d_state)
        self.proj_read_gate = nn.Linear(d_model, d_model)
        
        self.proj_out   = nn.Linear(d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Causal Conv1D
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = self.conv_act(x_conv).transpose(1, 2)
        
        # Estado matricial ampliado M_t: [batch_size, d_model, d_state=64]
        M_t = torch.zeros(batch_size, d_model, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            ctx_t = x_conv[:, t, :]
            
            g_t = torch.sigmoid(self.proj_gate(ctx_t)).unsqueeze(-1)
            decay_factor = torch.sigmoid(self.proj_alpha(ctx_t)).unsqueeze(-1)
            
            alpha_t = 1.0 - g_t * decay_factor
            beta_t  = g_t * torch.tanh(self.proj_beta(ctx_t)).unsqueeze(-1)
            
            k_t = torch.softmax(self.proj_key(ctx_t), dim=-1)
            v_t = self.proj_val(x_t)
            q_t = self.proj_query(ctx_t)
            
            outer_kv = torch.bmm(v_t.unsqueeze(2), k_t.unsqueeze(1))
            M_t = alpha_t * M_t + beta_t * outer_kv
            
            readout = torch.bmm(M_t, q_t.unsqueeze(2)).squeeze(2)
            read_gate = torch.sigmoid(self.proj_read_gate(ctx_t))
            
            y_t = self.act(self.proj_out(readout * read_gate)) + x_t
            outputs.append(y_t)
            
        return torch.stack(outputs, dim=1)

class SelectiveConv1DIIRTransformerV346(nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 128, d_state: int = 64, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model)
        
        self.layers = nn.ModuleList([
            SelectiveConv1DIIRLayerV346(d_model=d_model, d_state=d_state) for _ in range(num_layers)
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
