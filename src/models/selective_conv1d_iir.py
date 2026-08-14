import torch
import torch.nn as nn

class SelectiveConv1DIIRLayer(nn.Module):
    """
    Capa de Filtro IIR de Selección Adaptativa con Convolución Causal 1D (v344: Conv1D + Selective SSM).
    
    Arregla la ambigüedad posicional/contextual identificada en v343:
    1. Aplica CausalConv1D(kernel_size=4) para preprocesar el contexto temporal local.
    2. Calcula la compuerta de relevancia g_t = sigmoid(W_gate * \tilde{x}_t) basada en el contexto local.
    3. Alpha_t y Beta_t filtran los tokens de ruido (g_t ~ 0 -> alpha_t = 1.0, beta_t = 0.0).
    4. Actualiza la memoria matricial M_t = alpha_t * M_{t-1} + beta_t * (v_t k_t^T).
    """
    def __init__(self, d_model: int, d_state: int = 16, kernel_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.kernel_size = kernel_size
        
        # Convolución Causal 1D sobre el tiempo
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model  # Depthwise conv
        )
        self.conv_act = nn.GELU()
        
        # Proyecciones de compuerta basadas en el estado convolucionado
        self.proj_gate  = nn.Linear(d_model, 1)
        self.proj_alpha = nn.Linear(d_model, 1)
        self.proj_beta  = nn.Linear(d_model, 1)
        
        self.proj_key   = nn.Linear(d_model, d_state)
        self.proj_val   = nn.Linear(d_model, d_model)
        self.proj_query = nn.Linear(d_model, d_state)
        
        self.proj_out   = nn.Linear(d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        # 1. Causal Conv1D: [batch, d_model, seq_len]
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Recorte causal
        x_conv = self.conv_act(x_conv).transpose(1, 2) # [batch, seq_len, d_model]
        
        # Estado matricial M_t: [batch_size, d_model, d_state]
        M_t = torch.zeros(batch_size, d_model, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]            # Entrada directa
            ctx_t = x_conv[:, t, :]      # Entrada con contexto convolucional
            
            # 2. Compuerta de selección basada en el contexto convolucional
            g_t = torch.sigmoid(self.proj_gate(ctx_t)).unsqueeze(-1)  # [batch, 1, 1]
            
            decay_factor = torch.sigmoid(self.proj_alpha(ctx_t)).unsqueeze(-1)
            alpha_t = 1.0 - g_t * decay_factor
            beta_t  = g_t * torch.tanh(self.proj_beta(ctx_t)).unsqueeze(-1)
            
            k_t = torch.softmax(self.proj_key(ctx_t), dim=-1)
            v_t = self.proj_val(x_t)
            q_t = self.proj_query(ctx_t)
            
            # Producto exterior v_t (k_t)^T
            outer_kv = torch.bmm(v_t.unsqueeze(2), k_t.unsqueeze(1))
            
            # Actualización selectiva de memoria
            M_t = alpha_t * M_t + beta_t * outer_kv
            
            # Lectura del estado
            readout = torch.bmm(M_t, q_t.unsqueeze(2)).squeeze(2)
            
            y_t = self.act(self.proj_out(readout)) + x_t
            outputs.append(y_t)
            
        return torch.stack(outputs, dim=1)

class SelectiveConv1DIIRTransformer(nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 128, d_state: int = 16, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            SelectiveConv1DIIRLayer(d_model=d_model, d_state=d_state) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        out = self.embedding(x)
        for layer in self.layers:
            out = layer(out)
            
        out = self.norm(out)
        last_token_repr = out[:, -1, :]
        logits = self.fc_out(last_token_repr)
        return logits
