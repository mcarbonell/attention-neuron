import torch
import torch.nn as nn

class SelectiveDynamicIIRLayer(nn.Module):
    """
    Capa de Filtro IIR de Selección Adaptativa de Señal (Idea 1 + Selective Gating).
    
    Formula Matemática de Filtrado de Ruido (SNR Maximization):
      g_t     = sigmoid(W_gate * x_t)                 (Compuerta de Relevancia de Señal [0, 1])
      alpha_t = 1.0 - g_t * sigmoid(W_alpha * x_t)   (Si g_t ~ 0 -> alpha_t = 1.0 [Preserva Memoria 100%])
      beta_t  = g_t * tanh(W_beta * x_t)              (Si g_t ~ 0 -> beta_t = 0.0 [Cero Ruido a Memoria])
      
      k_t     = softmax(W_key * x_t)                 (Vector de Clave [d_state])
      v_t     = W_val * x_t                          (Vector de Valor [d_model])
      q_t     = W_query * x_t                        (Vector de Consulta [d_state])
      
      M_t     = alpha_t * M_{t-1} + beta_t * (v_t (k_t)^T)  (Memoria de Estado Inmune al Ruido)
      y_t     = M_t * q_t + x_t                      (Lectura Asociativa con Conexión Residual)
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Compuerta Selectiva de Relevancia (Gate)
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
        
        # Estado matricial M_t: [batch_size, d_model, d_state]
        M_t = torch.zeros(batch_size, d_model, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch_size, d_model]
            
            # 1. Compuerta de selección (Detecta si el token es relevante o es ruido)
            g_t = torch.sigmoid(self.proj_gate(x_t)).unsqueeze(-1)  # [batch_size, 1, 1]
            
            # 2. Factores de decaimiento e integración selectivos
            decay_factor = torch.sigmoid(self.proj_alpha(x_t)).unsqueeze(-1)
            alpha_t = 1.0 - g_t * decay_factor                     # Si g_t=0 -> alpha_t=1.0 (Sin degradación)
            beta_t  = g_t * torch.tanh(self.proj_beta(x_t)).unsqueeze(-1) # Si g_t=0 -> beta_t=0.0 (Sin ruido)
            
            k_t = torch.softmax(self.proj_key(x_t), dim=-1)         # [batch_size, d_state]
            v_t = self.proj_val(x_t)                                # [batch_size, d_model]
            q_t = self.proj_query(x_t)                              # [batch_size, d_state]
            
            # Producto exterior v_t (k_t)^T
            outer_kv = torch.bmm(v_t.unsqueeze(2), k_t.unsqueeze(1))
            
            # Actualización selectiva (Inmune a tokens de ruido)
            M_t = alpha_t * M_t + beta_t * outer_kv
            
            # Lectura asociativa del estado
            readout = torch.bmm(M_t, q_t.unsqueeze(2)).squeeze(2)
            
            y_t = self.act(self.proj_out(readout)) + x_t
            outputs.append(y_t)
            
        return torch.stack(outputs, dim=1)

class SelectiveDynamicIIRTransformer(nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 128, d_state: int = 16, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            SelectiveDynamicIIRLayer(d_model=d_model, d_state=d_state) for _ in range(num_layers)
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
