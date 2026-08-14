import torch
import torch.nn as nn

class DynamicIIRLayer(nn.Module):
    """
    Capa de Filtro IIR No Lineal Adaptativo con Estado Matricial Asociativo (Idea 1 + SSM Matrix Memory).
    
    Ecuación de Memoria de Estado en Tiempo Continuo / Discreto:
      alpha_t = sigmoid(W_alpha * x_t)         (Factor de decaimiento dinámico [0, 1])
      k_t     = softmax(W_key * x_t)           (Vector de clave proyectado [d_state])
      v_t     = W_val * x_t                    (Vector de valor proyectado [d_model])
      q_t     = W_query * x_t                  (Vector de consulta [d_state])
      
      M_t     = alpha_t * M_{t-1} + v_t (k_t)^T (Actualización de memoria por producto exterior)
      y_t     = M_t * q_t                      (Lectura asociativa M_t * q_t)
      
    Complejidad: O(N) en longitud de secuencia, O(d_model * d_state) en memoria de estado.
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.proj_alpha = nn.Linear(d_model, 1)
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
            
            alpha_t = torch.sigmoid(self.proj_alpha(x_t)).unsqueeze(-1) # [batch_size, 1, 1]
            k_t = torch.softmax(self.proj_key(x_t), dim=-1)             # [batch_size, d_state]
            v_t = self.proj_val(x_t)                                    # [batch_size, d_model]
            q_t = self.proj_query(x_t)                                  # [batch_size, d_state]
            
            # Producto exterior v_t (k_t)^T -> [batch_size, d_model, d_state]
            outer_kv = torch.bmm(v_t.unsqueeze(2), k_t.unsqueeze(1))
            
            # Actualización del estado IIR matricial
            M_t = alpha_t * M_t + outer_kv
            
            # Lectura del estado M_t mediante la consulta q_t -> [batch_size, d_model]
            readout = torch.bmm(M_t, q_t.unsqueeze(2)).squeeze(2)
            
            y_t = self.act(self.proj_out(readout)) + x_t
            outputs.append(y_t)
            
        return torch.stack(outputs, dim=1)

class DynamicIIRTransformer(nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 128, d_state: int = 16, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            DynamicIIRLayer(d_model=d_model, d_state=d_state) for _ in range(num_layers)
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
