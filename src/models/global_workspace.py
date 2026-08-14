import torch
import torch.nn as nn

class GlobalWorkspaceLayer(nn.Module):
    """
    Capa con Pizarra Global de Memoria (Idea 6: Global Workspace & Differentiable Meta-States).
    
    Cada capa procesa la representación local (x) y actualiza/lee un tensor de estado global (G)
    compartido verticalmente entre todas las capas de la red.
    """
    def __init__(self, d_model: int, num_global_slots: int = 8, nhead: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_global_slots = num_global_slots
        
        # Atención local
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Proyecciones de Lectura y Escritura en la Pizarra Global G
        self.read_proj = nn.Linear(d_model, d_model)
        self.write_proj = nn.Linear(d_model, d_model)
        
        # Capa Feed-Forward local
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, G):
        # x: [batch_size, seq_len, d_model]
        # G: [batch_size, num_global_slots, d_model]
        
        # 1. Leer de la Pizarra Global: fusionar el contexto global promediado en las representaciones locales
        g_summary = torch.mean(G, dim=1, keepdim=True)  # [batch_size, 1, d_model]
        x_read = x + torch.sigmoid(self.read_proj(x)) * g_summary
        
        # 2. Atención Local en la secuencia
        attn_out, _ = self.self_attn(x_read, x_read, x_read)
        x = self.norm1(x + attn_out)
        
        # 3. Feed-Forward local
        x = self.norm2(x + self.ffn(x))
        
        # 4. Escribir en la Pizarra Global: actualizar G verticalmente para las siguientes capas
        dG = torch.tanh(self.write_proj(torch.mean(x, dim=1, keepdim=True)))  # [batch_size, 1, d_model]
        G_next = G + dG  # Conexión residual sobre la variable de estado global
        
        return x, G_next

class GlobalWorkspaceTransformer(nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 128, num_global_slots: int = 8, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.num_global_slots = num_global_slots
        self.d_model = d_model
        
        # Variable de Estado Global Inicial Aprendible (nn.Parameter)
        self.G_init = nn.Parameter(torch.randn(1, num_global_slots, d_model) * 0.02)
        
        self.layers = nn.ModuleList([
            GlobalWorkspaceLayer(d_model=d_model, num_global_slots=num_global_slots) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.embedding(x)
        
        # Expandir el estado global inicial para el batch actual
        G = self.G_init.expand(batch_size, -1, -1)
        
        # Propagación a través de las capas transmitiendo G
        for layer in self.layers:
            out, G = layer(out, G)
            
        out = self.norm(out)
        last_token_repr = out[:, -1, :]
        logits = self.fc_out(last_token_repr)
        return logits
