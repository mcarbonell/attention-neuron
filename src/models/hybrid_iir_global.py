import torch
import torch.nn as nn
from .dynamic_iir_filter import DynamicIIRLayer

class HybridIIRGlobalLayer(nn.Module):
    """
    Capa Híbrida: Filtro IIR Dinámico (Idea 1 - Horizontal O(N)) + Pizarra Global (Idea 6 - Vertical).
    """
    def __init__(self, d_model: int, num_global_slots: int = 8):
        super().__init__()
        self.iir_filter = DynamicIIRLayer(d_model=d_model)
        self.read_proj = nn.Linear(d_model, d_model)
        self.write_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, G):
        # 1. Leer contexto global
        g_summary = torch.mean(G, dim=1, keepdim=True)
        x_read = x + torch.sigmoid(self.read_proj(x)) * g_summary
        
        # 2. Filtrado IIR Dinámico Horizontal O(N)
        out = self.iir_filter(x_read)
        out = self.norm(out)
        
        # 3. Escribir actualización en la Pizarra Global G para las siguientes capas
        dG = torch.tanh(self.write_proj(torch.mean(out, dim=1, keepdim=True)))
        G_next = G + dG
        
        return out, G_next

class HybridIIRGlobalTransformer(nn.Module):
    def __init__(self, vocab_size: int = 64, d_model: int = 128, num_global_slots: int = 8, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
        # Variable Global Aprendible Inicial
        self.G_init = nn.Parameter(torch.randn(1, num_global_slots, d_model) * 0.02)
        
        self.layers = nn.ModuleList([
            HybridIIRGlobalLayer(d_model=d_model, num_global_slots=num_global_slots) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.embedding(x)
        G = self.G_init.expand(batch_size, -1, -1)
        
        for layer in self.layers:
            out, G = layer(out, G)
            
        out = self.norm(out)
        last_token_repr = out[:, -1, :]
        logits = self.fc_out(last_token_repr)
        return logits
