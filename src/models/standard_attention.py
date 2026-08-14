import torch
import torch.nn as nn

class StandardAttentionTransformer(nn.Module):
    """
    Modelo Baseline: Transformer Estándar usando torch.nn.MultiheadAttention.
    Complejidad O(N^2) en tiempo y memoria respecto a la longitud de la secuencia.
    """
    def __init__(self, vocab_size: int = 64, d_model: int = 128, nhead: int = 4, num_layers: int = 2, max_len: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: [batch_size, seq_len]
        batch_size, seq_len = x.shape
        out = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        out = self.transformer(out)
        
        # Predicción basada en la representación del último token (Query)
        last_token_repr = out[:, -1, :]
        logits = self.fc_out(last_token_repr)
        return logits
