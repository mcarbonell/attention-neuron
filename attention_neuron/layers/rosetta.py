import torch
import torch.nn as nn
import math
from .base import BaseAttentionLayer

class RosettaLinear(BaseAttentionLayer):
    """
    Multi-Substrate Library Layer.
    A neuron mixes K different random substrates using learned attention weights.
    """
    def __init__(self, in_features, out_features, rank=32, num_substrates=4, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_substrates = num_substrates
        
        # K Fixed random substrates
        std = math.sqrt(2.0 / in_features)
        for k in range(num_substrates):
            self.register_buffer(f'w_init_{k}', torch.randn(out_features, in_features) * std)
        
        # Shared low-rank modulation parameters
        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.delta_in_a = nn.Parameter(torch.zeros(out_features, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_features))
        
        # Library Attention: softmax logits for mixing the K substrates per neuron
        self.library_logits = nn.Parameter(torch.zeros(out_features, num_substrates))
        
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        # 1. Calculate mixing weights per neuron
        mix_weights = torch.softmax(self.library_logits, dim=1).unsqueeze(2)
        
        # 2. Compute the mixed substrate
        w_mixed = 0
        for k in range(self.num_substrates):
            w_init_k = getattr(self, f'w_init_{k}')
            w_mixed += mix_weights[:, k, :] * w_init_k
            
        # 3. Apply low-rank modulation
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        
        w_evolved = w_mixed * (1.0 + w_m) + w_a
        
        return torch.matmul(x, w_evolved.t()) + (self.bias if self.bias is not None else 0)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, rank={}, substrates={}'.format(
            self.in_features, self.out_features, self.rank, self.num_substrates
        )
