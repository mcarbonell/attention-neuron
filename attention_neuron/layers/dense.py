import torch
import torch.nn as nn
import math
from .base import BaseAttentionLayer

class AttentionLinear(BaseAttentionLayer):
    """
    Standard Attention Neuron Layer.
    Modulates a frozen random substrate with low-rank multiplicative and additive terms.
    """
    def __init__(self, in_features, out_features, rank=64, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Fixed base weights (Kaiming Normal) - The "Dictionary"
        std = math.sqrt(2.0 / in_features)
        self.register_buffer('w_init', torch.randn(out_features, in_features) * std)
        
        # Dual modulation (rank-r)
        # Multiplicative modulation (Gating)
        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        
        # Additive modulation (Shifting)
        self.delta_in_a = nn.Parameter(torch.zeros(out_features, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_features))
        
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        # Compute low-rank modulation matrices
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        
        # Evolve weights: W = W_init * (1 + W_m) + W_a
        w_evolved = self.w_init * (1.0 + w_m) + w_a
        
        return torch.matmul(x, w_evolved.t()) + (self.bias if self.bias is not None else 0)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, rank={}'.format(
            self.in_features, self.out_features, self.rank
        )
