import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseAttentionLayer, get_dct_matrix_1d, get_walsh_matrix_1d

class DCTLinear(BaseAttentionLayer):
    """
    Spectral Layer using Discrete Cosine Transform.
    Learnable coefficients in frequency domain are mapped back to spatial weights.
    """
    def __init__(self, in_features, out_features, k_in=None, k_out=None, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k_in = min(k_in, in_features) if k_in else in_features
        self.k_out = min(k_out, out_features) if k_out else out_features
        
        self.register_buffer('D_in', get_dct_matrix_1d(in_features))
        self.register_buffer('D_out', get_dct_matrix_1d(out_features))
        
        # Learnable DCT coefficients
        self.dct_coeffs = nn.Parameter(torch.randn(self.k_out, self.k_in) * (1.0 / math.sqrt(self.k_in)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        # Synthesize weights only from the relevant slices of the spectral bases
        # W = D_out[:, :k_out]^T @ dct_coeffs @ D_in[:k_in, :]
        W = torch.matmul(self.D_out[:self.k_out, :].t(), 
                         torch.matmul(self.dct_coeffs, self.D_in[:self.k_in, :]))
        return F.linear(x, W, self.bias)

class WalshLinear(BaseAttentionLayer):
    """
    Spectral Layer using Walsh-Hadamard Transform.
    N must be a power of 2.
    """
    def __init__(self, in_features, out_features, k_in=None, k_out=None, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k_in = min(k_in, in_features) if k_in else in_features
        self.k_out = min(k_out, out_features) if k_out else out_features
        
        self.register_buffer('W_in', get_walsh_matrix_1d(in_features))
        self.register_buffer('W_out', get_walsh_matrix_1d(out_features))
        
        # Learnable Walsh coefficients
        self.walsh_coeffs = nn.Parameter(torch.randn(self.k_out, self.k_in) * (1.0 / math.sqrt(self.k_in)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        # Synthesize weights only from the relevant slices of the spectral bases
        # W = W_out[:, :k_out]^T @ walsh_coeffs @ W_in[:k_in, :]
        W = torch.matmul(self.W_out[:self.k_out, :].t(), 
                         torch.matmul(self.walsh_coeffs, self.W_in[:self.k_in, :]))
        return F.linear(x, W, self.bias)
