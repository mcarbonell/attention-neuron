import torch
import torch.nn as nn
import math

def get_dct_matrix_1d(N, device='cpu'):
    """Generates a 1D Discrete Cosine Transform matrix of size N x N."""
    dct_mat = torch.zeros(N, N, device=device)
    for i in range(N):
        for j in range(N):
            if i == 0:
                dct_mat[i, j] = math.sqrt(1/N)
            else:
                dct_mat[i, j] = math.sqrt(2/N) * math.cos((math.pi * i * (2*j + 1)) / (2*N))
    return dct_mat

def get_walsh_matrix_1d(N, device='cpu'):
    """Generates a 1D Walsh-Hadamard matrix of size N x N. N must be a power of 2."""
    # Ensure dimensions are powers of 2 for pure Walsh
    if (N & (N - 1)) != 0:
        raise ValueError("Walsh requires power of 2 dimensions")
        
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < N:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(N) # Orthogonal normalize

class BaseAttentionLayer(nn.Module):
    """Base class for Attention-modulated layers."""
    def __init__(self):
        super().__init__()

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
