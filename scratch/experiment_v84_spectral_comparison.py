import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import math
import matplotlib.pyplot as plt
import os
import random

# --- DCT Logic ---
def get_dct_matrix(N, device='cpu'):
    dct_mat = torch.zeros(N, N, device=device)
    for i in range(N):
        for j in range(N):
            if i == 0:
                dct_mat[i, j] = math.sqrt(1/N)
            else:
                dct_mat[i, j] = math.sqrt(2/N) * math.cos((math.pi * i * (2*j + 1)) / (2*N))
    return dct_mat

def idct_2d(coefficients, D):
    return torch.matmul(D.t(), torch.matmul(coefficients, D))

# --- Walsh-Hadamard Logic ---
def get_hadamard_matrix(N, device='cpu'):
    if N == 1:
        return torch.tensor([[1.0]], device=device)
    H_prev = get_hadamard_matrix(N // 2, device=device)
    top = torch.cat([H_prev, H_prev], dim=1)
    bottom = torch.cat([H_prev, -H_prev], dim=1)
    return torch.cat([top, bottom], dim=0)

def get_walsh_matrix_sequency(N, device='cpu'):
    H = get_hadamard_matrix(N, device=device)
    crossings = []
    for i in range(N):
        row = H[i]
        num_crossings = (row[:-1] * row[1:] < 0).sum().item()
        crossings.append((num_crossings, i))
    crossings.sort()
    indices = [idx for _, idx in crossings]
    return H[indices]

def iwalsh_2d(coeffs, H):
    N = H.shape[0]
    return torch.matmul(H, torch.matmul(coeffs, H)) / (N * N)

def train_reconstruction(target, basis_type, K, N=32, epochs=1000):
    device = target.device
    coeffs = nn.Parameter(torch.randn(K, K, device=device) * 0.01)
    optimizer = optim.Adam([coeffs], lr=0.1)
    
    if basis_type == 'DCT':
        D = get_dct_matrix(N, device=device)
    else:
        H = get_walsh_matrix_sequency(N, device=device)
        
    for _ in range(epochs):
        optimizer.zero_grad()
        full_spec = torch.zeros(N, N, device=device)
        full_spec[:K, :K] = coeffs
        
        if basis_type == 'DCT':
            recon = idct_2d(full_spec, D)
        else:
            recon = iwalsh_2d(full_spec, H)
            
        loss = F.mse_loss(recon, target)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        full_spec = torch.zeros(N, N, device=device)
        full_spec[:K, :K] = coeffs
        if basis_type == 'DCT':
            final = torch.clamp(idct_2d(full_spec, D), 0, 1)
        else:
            final = torch.clamp(iwalsh_2d(full_spec, H), 0, 1)
    return final.cpu().numpy()

def main():
    device = torch.device('cpu')
    N = 32
    # Load and pad MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])
    mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)
    idx = random.randint(0, len(mnist)-1)
    img, label = mnist[idx]
    img = img.to(device).squeeze()
    
    k_values = [4, 8, 16, 32]
    
    fig, axes = plt.subplots(3, len(k_values) + 1, figsize=(18, 10))
    
    # Row 0: Original image in the first col
    axes[0, 0].imshow(img.cpu().numpy(), cmap='gray')
    axes[0, 0].set_title(f"Original (Label {label})")
    for i in range(1, len(k_values)+1): axes[0, i].axis('off')
    
    print(f"Training comparisons for Label {label}...")
    
    for i, K in enumerate(k_values):
        print(f"  > Processing K={K}...")
        # Row 1: DCT
        dct_recon = train_reconstruction(img, 'DCT', K, N)
        axes[1, i+1].imshow(dct_recon, cmap='gray')
        axes[1, i+1].set_title(f"DCT K={K} ({K*K} params)")
        
        # Row 2: Walsh
        walsh_recon = train_reconstruction(img, 'Walsh', K, N)
        axes[2, i+1].imshow(walsh_recon, cmap='gray')
        axes[2, i+1].set_title(f"Walsh K={K} ({K*K} params)")
        
    axes[1, 0].text(0.5, 0.5, 'DCT\n(Smooth)', fontsize=14, ha='center', va='center')
    axes[1, 0].axis('off')
    axes[2, 0].text(0.5, 0.5, 'Walsh\n(Blocky)', fontsize=14, ha='center', va='center')
    axes[2, 0].axis('off')
    
    plt.tight_layout()
    out_path = "results/spectral_comparison_grid.png"
    plt.savefig(out_path)
    print(f"Grid saved to {out_path}")

if __name__ == "__main__":
    main()
