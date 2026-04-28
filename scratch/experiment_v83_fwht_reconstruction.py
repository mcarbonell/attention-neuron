import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import math
import matplotlib.pyplot as plt
import os
import random

# --- Manual Walsh-Hadamard Matrix (Sylvester Construction) ---
def get_walsh_matrix(N, device='cpu'):
    """
    Constructs a Walsh-Hadamard matrix of size N x N.
    N must be a power of 2.
    """
    if N == 1:
        return torch.tensor([[1.0]], device=device)
    H_prev = get_walsh_matrix(N // 2, device=device)
    top = torch.cat([H_prev, H_prev], dim=1)
    bottom = torch.cat([H_prev, -H_prev], dim=1)
    return torch.cat([top, bottom], dim=0)

def walsh_2d_matrix(image, H):
    """Transform: W = H * I * H^T (since H is symmetric)"""
    N = H.shape[0]
    return torch.matmul(H, torch.matmul(image, H))

def iwalsh_2d_matrix(coeffs, H):
    """Inverse: I = (1/N^2) * H * W * H"""
    N = H.shape[0]
    return torch.matmul(H, torch.matmul(coeffs, H)) / (N * N)

# --- Sequency-Ordered Walsh-Hadamard Matrix ---
def get_walsh_matrix_sequency(N, device='cpu'):
    """
    Constructs a Sequency-ordered (Walsh) matrix.
    Standard Hadamard (Sylvester) is reordered by number of zero crossings.
    """
    H = get_walsh_matrix(N, device=device)
    
    # Count zero crossings for each row to find "frequency" (sequency)
    # A crossing is when H[i, j] * H[i, j+1] < 0
    crossings = []
    for i in range(N):
        row = H[i]
        num_crossings = (row[:-1] * row[1:] < 0).sum().item()
        crossings.append((num_crossings, i))
    
    # Sort indices by number of crossings
    crossings.sort()
    indices = [idx for _, idx in crossings]
    
    return H[indices]

def main():
    device = torch.device('cpu')
    N_TARGET = 32
    K_SIZE = 8 # Use ALL coefficients for full reconstruction
    
    # 1. Load and Pad MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2) 
    ])
    mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    idx = random.randint(0, len(mnist)-1)
    target_img, label = mnist[idx]
    target_img = target_img.to(device).squeeze() 
    
    print(f"Target Label: {label}")
    
    # 2. Setup Sequency-Ordered Walsh Matrix
    H = get_walsh_matrix_sequency(N_TARGET, device=device)
    walsh_coeffs = nn.Parameter(torch.randn(K_SIZE, K_SIZE, device=device) * 0.1)
    
    optimizer = optim.Adam([walsh_coeffs], lr=0.1)
    
    # 3. Training Loop
    epochs = 2000
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # In this case K_SIZE = N_TARGET, but we keep the logic
        full_spectrum = torch.zeros(N_TARGET, N_TARGET, device=device)
        full_spectrum[:K_SIZE, :K_SIZE] = walsh_coeffs
        
        reconstructed = iwalsh_2d_matrix(full_spectrum, H)
        
        mse_loss = F.mse_loss(reconstructed, target_img)
        loss = mse_loss # Remove L1 for perfect reconstruction test
        
        loss.backward()
        optimizer.step()
        
        if epoch % 400 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    # 4. Final Reconstructions
    with torch.no_grad():
        full_spectrum_32 = torch.zeros(32, 32, device=device)
        full_spectrum_32[:K_SIZE, :K_SIZE] = walsh_coeffs
        res_32 = torch.clamp(iwalsh_2d_matrix(full_spectrum_32, H), 0, 1).cpu().numpy()
        
        # High Res (64x64)
        N_HIGH = 64
        H_HIGH = get_walsh_matrix_sequency(N_HIGH, device=device)
        full_spectrum_high = torch.zeros(N_HIGH, N_HIGH, device=device)
        full_spectrum_high[:K_SIZE, :K_SIZE] = walsh_coeffs * (N_HIGH / N_TARGET)
        res_high = torch.clamp(iwalsh_2d_matrix(full_spectrum_high, H_HIGH), 0, 1).cpu().numpy()
        
        original = target_img.cpu().numpy()

    # 5. Visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f"Original MNIST (32x32)")
    
    axes[1].imshow(res_32, cmap='gray')
    axes[1].set_title(f"Walsh Recon (32x32)")
    
    axes[2].imshow(res_high, cmap='gray')
    axes[2].set_title(f"Walsh High-Res (64x64)")
    
    coeffs_img = walsh_coeffs.detach().cpu().numpy()
    im = axes[3].imshow(coeffs_img, cmap='magma')
    axes[3].set_title(f"Walsh Coeffs ({K_SIZE}x{K_SIZE})")
    fig.colorbar(im, ax=axes[3])
    
    plt.tight_layout()
    output_path = "results/fwht_reconstruction_test.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(output_path)
    print(f"Sequency-Ordered Walsh reconstruction saved to {output_path}")

if __name__ == "__main__":
    main()
