import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import math
import matplotlib.pyplot as plt
import os
import random

# Reuse DCT matrix logic
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
    # I = D^T * C * D
    # where C is padded coefficients
    return torch.matmul(D.t(), torch.matmul(coefficients, D))

def main():
    device = torch.device('cpu') # Fast enough for 1 image
    N = 28
    K = 28 # Let's use 8x8 to show compression, or 28 for perfect
    
    # 1. Load MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Pick a random image
    idx = random.randint(0, len(mnist)-1)
    target_img, label = mnist[idx]
    target_img = target_img.squeeze(0).to(device)
    
    print(f"Target Label: {label}")
    
    # 2. Setup DCT Neuron (Learnable Coefficients)
    D = get_dct_matrix(N, device=device)
    # We learn KxK coefficients and pad them to NxN
    dct_coeffs = nn.Parameter(torch.randn(K, K, device=device) * 0.01)
    
    optimizer = optim.Adam([dct_coeffs], lr=0.1)
    
    # 3. Training Loop
    epochs = 1000
    l1_lambda = 0.001 # Sparsity penalty
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Pad coefficients to 28x28
        padded_coeffs = torch.zeros(N, N, device=device)
        padded_coeffs[:K, :K] = dct_coeffs
        
        # Reconstruct image
        reconstructed = idct_2d(padded_coeffs, D)
        
        # Loss: MSE + L1 for sparsity (keeps background cleaner)
        mse_loss = F.mse_loss(reconstructed, target_img)
        l1_loss = torch.norm(dct_coeffs, 1)
        loss = mse_loss + l1_lambda * l1_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f} (MSE: {mse_loss.item():.6f})")

    # 4. Final Reconstructions
    with torch.no_grad():
        # Standard 28x28 reconstruction
        padded_coeffs_28 = torch.zeros(N, N, device=device)
        padded_coeffs_28[:K, :K] = dct_coeffs
        res_28 = torch.clamp(idct_2d(padded_coeffs_28, D), 0, 1).cpu().numpy()
        
        # High-resolution reconstruction (e.g., 112x112 -> 4x scale)
        SCALE = 4
        N_HIGH = N * SCALE
        D_HIGH = get_dct_matrix(N_HIGH, device=device)
        padded_coeffs_high = torch.zeros(N_HIGH, N_HIGH, device=device)
        # The key: We place the SAME 8x8 coefficients in a larger frequency canvas
        # We scale them by the resolution ratio to maintain intensity
        padded_coeffs_high[:K, :K] = dct_coeffs * SCALE 
        res_high = torch.clamp(idct_2d(padded_coeffs_high, D_HIGH), 0, 1).cpu().numpy()
        
        original = target_img.cpu().numpy()

    # 5. Visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f"Original MNIST (28x28)")
    
    axes[1].imshow(res_28, cmap='gray')
    axes[1].set_title(f"DCT Recon (28x28)")
    
    axes[2].imshow(res_high, cmap='gray')
    axes[2].set_title(f"High-Res Zoom ({N_HIGH}x{N_HIGH})")
    
    # Show the learned coefficients (Frequency Domain)
    coeffs_img = dct_coeffs.detach().cpu().numpy()
    im = axes[3].imshow(coeffs_img, cmap='viridis')
    axes[3].set_title(f"Learned Coeffs ({K}x{K})")
    fig.colorbar(im, ax=axes[3])
    
    plt.tight_layout()
    output_path = "results/dct_resolution_independence.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(output_path)
    print(f"Resolution independence test saved to {output_path}")

if __name__ == "__main__":
    main()
