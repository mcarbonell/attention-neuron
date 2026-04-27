import torch
import matplotlib.pyplot as plt
import math
import os
import sys

# Add current directory to path to allow importing from scratch
sys.path.append(os.getcwd())

from scratch.prototype_v59_dct_attention_mnist import DCTAttentionNet, get_dct_matrix

def visualize():
    N = 28
    K = 8
    H = 512
    device = 'cpu'
    
    # Load model
    model = DCTAttentionNet(hidden_dim=H, k_size=K, device=device)
    model_path = f"models/v59_dct_k{K}_h{H}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from {model_path}")
    else:
        print("No trained weights found, showing random/initial patterns.")
    
    D = get_dct_matrix(N, device=device)
    weights = model.dct_weights.detach() # (H, K, K)
    
    fig, axes = plt.subplots(6, 6, figsize=(12, 12))
    plt.suptitle(f"LEARNED DCT Attention Neurons (K={K}x{K})", fontsize=16)
    
    for i in range(36):
        ax = axes[i//6, i%6]
        
        # Reconstruct spatial weights: W = D^T * C_padded * D
        C = weights[i]
        C_padded = torch.zeros(N, N)
        C_padded[:K, :K] = C
        
        W = torch.matmul(D.t(), torch.matmul(C_padded, D))
        
        vmax = torch.max(torch.abs(W))
        ax.imshow(W.numpy(), cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax.axis('off')
        # Show some stats about frequency usage
        energy = torch.sum(C**2)
        ax.set_title(f"N {i}\nE:{energy:.2f}", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("v59_dct_gallery.png")
    print("Gallery saved to v59_dct_gallery.png")
    # plt.show() # Disabled for headless run, but user can open png


if __name__ == "__main__":
    visualize()
