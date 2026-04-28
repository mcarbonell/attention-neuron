import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import sys
import math
import time

# Add the parent directory to sys.path to import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention_neuron.layers.base import get_dct_matrix_1d

class DCT2DLinear(nn.Module):
    """
    A layer that synthesizes 2D spatial weights from a 2D DCT core.
    Ideal for image-based inputs like MNIST.
    """
    def __init__(self, side_len, out_features, k_size=8, bias=True):
        super().__init__()
        self.side_len = side_len # e.g., 28 for MNIST
        self.in_features = side_len * side_len
        self.out_features = out_features
        self.k_size = k_size # e.g., 8 for an 8x8 frequency core
        
        # Precompute 1D DCT basis for the side length
        self.register_buffer('D', get_dct_matrix_1d(side_len))
        
        # Learnable 2D DCT coefficients for each output neuron
        # Shape: (out_features, k_size, k_size)
        self.dct_coeffs = nn.Parameter(torch.randn(out_features, k_size, k_size) * (1.0 / k_size))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        # 1. Synthesize the 2D weights for all neurons
        # C_padded: (out_features, side_len, side_len)
        C_padded = torch.zeros(self.out_features, self.side_len, self.side_len, device=x.device)
        C_padded[:, :self.k_size, :self.k_size] = self.dct_coeffs
        
        # 2. Apply 2D DCT: W_2d = D^T @ C @ D
        # We can do this efficiently for all neurons at once using matmul
        # W_2d = D.T @ C_padded @ D
        W_2d = torch.matmul(self.D.t(), torch.matmul(C_padded, self.D))
        
        # 3. Flatten weights to (out_features, in_features)
        W = W_2d.view(self.out_features, self.in_features)
        
        return F.linear(x, W, self.bias)

class SingleLayerDCT2D(nn.Module):
    def __init__(self, k_size=12):
        super().__init__()
        # 28x28 -> 10 using a k_size x k_size DCT core
        self.layer = DCT2DLinear(side_len=28, out_features=10, k_size=k_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layer(x)

def train_and_visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Experiment V69: MNIST Single Layer 2D-DCT Visualization ---")
    print(f"Device: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 100
    LR = 0.0001
    K_SIZE = 28 # 16x16 core = 256 parameters per neuron (same as K_IN=256 in 1D)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)
    
    model = SingleLayerDCT2D(k_size=K_SIZE).to(device)
    
    actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Learnable Parameters: {actual_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = correct / 10000
        print(f"Epoch {epoch:2d} | Acc: {acc:.4f}")

    # --- VISUALIZATION ---
    print("\nSynthesizing 2D neuron templates from DCT2D coefficients...")
    model.eval()
    with torch.no_grad():
        C_padded = torch.zeros(10, 28, 28, device=device)
        C_padded[:, :K_SIZE, :K_SIZE] = model.layer.dct_coeffs
        W_2d = torch.matmul(model.layer.D.t(), torch.matmul(C_padded, model.layer.D))
        templates = W_2d.cpu().numpy()

    # Plot the 10 neuron templates
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(f"MNIST 2D-DCT Neuron Templates (Core: {K_SIZE}x{K_SIZE})")
    
    for i in range(10):
        ax = axes[i//5, i%5]
        template = templates[i]
        # Invert templates for better visualization (bright = digit features)
        ax.imshow(-template, cmap='magma')
        ax.set_title(f"Neuron {i}")
        ax.axis('off')
        
    plt.tight_layout()
    
    os.makedirs("results/figures", exist_ok=True)
    save_path = "results/figures/v69_mnist_dct2d_templates.png"
    plt.savefig(save_path)
    print(f"Visualization saved to: {save_path}")

if __name__ == "__main__":
    train_and_visualize()
