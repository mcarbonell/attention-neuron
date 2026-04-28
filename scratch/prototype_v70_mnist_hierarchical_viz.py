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
    """Synthesizes 2D weights from a DCT core."""
    def __init__(self, side_len, out_features, k_size=8, bias=True):
        super().__init__()
        self.side_len = side_len
        self.in_features = side_len * side_len
        self.out_features = out_features
        self.k_size = k_size
        self.register_buffer('D', get_dct_matrix_1d(side_len))
        self.dct_coeffs = nn.Parameter(torch.randn(out_features, k_size, k_size) * (1.0 / k_size))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        C_padded = torch.zeros(self.out_features, self.side_len, self.side_len, device=x.device)
        C_padded[:, :self.k_size, :self.k_size] = self.dct_coeffs
        W_2d = torch.matmul(self.D.t(), torch.matmul(C_padded, self.D))
        W = W_2d.view(self.out_features, self.in_features)
        return F.linear(x, W, self.bias)

class HierarchicalNet(nn.Module):
    """
    V70: Two-layer network for hierarchical visualization.
    Layer 1: 784 -> 20 (DCT-2D bases)
    Layer 2: 20 -> 10 (Linear combination)
    """
    def __init__(self, k_size=12):
        super().__init__()
        self.layer1 = DCT2DLinear(28, 20, k_size=k_size, bias=False)
        self.layer2 = nn.Linear(20, 10) # The "Mixer"

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = torch.relu(self.layer1(x))
        return self.layer2(h)

def train_and_visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Experiment V70: Hierarchical MNIST Visualization ---")
    
    BATCH_SIZE = 128
    EPOCHS = 15
    LR = 0.001
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)
    
    model = HierarchicalNet().to(device)
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
        print(f"Epoch {epoch:2d} | Acc: {correct/10000:.4f}")

    # --- VISUALIZATION ---
    print("\nExtracting hierarchical templates...")
    model.eval()
    with torch.no_grad():
        # 1. Get Layer 1 Hidden Bases (20, 28, 28)
        C1 = torch.zeros(20, 28, 28, device=device)
        C1[:, :model.layer1.k_size, :model.layer1.k_size] = model.layer1.dct_coeffs
        W1_2d = torch.matmul(model.layer1.D.t(), torch.matmul(C1, model.layer1.D))
        bases = W1_2d.cpu().numpy() # (20, 28, 28)
        
        # 2. Get Layer 2 Mixing Weights (10, 20)
        W2 = model.layer2.weight.cpu().numpy()
        
        # 3. Calculate Effective Output Templates (10, 28, 28)
        # Each output neuron is a linear combination of hidden bases
        # W_final = W2 @ W1_flattened
        W1_flat = W1_2d.view(20, 784).cpu().numpy()
        W_final_flat = W2 @ W1_flat
        output_templates = W_final_flat.reshape(10, 28, 28)

    # Plotting
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("V70: Hierarchical Representation Learning", fontsize=16)
    
    # Plot 20 Hidden Bases
    gs1 = fig.add_gridspec(2, 10, left=0.05, right=0.95, top=0.9, bottom=0.6)
    for i in range(20):
        ax = fig.add_subplot(gs1[i//10, i%10])
        ax.imshow(-bases[i], cmap='magma')
        ax.axis('off')
        if i == 0: ax.set_title("Layer 1 Bases", loc='left', color='blue')

    # Plot 10 Final Combined Templates
    gs2 = fig.add_gridspec(2, 5, left=0.1, right=0.9, top=0.5, bottom=0.1)
    for i in range(10):
        ax = fig.add_subplot(gs2[i//5, i%5])
        ax.imshow(-output_templates[i], cmap='magma')
        ax.axis('off')
        ax.set_title(f"Final Digit {i}")
        if i == 0: ax.set_title("Final Combined Templates", loc='left', color='red')

    os.makedirs("results/figures", exist_ok=True)
    save_path = "results/figures/v70_mnist_hierarchical_templates.png"
    plt.savefig(save_path)
    print(f"Visualization saved to: {save_path}")
    
    # Analysis: Which bases form digit 8?
    digit_idx = 8
    importance = np.abs(W2[digit_idx])
    top_3 = importance.argsort()[-3:][::-1]
    print(f"\nTop 3 bases for Digit {digit_idx}: {top_3} with weights {W2[digit_idx, top_3]}")

if __name__ == "__main__":
    import numpy as np
    train_and_visualize()
