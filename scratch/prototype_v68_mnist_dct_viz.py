import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import sys
import time

# Add the parent directory to sys.path to import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention_neuron import DCTLinear

class SingleLayerDCT(nn.Module):
    """
    Ultra-simple MNIST classifier: 
    A single DCT-synthesized layer (784 -> 10).
    """
    def __init__(self, k_in=128):
        super().__init__()
        # 784 -> 10. We use a k_in core to control how many frequencies we learn.
        # k_out is 10 because we only have 10 output neurons.
        self.layer = DCTLinear(784, 10, k_in=k_in, k_out=10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layer(x)

def train_and_visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Experiment V68: MNIST Single Layer DCT Visualization ---")
    print(f"Device: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 20
    LR = 0.001
    K_IN = 256 # Use more frequencies for sharper visualization
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)
    
    model = SingleLayerDCT(k_in=K_IN).to(device)
    
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
    print("\nSynthesizing neuron templates from DCT coefficients...")
    model.eval()
    with torch.no_grad():
        # Reconstruct the full weight matrix W (10, 784)
        # Using the same logic as inside DCTLinear.forward
        C_padded = torch.zeros(model.layer.out_features, model.layer.in_features, device=device)
        C_padded[:model.layer.k_out, :model.layer.k_in] = model.layer.dct_coeffs
        
        # W = D_out.T @ C @ D_in
        W = torch.matmul(model.layer.D_out.t(), torch.matmul(C_padded, model.layer.D_in))
        
        # W has shape (10, 784)
        templates = W.cpu().numpy()

    # Plot the 10 neuron templates
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(f"MNIST Neuron Templates (Synthesized from {K_IN} DCT Coeffs)")
    
    for i in range(10):
        ax = axes[i//5, i%5]
        template = templates[i].reshape(28, 28)
        im = ax.imshow(template, cmap='viridis')
        ax.set_title(f"Neuron {i}")
        ax.axis('off')
        
    plt.tight_layout()
    
    # Save the result
    os.makedirs("results/figures", exist_ok=True)
    save_path = "results/figures/v68_mnist_dct_templates.png"
    plt.savefig(save_path)
    print(f"Visualization saved to: {save_path}")
    
    # Show it if possible (in a real environment this might not pop up, but we save it)
    # plt.show()

if __name__ == "__main__":
    train_and_visualize()
