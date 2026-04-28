import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import time

class VectorCanvas(nn.Module):
    """
    V71: Generates an image by summing N parametric Bezier strokes.
    We optimize the control points and the intensity (opacity) of each stroke.
    """
    def __init__(self, num_strokes=10, device='cpu'):
        super().__init__()
        self.num_strokes = num_strokes
        
        # 3 points per stroke (x, y), initialized randomly in the center
        self.points = nn.Parameter(torch.rand(num_strokes, 3, 2) * 14.0 + 7.0)
        
        # Opacity/weight of each stroke
        self.weights = nn.Parameter(torch.ones(num_strokes, 1))
        
        # Fixed thickness
        self.sigma = 1.2
        
        y, x = torch.meshgrid(torch.linspace(0, 27, 28), torch.linspace(0, 27, 28), indexing='ij')
        self.register_buffer("grid", torch.stack([x, y], dim=-1).view(1, 784, 2).to(device))

    def forward(self):
        # 1. Sample bezier curves
        t = torch.linspace(0, 1, 15, device=self.points.device).view(1, 15, 1)
        p0 = self.points[:, 0:1, :]
        p1 = self.points[:, 1:2, :]
        p2 = self.points[:, 2:3, :]
        bezier_points = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2 # (N, 15, 2)
        
        # 2. Render strokes on grid
        diff = self.grid.unsqueeze(0) - bezier_points.unsqueeze(2) # (N, 15, 784, 2)
        dist_sq = torch.sum(diff**2, dim=-1) # (N, 15, 784)
        min_dist_sq, _ = torch.min(dist_sq, dim=1) # (N, 784)
        
        # 3. Gaussian blur to create thickness
        strokes = torch.exp(-min_dist_sq / (2 * self.sigma**2)) # (N, 784)
        
        # 4. Sum all strokes weighted by their opacity
        canvas = torch.sum(strokes * torch.relu(self.weights), dim=0) # (784,)
        
        # Clamp to [0, 1] like a real image
        return torch.clamp(canvas, 0, 1).view(28, 28)

def train_vectorizer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V71: Neural Vectorizer (SGD Image Tracing) ---")
    
    # 1. Get a single target image (a '5')
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    
    # Find a good '5'
    target_img = None
    for img, label in dataset:
        if label == 5:
            target_img = img.squeeze().to(device) # (28, 28)
            break
            
    model = VectorCanvas(num_strokes=12, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.5) # High LR because we are moving coordinates
    criterion = nn.MSELoss()
    
    EPOCHS = 300
    history = []
    
    print("Vectorizing the image...")
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        generated_img = model()
        loss = criterion(generated_img, target_img)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == 1:
            history.append(generated_img.detach().cpu().numpy())
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    # --- VISUALIZATION ---
    print("Generating evolution plot...")
    fig, axes = plt.subplots(2, len(history)//2, figsize=(15, 6))
    fig.suptitle("Evolution of SGD Vector Tracing (12 Bezier Strokes)", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < len(history):
            ax.imshow(history[i], cmap='magma')
            ax.set_title(f"Step {i*20 if i>0 else 1}")
        ax.axis('off')
        
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/v71_vector_tracing_evolution.png")
    
    # Plot final comparison
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(target_img.cpu().numpy(), cmap='magma')
    ax1.set_title("Target Pixel Image")
    ax1.axis('off')
    
    ax2.imshow(history[-1], cmap='magma')
    ax2.set_title("Final Vector Reconstruction")
    ax2.axis('off')
    
    plt.savefig("results/figures/v71_vector_tracing_final.png")
    print("Done! Check results/figures/v71_vector_tracing_evolution.png")

if __name__ == "__main__":
    train_vectorizer()
