import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

class RGBVectorCanvas(nn.Module):
    """
    V72: Differentiable RGB Painter.
    Generates an image by composing N colored Bezier strokes using Painter's Algorithm.
    """
    def __init__(self, num_strokes=100, device='cpu'):
        super().__init__()
        self.num_strokes = num_strokes
        
        # 3 points per stroke (x, y), initialized randomly in the center
        self.points = nn.Parameter(torch.rand(num_strokes, 3, 2) * 16.0 + 8.0)
        
        # Colors (RGB) - logits to allow unconstrained optimization
        self.colors_logit = nn.Parameter(torch.randn(num_strokes, 3))
        
        # Opacity (Alpha)
        self.alpha_logit = nn.Parameter(torch.randn(num_strokes, 1))
        
        # Fixed thickness for the prototype
        self.sigma = 1.0 
        
        # CIFAR-10 is 32x32
        y, x = torch.meshgrid(torch.linspace(0, 31, 32), torch.linspace(0, 31, 32), indexing='ij')
        self.register_buffer("grid", torch.stack([x, y], dim=-1).view(1, 1024, 2).to(device))

    def forward(self):
        # 1. Sample bezier curves
        t = torch.linspace(0, 1, 15, device=self.points.device).view(1, 15, 1)
        p0 = self.points[:, 0:1, :]
        p1 = self.points[:, 1:2, :]
        p2 = self.points[:, 2:3, :]
        bezier_points = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2 # (N, 15, 2)
        
        # 2. Render stroke shapes on grid
        diff = self.grid.unsqueeze(0) - bezier_points.unsqueeze(2) # (N, 15, 1024, 2)
        dist_sq = torch.sum(diff**2, dim=-1) # (N, 15, 1024)
        min_dist_sq, _ = torch.min(dist_sq, dim=1) # (N, 1024)
        
        # 3. Gaussian stroke masks
        masks = torch.exp(-min_dist_sq / (2 * self.sigma**2)).view(self.num_strokes, 32, 32) # (N, 32, 32)
        
        # Constrain colors and alphas to [0, 1]
        colors = torch.sigmoid(self.colors_logit) # (N, 3)
        alphas = torch.sigmoid(self.alpha_logit).view(self.num_strokes, 1, 1) # (N, 1, 1)
        
        # 4. Alpha Compositing (Painter's Algorithm)
        # Start with a white background
        canvas = torch.ones(3, 32, 32, device=self.points.device)
        
        # Render layers one by one (z-index is determined by the loop order)
        for i in range(self.num_strokes):
            blend = masks[i:i+1, :, :] * alphas[i] # (1, 32, 32)
            c = colors[i].view(3, 1, 1)
            # Standard alpha blend: Canvas = Canvas * (1 - Alpha) + New_Color * Alpha
            canvas = canvas * (1 - blend) + c * blend
            
        return canvas

def train_rgb_vectorizer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V72: RGB Vectorizer AI (Differentiable Painter) ---")
    print(f"Device: {device}")
    
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    
    # Pick a random image from CIFAR-10
    idx = torch.randint(0, len(dataset), (1,)).item()
    target_img, label = dataset[idx]
    print(f"Targeting Image Index: {idx} (Label Index: {label})")
    target_img = target_img.to(device) # (3, 32, 32)
    
    model = RGBVectorCanvas(num_strokes=150, device=device).to(device)
    
    # Fast learning rate because we want the strokes to move rapidly
    optimizer = optim.Adam(model.parameters(), lr=0.1) 
    
    EPOCHS = 600
    history = []
    
    print("Painting the image... (this might take a minute)")
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        generated_img = model()
        
        # 1. Reconstruction Loss (MSE)
        loss_recon = nn.functional.mse_loss(generated_img, target_img)
        
        # 2. Sparsity Loss (L1 on opacity): Penalize unneeded strokes
        loss_sparse = 0.005 * torch.mean(torch.sigmoid(model.alpha_logit))
        
        loss = loss_recon + loss_sparse
        loss.backward()
        optimizer.step()
        
        if epoch % 60 == 0 or epoch == 1:
            history.append(generated_img.detach().cpu().permute(1, 2, 0).numpy())
            print(f"Epoch {epoch:3d} | Total Loss: {loss.item():.4f} | Recon: {loss_recon.item():.4f}")

    # --- VISUALIZATION ---
    print("\nGenerating evolution plot...")
    fig, axes = plt.subplots(2, (len(history)+1)//2, figsize=(15, 6))
    fig.suptitle("Evolution of RGB Vector Painting (150 Strokes)", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < len(history):
            ax.imshow(history[i])
            ax.set_title(f"Epoch {i*60 if i>0 else 1}")
        ax.axis('off')
        
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/v72_rgb_painting_evolution.png")
    
    # Plot final comparison
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(target_img.cpu().permute(1, 2, 0).numpy())
    ax1.set_title("Target Image (CIFAR-10)")
    ax1.axis('off')
    
    ax2.imshow(history[-1])
    ax2.set_title("Final Vector Reconstruction")
    ax2.axis('off')
    
    plt.savefig("results/figures/v72_rgb_painting_final.png")
    print("Done! Check results/figures/v72_rgb_painting_final.png")

if __name__ == "__main__":
    train_rgb_vectorizer()
