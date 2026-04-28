import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

class VectorCanvas(nn.Module):
    """
    V73: Vectorizes an archetype (mean) image.
    Uses 15 Bezier strokes.
    """
    def __init__(self, num_strokes=15, device='cpu'):
        super().__init__()
        self.num_strokes = num_strokes
        
        # 3 points per stroke (x, y)
        self.points = nn.Parameter(torch.rand(num_strokes, 3, 2) * 14.0 + 7.0)
        self.weights = nn.Parameter(torch.ones(num_strokes, 1))
        self.sigma = 1.2
        
        y, x = torch.meshgrid(torch.linspace(0, 27, 28), torch.linspace(0, 27, 28), indexing='ij')
        self.register_buffer("grid", torch.stack([x, y], dim=-1).view(1, 784, 2).to(device))

    def forward(self):
        t = torch.linspace(0, 1, 15, device=self.points.device).view(1, 15, 1)
        p0 = self.points[:, 0:1, :]
        p1 = self.points[:, 1:2, :]
        p2 = self.points[:, 2:3, :]
        bezier_points = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2 # (N, 15, 2)
        
        diff = self.grid.unsqueeze(0) - bezier_points.unsqueeze(2) # (N, 15, 784, 2)
        dist_sq = torch.sum(diff**2, dim=-1) # (N, 15, 784)
        min_dist_sq, _ = torch.min(dist_sq, dim=1) # (N, 784)
        
        strokes = torch.exp(-min_dist_sq / (2 * self.sigma**2)) # (N, 784)
        canvas = torch.sum(strokes * torch.relu(self.weights), dim=0) # (784,)
        
        return torch.clamp(canvas, 0, 1).view(28, 28)

def train_all_archetypes_vectorizer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V73: Neural Vectorizer (All MNIST Archetypes) ---")
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    
    # 1. Calculate all Archetypes (Mean Images)
    print("Calculating mean images for all digits 0-9...")
    archetypes = []
    for digit in range(10):
        images_of_digit = [img.squeeze() for img, label in dataset if label == digit]
        archetypes.append(torch.stack(images_of_digit).mean(dim=0).to(device))
        
    final_vectors = []
    
    # 2. Vectorize each archetype
    EPOCHS = 200 # Slightly fewer epochs since we're doing 10 in a row
    
    for digit in range(10):
        print(f"Vectorizing Archetype '{digit}'...", end='', flush=True)
        target_img = archetypes[digit]
        
        model = VectorCanvas(num_strokes=15, device=device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.5) 
        criterion = nn.MSELoss()
        
        for epoch in range(1, EPOCHS + 1):
            optimizer.zero_grad()
            generated_img = model()
            loss = criterion(generated_img, target_img)
            loss.backward()
            optimizer.step()
            
        final_vectors.append(generated_img.detach().cpu().numpy())
        print(f" Loss: {loss.item():.4f}")

    # --- VISUALIZATION ---
    print("\nGenerating comprehensive grid plot...")
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    fig.suptitle("MNIST Archetypes (Top) vs Differentiable Vectorization (Bottom)", fontsize=18)
    
    for i in range(10):
        axes[0, i].imshow(archetypes[i].cpu().numpy(), cmap='magma')
        axes[0, i].set_title(f"Mean '{i}'")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(final_vectors[i], cmap='magma')
        axes[1, i].set_title(f"Vector '{i}'")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/v73_all_archetypes.png")
    print("Done! Check results/figures/v73_all_archetypes.png")

if __name__ == "__main__":
    train_all_archetypes_vectorizer()
