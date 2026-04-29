import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import math
import matplotlib.pyplot as plt
import os

# --- DCT Basis Functions ---
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
    # coefficients: (B, K, K) or (K, K)
    # D: (N, N)
    if coefficients.dim() == 3:
        # Batch mode
        return torch.matmul(D.t(), torch.matmul(coefficients, D))
    return torch.matmul(D.t(), torch.matmul(coefficients, D))

class DCTArchetypeLayer(nn.Module):
    def __init__(self, num_archetypes, N=28, K=8, device='cpu'):
        super().__init__()
        self.num_archetypes = num_archetypes
        self.N = N
        self.K = K
        self.register_buffer('D', get_dct_matrix(N, device=device))
        
        # Learnable DCT coefficients for each archetype
        # Initialize small to let them grow into shapes
        self.coeffs = nn.Parameter(torch.randn(num_archetypes, K, K) * 0.01)
        
    def get_images(self):
        """Generates the spatial archetypes from the DCT coefficients."""
        # Pad KxK to NxN
        full_coeffs = torch.zeros(self.num_archetypes, self.N, self.N, device=self.coeffs.device)
        full_coeffs[:, :self.K, :self.K] = self.coeffs
        
        # Transform to pixel space
        images = torch.matmul(self.D.t(), torch.matmul(full_coeffs, self.D))
        return torch.clamp(images, 0, 1)

    def forward(self, x):
        """
        x: (B, 1, N, N)
        Returns: Distances to each archetype (B, num_archetypes)
        """
        B = x.size(0)
        x = x.view(B, self.N * self.N)
        
        # Generate archetypes
        archetypes = self.get_images().view(self.num_archetypes, self.N * self.N)
        
        # Compute squared Euclidean distance: ||x - a||^2 = ||x||^2 + ||a||^2 - 2<x, a>
        # (B, 1)
        x_norm = (x**2).sum(dim=1, keepdim=True)
        # (1, num_archetypes)
        a_norm = (archetypes**2).sum(dim=1, keepdim=True).t()
        # (B, num_archetypes)
        interaction = torch.matmul(x, archetypes.t())
        
        distances = x_norm + a_norm - 2 * interaction
        return distances

class NeuralPAC(nn.Module):
    def __init__(self, num_classes=10, archetypes_per_class=1, N=28, K=8, device='cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.total_archetypes = num_classes * archetypes_per_class
        
        self.archetype_layer = DCTArchetypeLayer(self.total_archetypes, N, K, device)
        
        # Map archetypes to classes (initially 1-to-1 or fixed)
        # For simplicity: archetypes 0-archetypes_per_class-1 belong to class 0, and so on.
        self.register_buffer('archetype_labels', torch.arange(num_classes).repeat_interleave(archetypes_per_class))

    def forward(self, x):
        # distances: (B, total_archetypes)
        distances = self.archetype_layer(x)
        
        # We want to minimize distance, so we use negative distance for "logits"
        # Or more simply, for each class, find the minimum distance to any of its archetypes
        logits = torch.zeros(x.size(0), self.num_classes, device=x.device)
        for c in range(self.num_classes):
            mask = (self.archetype_labels == c)
            class_distances = distances[:, mask]
            # Use negative min distance as logit (closer = higher probability)
            logits[:, c] = -torch.min(class_distances, dim=1)[0]
            
        return logits

def train():
    device = torch.device('cpu') 
    print(f"--- V85: NEURAL-PAC (Selective Positive Update) ---")
    
    # Hyperparams
    K_SIZE = 12 # Increased K for better detail
    ARCH_PER_CLASS = 1 
    LR = 0.1 # High LR for fast averaging
    EPOCHS = 5
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=1, shuffle=True) # Batch size 1 for pure selective update

    model = NeuralPAC(archetypes_per_class=ARCH_PER_CLASS, K=K_SIZE, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    plt.ion()
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # 1. Forward Pass to get distances
            # distances: (1, total_archetypes)
            distances = model.archetype_layer(data)
            
            # 2. SELECTIVE UPDATE LOGIC (The PAC Taboo Break)
            # We ONLY care about the distance to the archetype of the TRUE class
            # All other archetypes remain frozen for this sample.
            true_class = target.item()
            
            # Find which archetype in our layer corresponds to the true class
            # (In our simple model, it's index == true_class since ARCH_PER_CLASS=1)
            target_distance = distances[0, true_class]
            
            optimizer.zero_grad()
            # Loss is just the distance (MSE). Minimizing this = moving archetype towards data
            loss = target_distance 
            loss.backward()
            
            # Zero out gradients for all archetypes EXCEPT the true one
            # This ensures only the "owner" of the label learns
            with torch.no_grad():
                mask = torch.ones_like(model.archetype_layer.coeffs)
                mask[true_class] = 1.0 # Keep this one
                # Effectively, model.archetype_layer.coeffs.grad is already 0 for others
                # because they didn't participate in the target_distance graph
            
            optimizer.step()
            
            if batch_idx % 2000 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader.dataset)}] | Dist to Target: {loss.item():.4f}")

        # Visualization
        print(f"Epoch {epoch} complete. Visualizing archetypes...")
        with torch.no_grad():
            archetypes = model.archetype_layer.get_images().cpu()
            for i in range(10):
                axes[i].imshow(archetypes[i], cmap='gray')
                axes[i].axis('off')
                axes[i].set_title(f"Archetype {i}")
        
        plt.draw()
        plt.pause(0.1)
        os.makedirs("results/figures", exist_ok=True)
        plt.savefig(f"results/figures/v85_pure_archetypes_epoch_{epoch}.png")

    print("Selective training finished. Archetypes are now pure averages.")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train()
