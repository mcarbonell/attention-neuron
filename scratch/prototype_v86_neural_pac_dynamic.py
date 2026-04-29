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

class DynamicDCTArchetypeLayer(nn.Module):
    def __init__(self, num_classes=10, N=28, K=12, device='cpu'):
        super().__init__()
        self.N = N
        self.K = K
        self.num_classes = num_classes
        self.register_buffer('D', get_dct_matrix(N, device=device))
        
        # We store archetypes as a list of parameters to allow dynamic growth
        # Initial: 1 archetype per class
        self.coeffs = nn.ParameterList([
            nn.Parameter(torch.randn(K, K, device=device) * 0.01) for _ in range(num_classes)
        ])
        
        # Mapping: which class does each archetype index belong to?
        self.archetype_to_class = list(range(num_classes))

    def get_images(self):
        """Generates all current spatial archetypes."""
        images = []
        for c in self.coeffs:
            full_coeffs = torch.zeros(self.N, self.N, device=c.device)
            full_coeffs[:self.K, :self.K] = c
            img = torch.matmul(self.D.t(), torch.matmul(full_coeffs, self.D))
            images.append(torch.clamp(img, 0, 1))
        return torch.stack(images)

    def spawn_archetype(self, image_tensor, target_class):
        """
        Creates a new archetype initialized from a specific image.
        image_tensor: (1, N, N)
        """
        # 1. Convert image to DCT space (Forward DCT)
        # x_dct = D * image * D^T
        img = image_tensor.squeeze()
        dct_full = torch.matmul(self.D, torch.matmul(img, self.D.t()))
        
        # 2. Crop to KxK
        new_coeffs_data = dct_full[:self.K, :self.K].detach().clone()
        
        # 3. Add to our list
        new_param = nn.Parameter(new_coeffs_data)
        self.coeffs.append(new_param)
        self.archetype_to_class.append(target_class)
        
        print(f" [SPAWN] New archetype for class {target_class}. Total: {len(self.coeffs)}")

    def forward(self, x):
        """
        Returns distances to ALL current archetypes.
        x: (B, 1, N, N) -> (B, total_archetypes)
        """
        B = x.size(0)
        x_flat = x.view(B, -1)
        archetypes = self.get_images().view(len(self.coeffs), -1)
        
        # Euclidean Distances
        x_norm = (x_flat**2).sum(dim=1, keepdim=True)
        a_norm = (archetypes**2).sum(dim=1, keepdim=True).t()
        interaction = torch.matmul(x_flat, archetypes.t())
        
        return x_norm + a_norm - 2 * interaction

def train():
    device = torch.device('cpu')
    print(f"--- V86: DYNAMIC NEURAL-PAC (High Precision Run) ---")
    
    K_SIZE = 12
    INITIAL_LR = 0.05
    SPAWN_THRESHOLD = 0.12 # Slightly more sensitive for better detail
    MAX_ARCHETYPES = 200 # Increased for rich taxonomy
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=1, shuffle=True)

    layer = DynamicDCTArchetypeLayer(num_classes=10, K=K_SIZE, device=device)
    current_lr = INITIAL_LR
    optimizer = optim.Adam(layer.parameters(), lr=current_lr)

    for epoch in range(1, 11): # More epochs to refine the 200 experts
        correct = 0
        total = 0
        
        # Anneal LR to sharpen archetypes in later epochs
        if epoch > 3:
            current_lr = INITIAL_LR * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        if epoch > 6:
            current_lr = INITIAL_LR * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        for batch_idx, (data, target) in enumerate(train_loader):
            target_class = target.item()
            
            distances = layer(data) 
            pred_idx = torch.argmin(distances, dim=1).item()
            pred_class = layer.archetype_to_class[pred_idx]
            
            class_indices = [i for i, c in enumerate(layer.archetype_to_class) if c == target_class]
            class_distances = distances[0, class_indices]
            
            best_arch_in_class_idx = class_indices[torch.argmin(class_distances).item()]
            min_dist_to_class = torch.min(class_distances).item()
            
            # --- THE SPAWNING TRIGGER ---
            can_spawn = (epoch > 1) and (len(layer.coeffs) < MAX_ARCHETYPES)
            
            if can_spawn and (pred_class != target_class or min_dist_to_class > SPAWN_THRESHOLD):
                layer.spawn_archetype(data, target_class)
                optimizer = optim.Adam(layer.parameters(), lr=current_lr)
                continue 
            
            optimizer.zero_grad()
            loss = distances[0, best_arch_in_class_idx]
            loss.backward()
            optimizer.step()
            
            if pred_class == target_class: correct += 1
            total += 1
            
            if batch_idx % 10000 == 0:
                acc = (correct/total)*100 if total > 0 else 0
                print(f"Epoch {epoch} | Batch {batch_idx} | Acc: {acc:.2f}% | Archetypes: {len(layer.coeffs)} | LR: {current_lr:.4f}")

        # Visualization
        with torch.no_grad():
            imgs = layer.get_images()
            num = len(imgs)
            cols = 20 # Wider grid for more archetypes
            rows = math.ceil(num / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(20, 1.0 * rows))
            for i in range(num):
                r, c = i // cols, i % cols
                ax = axes[r, c] if rows > 1 else axes[c]
                ax.imshow(imgs[i].cpu(), cmap='gray')
                ax.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(f"results/figures/v86_dynamic_gallery_epoch_{epoch}.png")
            plt.close()

    print("High Precision Neural-PAC training finished.")

    print("Dynamic PAC training finished.")

if __name__ == "__main__":
    main_dir = os.path.dirname(os.path.abspath(__file__))
    train()
