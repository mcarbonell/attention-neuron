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
        self.coeffs = nn.ParameterList([
            nn.Parameter(torch.randn(K, K, device=device) * 0.01) for _ in range(num_classes)
        ])
        self.archetype_to_class = list(range(num_classes))

    def get_images(self):
        images = []
        for c in self.coeffs:
            full_coeffs = torch.zeros(self.N, self.N, device=c.device)
            full_coeffs[:self.K, :self.K] = c
            img = torch.matmul(self.D.t(), torch.matmul(full_coeffs, self.D))
            images.append(torch.clamp(img, 0, 1))
        return torch.stack(images)

    def spawn_archetype(self, image_tensor, target_class):
        img = image_tensor.squeeze()
        dct_full = torch.matmul(self.D, torch.matmul(img, self.D.t()))
        new_coeffs_data = dct_full[:self.K, :self.K].detach().clone()
        new_param = nn.Parameter(new_coeffs_data)
        self.coeffs.append(new_param)
        self.archetype_to_class.append(target_class)
        print(f" [SPAWN] Archetype for class {target_class}. Total: {len(self.coeffs)}")

    def forward(self, x):
        B = x.size(0)
        x_flat = x.view(B, -1)
        archetypes = self.get_images().view(len(self.coeffs), -1)
        x_norm = (x_flat**2).sum(dim=1, keepdim=True)
        a_norm = (archetypes**2).sum(dim=1, keepdim=True).t()
        interaction = torch.matmul(x_flat, archetypes.t())
        return x_norm + a_norm - 2 * interaction

def train():
    device = torch.device('cpu')
    print(f"--- V89: NEURAL-PAC (Thought-Driven Loop) ---")
    
    K_SIZE = 12
    INITIAL_LR = 0.05
    SPAWN_THRESHOLD = 0.10 # More conservative
    REVISION_THRESHOLD = 0.08 # When to start "thinking"
    MAX_INTENSITY_STEPS = 8 # More time to think before giving up
    MAX_ARCHETYPES = 250
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=1, shuffle=True)

    layer = DynamicDCTArchetypeLayer(num_classes=10, K=K_SIZE, device=device)
    current_lr = INITIAL_LR
    optimizer = optim.Adam(layer.parameters(), lr=current_lr)

    for epoch in range(1, 11):
        correct = 0
        total = 0
        
        # Anneal LR
        if epoch > 4:
            current_lr = INITIAL_LR * 0.2
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        for batch_idx, (data, target) in enumerate(train_loader):
            target_class = target.item()
            
            # --- 1. INITIAL EVALUATION (Inference) ---
            with torch.no_grad():
                distances = layer(data)
                pred_idx = torch.argmin(distances, dim=1).item()
                pred_class = layer.archetype_to_class[pred_idx]
                
                class_indices = [i for i, c in enumerate(layer.archetype_to_class) if c == target_class]
                class_distances = distances[0, class_indices]
                best_arch_idx = class_indices[torch.argmin(class_distances).item()]
                min_dist = torch.min(class_distances).item()

            # --- 2. THE THOUGHT LOOP (Intensity / Revision) ---
            # If we miss or the distance is high, "re-think" before passing on
            if pred_class != target_class or min_dist > REVISION_THRESHOLD:
                # Intensive study session
                for step in range(MAX_INTENSITY_STEPS):
                    optimizer.zero_grad()
                    dists = layer(data)
                    loss = dists[0, best_arch_idx]
                    loss.backward()
                    optimizer.step()
                    
                    # Re-evaluate
                    with torch.no_grad():
                        new_dists = layer(data)
                        min_dist = new_dists[0, best_arch_idx].item()
                        new_pred_class = layer.archetype_to_class[torch.argmin(new_dists, dim=1).item()]
                        # If we correctly classify and distance is okay, we stop thinking
                        if new_pred_class == target_class and min_dist < REVISION_THRESHOLD:
                            break
            else:
                # 3. SOFT UPDATE (Normal learning for correct samples)
                optimizer.zero_grad()
                dists = layer(data)
                loss = dists[0, best_arch_idx]
                loss.backward()
                optimizer.step()

            # --- 4. THE SPAWNING DECISION (The Final Resort) ---
            # ONLY spawn if after thinking the distance is STILL too high
            if epoch > 1 and len(layer.coeffs) < MAX_ARCHETYPES:
                if min_dist > SPAWN_THRESHOLD:
                    layer.spawn_archetype(data, target_class)
                    optimizer = optim.Adam(layer.parameters(), lr=current_lr)
            
            if pred_class == target_class: correct += 1
            total += 1
            
            if batch_idx % 10000 == 0:
                acc = (correct/total)*100 if total > 0 else 0
                print(f"Epoch {epoch} | Batch {batch_idx} | Acc: {acc:.2f}% | Archetypes: {len(layer.coeffs)} | LR: {current_lr:.4f}")

        # Visualization
        with torch.no_grad():
            imgs = layer.get_images()
            num = len(imgs)
            cols = 25
            rows = math.ceil(num / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(25, 1.0 * rows))
            for i in range(num):
                r, c = i // cols, i % cols
                ax = axes[r, c] if rows > 1 else axes[c]
                ax.imshow(imgs[i].cpu(), cmap='gray')
                ax.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(f"results/figures/v89_thought_gallery_epoch_{epoch}.png")
            plt.close()

    print("Thought-Driven Neural-PAC finished.")

if __name__ == "__main__":
    train()
