"""
scratch/prototype_v223_v2_purified_diffusion.py - Purified Spectral Diffusion (MNIST) V2

Revision:
1. SPECTRAL_SCALE = 1.0 (para no ahogar la señal).
2. EPOCHS = 40 (mas tiempo para compensar que cada experto ve menos datos).
3. Red de expertos mas profunda (3 capas densas).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import math
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import json

# --- CONFIGURACION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 28
TIMESTEPS = 200
BATCH_SIZE = 128
LR = 0.002
EPOCHS = 40 
SPECTRAL_SCALE = 1.0 

# --- 1. DCT UTILITIES ---
def get_dct_matrix(N, device='cpu'):
    dct_mat = torch.zeros(N, N, device=device)
    for i in range(N):
        for j in range(N):
            if i == 0:
                dct_mat[i, j] = math.sqrt(1/N)
            else:
                dct_mat[i, j] = math.sqrt(2/N) * math.cos((math.pi * i * (2*j + 1)) / (2*N))
    return dct_mat

D_MAT = get_dct_matrix(IMG_SIZE, device=device)

def dct_2d(image):
    return torch.matmul(D_MAT, torch.matmul(image, D_MAT.t()))

def idct_2d(coeffs):
    return torch.matmul(D_MAT.t(), torch.matmul(coeffs, D_MAT))

# --- 2. DIFFUSION ENGINE ---
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

betas = linear_beta_schedule(TIMESTEPS).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# --- 3. ARCHITECTURE ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ExpertDenoiseNet(nn.Module):
    def __init__(self, img_size=28, embed_dim=64):
        super().__init__()
        self.flat_dim = img_size * img_size
        self.net = nn.Sequential(
            nn.Linear(self.flat_dim + embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.flat_dim)
        )

    def forward(self, x, t_embed):
        x_flat = x.view(x.shape[0], -1)
        combined = torch.cat([x_flat, t_embed], dim=1)
        return self.net(combined).view(x.shape)

class PurifiedDenoiseBank(nn.Module):
    def __init__(self, n_classes=10, embed_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.experts = nn.ModuleList([ExpertDenoiseNet(embed_dim=embed_dim) for _ in range(n_classes)])

    def forward(self, x, t, labels):
        t_embed = self.time_mlp(t)
        preds = torch.zeros_like(x)
        # Optimizacion: Procesar solo las clases presentes en el batch
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = (labels == label)
            preds[mask] = self.experts[label](x[mask], t_embed[mask])
        return preds

    def sample_with_expert(self, x, t, expert_id):
        t_embed = self.time_mlp(t)
        return self.experts[expert_id](x, t_embed)

# --- 4. TRAINING LOOP ---
def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/raw", exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = PurifiedDenoiseBank().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PurifiedDenoiseBank Params: {total_params}")
    
    t0 = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device).squeeze(1)
            labels = labels.to(device)
            
            with torch.no_grad():
                x_start = dct_2d(images) * SPECTRAL_SCALE
            
            t = torch.randint(0, TIMESTEPS, (images.shape[0],), device=device).long()
            noise = torch.randn_like(x_start)
            x_noisy = q_sample(x_start, t, noise)
            
            optimizer.zero_grad()
            predicted_noise = model(x_noisy, t, labels)
            
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:<3} | Avg Loss: {avg_loss:.6f}")

    wall_clock_time = time.time() - t0
    
    # --- SAMPLING ---
    model.eval()
    print("\nGenerating Pure Archetypes...")
    final_samples = []
    with torch.no_grad():
        for target_label in range(10):
            x = torch.randn((1, IMG_SIZE, IMG_SIZE), device=device)
            for i in reversed(range(0, TIMESTEPS)):
                t = torch.full((1,), i, device=device, dtype=torch.long)
                betas_t = betas[i]
                sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[i]
                sqrt_recip_alphas_t = sqrt_recip_alphas[i]
                
                model_output = model.sample_with_expert(x, t, target_label)
                model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)
                
                if i == 0: x = model_mean
                else:
                    posterior_variance_t = posterior_variance[i]
                    noise = torch.randn_like(x)
                    x = model_mean + torch.sqrt(posterior_variance_t) * noise
            
            gen_img = idct_2d(x / SPECTRAL_SCALE)
            gen_img = torch.clamp((gen_img + 1) / 2, 0, 1)
            final_samples.append(gen_img.squeeze(0).cpu().numpy())

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(final_samples[i], cmap='gray')
        ax.set_title(f"Expert {i} (Pure)")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("results/v223_v2_purified_archetypes.png")
    print("Samples saved to results/v223_v2_purified_archetypes.png")
    
    metrics = {
        "final_objective": avg_loss,
        "wall_clock_time": wall_clock_time,
        "PEI": (max(0, 1 - math.sqrt(avg_loss))) / math.log10(total_params + 1),
        "total_params": total_params,
    }
    with open("results/raw/v223_v2_results.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
