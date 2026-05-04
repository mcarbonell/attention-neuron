"""
scratch/prototype_v222_spectral_diffusion.py - Spectral Diffusion on MNIST

Experimento V222:
Validar la capacidad generativa de una red espectral mediante difusion en el dominio DCT.
La red no predice pixeles, sino coeficientes espectrales, lo que permite coherencia global
y eficiencia parametrica extrema.
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
TIMESTEPS = 200 # Reducido para prototipo rapido
BATCH_SIZE = 128
LR = 0.001
EPOCHS = 10 # Prototipo rapido, suficiente para ver convergencia de loss

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

# --- 3. ARCHITECTURE (CAN-SPECTRAL) ---
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

class SpectralDenoiseNet(nn.Module):
    def __init__(self, img_size=28, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.flat_dim = img_size * img_size
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        
        # CAN Experts: Diferentes estrategias para predecir el ruido espectral
        # Experto Lineal (Relaciones densas entre coeficientes)
        self.expert_linear = nn.Sequential(
            nn.Linear(self.flat_dim + embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.flat_dim)
        )
        
        # Experto Harmonico (Asume periodicidad en el espectro)
        self.expert_harmonic = nn.Sequential(
            nn.Linear(self.flat_dim + embed_dim, 128),
            nn.Sigmoid(), # Soft activation
            nn.Linear(128, self.flat_dim)
        )
        
        self.gate = nn.Linear(embed_dim, 2)

    def forward(self, x, t):
        t_embed = self.time_mlp(t)
        x_flat = x.view(x.shape[0], -1)
        combined = torch.cat([x_flat, t_embed], dim=1)
        
        out_lin = self.expert_linear(combined)
        out_har = self.expert_harmonic(combined)
        
        gate_weights = torch.softmax(self.gate(t_embed), dim=1)
        
        out = gate_weights[:, 0:1] * out_lin + gate_weights[:, 1:2] * out_har
        return out.view(-1, self.img_size, self.img_size)

# --- 4. TRAINING & EVALUATION ---
def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/raw", exist_ok=True)
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) # Normalizar a [-1, 1]
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SpectralDenoiseNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SpectralDenoiseNet Params: {total_params}")
    
    t0 = time.time()
    history = []
    
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0
        model.train()
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device).squeeze(1)
            
            # 1. Transformar a DCT
            with torch.no_grad():
                x_start = dct_2d(images)
            
            # 2. Samplear ruido y tiempo
            t = torch.randint(0, TIMESTEPS, (images.shape[0],), device=device).long()
            noise = torch.randn_like(x_start)
            
            # 3. Difusion Forward
            x_noisy = q_sample(x_start, t, noise)
            
            # 4. Predecir ruido
            optimizer.zero_grad()
            predicted_noise = model(x_noisy, t)
            
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i == 0:
                print(f"Epoch {epoch} | Initial Batch Loss: {loss.item():.6f}")

        avg_loss = epoch_loss / len(train_loader)
        history.append(avg_loss)
        print(f"Epoch {epoch} Finished | Avg Loss: {avg_loss:.6f}")

    wall_clock_time = time.time() - t0
    
    # --- SAMPLING (REVERSE DIFFUSION) ---
    model.eval()
    print("\nSampling new images from spectral noise...")
    with torch.no_grad():
        # Empezar con ruido puro en el dominio DCT
        x = torch.randn((16, IMG_SIZE, IMG_SIZE), device=device)
        
        for i in reversed(range(0, TIMESTEPS)):
            t = torch.full((16,), i, device=device, dtype=torch.long)
            
            # P-Sample step
            betas_t = betas[i]
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[i]
            sqrt_recip_alphas_t = sqrt_recip_alphas[i]
            
            model_output = model(x, t)
            
            model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
            )
            
            if i == 0:
                x = model_mean
            else:
                posterior_variance_t = posterior_variance[i]
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        # Transformar de vuelta a pixeles
        generated_images = idct_2d(x)
        generated_images = torch.clamp((generated_images + 1) / 2, 0, 1) # Denormalizar

    # --- SAVE RESULTS ---
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated_images[i].cpu().numpy(), cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("results/v222_spectral_diffusion_samples.png")
    print("Samples saved to results/v222_spectral_diffusion_samples.png")
    
    # Final Metrics
    final_objective = history[-1]
    # PEI Calculation (Approximate accuracy as 1 - sqrt(MSE_noise))
    accuracy_proxy = max(0, 1 - math.sqrt(final_objective))
    pei = accuracy_proxy / math.log10(total_params + 1)
    
    metrics = {
        "final_objective": final_objective,
        "total_evaluations": EPOCHS * len(train_loader),
        "wall_clock_time": wall_clock_time,
        "PEI": pei,
        "total_params": total_params,
        "hardware_info": "CPU/GPU Auto",
        "commit_hash": "v222_prototype"
    }
    
    with open("results/raw/v222_results.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("\n--- METRICS ---")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
