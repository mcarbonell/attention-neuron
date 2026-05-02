"""
scratch/prototype_v206_dct_resonant.py — DCT Resonant Firewall

Experimento de frontera (V206):
Fusión de la Neurona de Resonancia con Compresión Espectral (DCT).
En lugar de sintonizar los 784 píxeles (dominio espacial), proyectamos la
imagen al dominio de frecuencias espaciales mediante una base 2D-DCT FIJA.
Nos quedamos solo con los K=64 coeficientes principales (baja frecuencia).
Luego, la capa de Resonancia opera sobre estas 64 frecuencias.

Resultado: Una compresión paramétrica extrema (>10x) sin perder
las propiedades de resonancia y firewall.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import math
import json

# --- Utilidad DCT Fija ---

def create_2d_dct_basis(H=28, W=28, k_out=64):
    """
    Genera una matriz de proyección DCT-II 2D fija de tamaño (H*W, k_out).
    Recortada a las frecuencias más bajas (esquina superior izquierda).
    """
    basis = torch.zeros(k_out, H, W)
    k_side = int(math.sqrt(k_out)) # ej: 8x8 = 64
    
    for u in range(k_side):
        for v in range(k_side):
            idx = u * k_side + v
            cu = 1 / math.sqrt(2) if u == 0 else 1.0
            cv = 1 / math.sqrt(2) if v == 0 else 1.0
            
            for x in range(H):
                for y in range(W):
                    val = cu * cv * math.cos((2*x+1)*u*math.pi/(2*H)) * math.cos((2*y+1)*v*math.pi/(2*W))
                    basis[idx, x, y] = val * (2 / math.sqrt(H * W))
                    
    return basis.view(k_out, H * W).t() # Shape: (784, 64)

# --- Capas ---

class FastResonantLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.phase_sintonizer = nn.Parameter(torch.rand(out_features, in_features) * 2 * math.pi)
        self.magnitude = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))

    def forward(self, x_phase):
        x_cos = torch.cos(x_phase)
        x_sin = torch.sin(x_phase)
        w_cos = torch.cos(self.phase_sintonizer) * self.magnitude
        w_sin = torch.sin(self.phase_sintonizer) * self.magnitude
        return F.relu(F.linear(x_cos, w_cos) + F.linear(x_sin, w_sin))

# --- Modelos ---

class SpectralResonantMNIST(nn.Module):
    def __init__(self, phase_scale=math.pi/4, k_features=64):
        super().__init__()
        self.phase_scale = phase_scale
        
        # Matriz de proyección DCT (Fija, sin parámetros de aprendizaje)
        self.register_buffer('dct_basis', create_2d_dct_basis(28, 28, k_features))
        
        # Reducción drástica: De 784 a K entradas
        self.res1 = FastResonantLayer(k_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.res2 = FastResonantLayer(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        # 1. Proyección Espectral (784 -> 64)
        x_dct = torch.matmul(x, self.dct_basis)
        
        # 2. Mapeo a Fase (con el Fix de Robustez V205)
        x_phase = x_dct * self.phase_scale
        
        # 3. Resonancia
        x = self.res1(x_phase)
        x = self.bn1(x)
        return self.res2(x)

# --- Experimento ---

def evaluate_with_noise(model, loader, device, noise_std):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            if noise_std > 0:
                noise = torch.randn_like(data) * noise_std
                data = data + noise
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(loader.dataset)

def run_spectral_experiment():
    print("🛸 Iniciando Experimento V206: Resonancia Espectral (DCT + Fase)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)
    
    model = SpectralResonantMNIST().to(device)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros Entrenables: {params} (¡Reducción extrema!)")
    
    epochs = 4
    train_noise_std = 0.5 
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    print("\n🎬 Entrenando (Con Noise Augmentation)...")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Ruido para forzar la robustez del Firewall
            noise = torch.randn_like(data) * train_noise_std
            data_noisy = data + noise
            
            optimizer.zero_grad()
            output = model(data_noisy)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        acc = evaluate_with_noise(model, test_loader, device, 0.0)
        print(f"   Época {epoch} | Acc Test: {acc:.4f}")
        
    print("\n🌪️ Test de Robustez (Gaussian Noise)...")
    noise_levels = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    print(f"\n{'Noise (std)':<12} | {'DCT Resonant':<12}")
    print("-" * 28)
    
    for std in noise_levels:
        acc = evaluate_with_noise(model, test_loader, device, std)
        print(f"{std:<12.1f} | {acc:<12.4f}")
        
    print("\n✅ V206 Completado. Espectro y Resonancia unificados.")

if __name__ == "__main__":
    run_spectral_experiment()
