"""
scratch/prototype_v204_resonant_firewall.py — Resonant Firewall (Noise Robustness)

Experimento de frontera (V204):
Basándonos en la teoría biológica del "Firewall Natural":
Las neuronas biológicas solo se comunican si sus frecuencias de oscilación resuenan.
El ruido aleatorio (fases desincronizadas) produce interferencia destructiva y se cancela,
mientras que la señal coherente sobrevive.

Este experimento compara la robustez ante ruido extremo entre:
1. Una MLP Clásica (Densas, Suma Ponderada)
2. La Red de Resonancia de Fase (V203)

Ambas con el mismo número de neuronas (784 -> 128 -> 10).
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

class ResonantMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.res1 = FastResonantLayer(784, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.res2 = FastResonantLayer(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x_phase = x * math.pi
        x = self.res1(x_phase)
        x = self.bn1(x)
        return self.res2(x)

class ClassicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)

# --- Experimento ---

def evaluate_with_noise(model, loader, device, noise_std):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            # Inyectar Ruido Gaussiano
            if noise_std > 0:
                noise = torch.randn_like(data) * noise_std
                data = data + noise
            
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(loader.dataset)

def run_firewall_experiment():
    print("🛡️ Iniciando Experimento V204: El Firewall Biológico (Resonancia vs Ruido)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)
    
    models = {
        "Classic_MLP": ClassicMLP().to(device),
        "Resonant_Net": ResonantMNIST().to(device)
    }
    
    epochs = 3
    results = {}
    
    for name, model in models.items():
        print(f"\n🎬 Entrenando {name}...")
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(1, epochs + 1):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Fast Feedback
                if epoch == 1 and batch_idx < 3:
                    print(f"   [{name}] Epoch 1 - Batch {batch_idx} | Loss: {loss.item():.4f}")
                    
        acc = evaluate_with_noise(model, test_loader, device, 0.0)
        print(f"✅ {name} entrenado. Acc (Sin Ruido): {acc:.4f}")
    
    # --- Test de Resistencia al Ruido (Firewall) ---
    print("\n🌪️ Iniciando Test de Robustez (Gaussian Noise)...")
    noise_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    print("\nResultados de Precisión:")
    print(f"{'Noise (std)':<12} | {'Classic MLP':<12} | {'Resonant Net':<12}")
    print("-" * 42)
    
    for std in noise_levels:
        acc_mlp = evaluate_with_noise(models["Classic_MLP"], test_loader, device, std)
        acc_res = evaluate_with_noise(models["Resonant_Net"], test_loader, device, std)
        print(f"{std:<12.1f} | {acc_mlp:<12.4f} | {acc_res:<12.4f}")
        
    print("\n✅ Experimento V204 Completado.")

if __name__ == "__main__":
    run_firewall_experiment()
