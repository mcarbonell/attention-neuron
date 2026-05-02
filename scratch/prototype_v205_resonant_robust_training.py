"""
scratch/prototype_v205_resonant_robust_training.py — Resonant Firewall (Fix)

Experimento de frontera (V205):
Análisis del colapso del V204.
Error Conceptual Encontrado: "Phase Jitter" (Jitter de Fase).
Al mapear la intensidad del píxel directamente a la fase con un multiplicador de PI
(x_phase = x * math.pi), un ruido aditivo de 1.0 se convierte en un salto de 180º (PI).
Dentro de una función periódica como el coseno, desfasar 180º invierte la señal 
completamente (destrucción total). La MLP suma el ruido linealmente, por lo que 
el ruido de media 0 se cancela solo. En la resonancia, E[cos(x + ruido)] -> 0.

Solución:
1. Reducir la sensibilidad de fase (Phase Scaling) para que el ruido no dé "vueltas" completas.
2. Entrenar a la red con ruido (Phase Noise Augmentation) para que las neuronas 
   aprendan a sintonizar bandas de resonancia más anchas y robustas.
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
    def __init__(self, phase_scale=1.0):
        super().__init__()
        self.phase_scale = phase_scale
        self.res1 = FastResonantLayer(784, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.res2 = FastResonantLayer(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x_phase = x * self.phase_scale
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

def run_fix_experiment():
    print("🛡️ Iniciando Experimento V205: Fix de Phase Jitter y Robustez")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)
    
    # Reducimos el multiplicador de fase a PI/4 para evitar el wrap-around destructivo
    # y que el ruido cause desincronizaciones parciales, no totales.
    models = {
        "Resonant_Pi_Norm": ResonantMNIST(phase_scale=math.pi).to(device),
        "Resonant_Pi_Div4": ResonantMNIST(phase_scale=math.pi/4).to(device)
    }
    
    epochs = 4
    train_noise_std = 0.5 # Entrenamos CON ruido para vacunar a la red
    
    for name, model in models.items():
        print(f"\n🎬 Entrenando {name} (Data Augmentation Noise={train_noise_std})...")
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(1, epochs + 1):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Inyección de ruido en entrenamiento (Vacuna)
                noise = torch.randn_like(data) * train_noise_std
                data_noisy = data + noise
                
                optimizer.zero_grad()
                output = model(data_noisy)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
            acc = evaluate_with_noise(model, test_loader, device, 0.0)
            print(f"   Época {epoch} | Acc sin ruido: {acc:.4f}")
    
    print("\n🌪️ Test de Robustez (Gaussian Noise)...")
    noise_levels = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    print(f"\n{'Noise (std)':<12} | {'Resonant PI':<12} | {'Resonant PI/4':<12}")
    print("-" * 42)
    
    for std in noise_levels:
        acc_pi = evaluate_with_noise(models["Resonant_Pi_Norm"], test_loader, device, std)
        acc_div4 = evaluate_with_noise(models["Resonant_Pi_Div4"], test_loader, device, std)
        print(f"{std:<12.1f} | {acc_pi:<12.4f} | {acc_div4:<12.4f}")
        
    print("\n✅ V205 Completado. El Jitter de Fase puede mitigarse limitando la escala angular.")

if __name__ == "__main__":
    run_fix_experiment()
