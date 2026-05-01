import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.special as special
from torchvision import datasets, transforms
import time
import os
import json
import numpy as np

# --- CONFIGURACIÓN DE DISPOSITIVO ---
device = torch.device('cpu')
try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using DirectML device: {device}")
except ImportError:
    print("torch-directml not found, using CPU")

# --- FUNCIONES ESPECIALES ---
def sinc(x):
    eps = 1e-8
    return torch.sin(x) / (x + eps)

def hermite_gauss(x):
    """ Función de Hermite de primer orden modulada por Gaussiana (Localizada) """
    # H1(x) * exp(-x^2/2)
    return (2 * x) * torch.exp(-torch.clamp(x**2, max=20) / 2)

# --- CAPA DE FUNCIONES ESPECIALES (EXÓTICA) ---
class SpecialSymphonyLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        # Frecuencia (Pesos), Fase (Bias), Amplitud (Scale)
        self.freq = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.phase = nn.Parameter(torch.randn(out_features) * 2 * np.pi)
        self.amp = nn.Parameter(torch.ones(out_features) * 0.1)
        
        # Repartimos las voces en 4 grupos
        self.g = out_features // 4
        
    def forward(self, x):
        z = F.linear(x, self.freq, self.phase)
        out = torch.zeros_like(z)
        
        # Grupo 1: SENO (Estructura Periódica)
        out[:, 0:self.g] = torch.sin(z[:, 0:self.g])
        
        # Grupo 2: SINC (Foco y Atención)
        out[:, self.g:2*self.g] = sinc(z[:, self.g:2*self.g])
        
        # Grupo 3: BESSEL J0 (Resonancia Circular / Bucles)
        # Nota: bessel_j0 es excelente para patrones con simetría radial
        out[:, 2*self.g:3*self.g] = special.bessel_j0(z[:, 2*self.g:3*self.g])
        
        # Grupo 4: HERMITE-GAUSS (Eventos Locales / Puntos críticos)
        out[:, 3*self.g:] = hermite_gauss(z[:, 3*self.g:])
        
        return out * self.amp

# --- MODELO EXÓTICO ---
class ExoticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.exotic_bank = SpecialSymphonyLayer(784, 256)
        self.classifier = nn.Linear(256, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.exotic_bank(x)
        return self.classifier(x)

def run_experiment():
    print(f"\n--- EXPERIMENTO V172: SPECIAL FUNCTIONS SYMPHONY (BESSEL & HERMITE) ---")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    model = ExoticNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    print(f"Orquesta Exótica Lista: Sin, Sinc, Bessel J0, Hermite-Gauss.")
    print(f"Parámetros: {sum(p.numel() for p in model.parameters())}")

    model.train()
    t0 = time.perf_counter()
    for epoch in range(1, 6):
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            if batch_idx == 0:
                print(f"Época {epoch} [Batch 0] - Loss: {loss.item():.4f}")

        acc = 100. * correct / len(train_loader.dataset)
        print(f"Fin Época {epoch} - Accuracy Train: {acc:.2f}%")

    wall_time = time.perf_counter() - t0

    model.eval()
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = 100. * test_correct / len(test_loader.dataset)
    
    print("\n" + "="*55)
    print(f"RESULTADO SINFONÍA EXÓTICA (V172)")
    print(f"="*55)
    print(f"Precisión Test:  {test_acc:.2f}%")
    print(f"Componentes:     Bessel, Hermite, Sinc, Sin")
    print(f"Tiempo Total:    {wall_time:.2f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v172_special_functions.json", "w") as f:
        json.dump({"accuracy": test_acc, "wall_time": wall_time}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
