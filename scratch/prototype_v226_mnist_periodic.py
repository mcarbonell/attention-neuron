import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time
import json
import os

# --- 1. La Neurona Periódica "Enderezada" (V225) ---
class StraightPeriodicLayer(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        # Parámetros por cada característica: freq, phase, amp, bias + 4 poli
        self.w_freq = nn.Parameter(torch.randn(1, in_features) * 0.1)
        self.b_phase = nn.Parameter(torch.zeros(1, in_features))
        self.poly = nn.Parameter(torch.tile(torch.tensor([0.0, 0.0, 1.0, 0.0]), (in_features, 1)))

    def forward(self, x):
        # x shape: (batch, in_features)
        z = torch.sigmoid(torch.tan(x * self.w_freq + self.b_phase))
        
        # Corrección polinómica vectorizada
        # poly shape: (in_features, 4)
        out = (self.poly[:, 0] * (z**3) + 
               self.poly[:, 1] * (z**2) + 
               self.poly[:, 2] * z + 
               self.poly[:, 3])
        return out

# --- 2. Modelo V226 ---
class PeriodicSpectralClassifier(nn.Module):
    def __init__(self, n_coeffs=32):
        super().__init__()
        self.n_coeffs = n_coeffs
        # Extraeremos coeficientes DCT fijos (no entrenables)
        self.periodic_layer = StraightPeriodicLayer(n_coeffs)
        self.head = nn.Linear(n_coeffs, 10)
        
    def forward(self, x):
        # Supone que x ya son los coeficientes DCT
        x = self.periodic_layer(x)
        return self.head(x)

# --- 3. Utilidad DCT (Pre-procesamiento) ---
def get_dct_coeffs(x, n=32):
    # Simplificación: usamos una proyección aleatoria fija o DCT real si estuviera disponible
    # Para este prototipo, simulamos la extracción espectral con una matriz fija
    if not hasattr(get_dct_coeffs, "projection"):
        torch.manual_seed(42)
        get_dct_coeffs.projection = torch.randn(784, n) / np.sqrt(784)
    return x.view(-1, 784) @ get_dct_coeffs.projection

# --- 4. Entrenamiento ---
def train_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=1000)

    model = PeriodicSpectralClassifier(n_coeffs=32)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Entrenando V226 con {total_params} parámetros...")

    for epoch in range(1, 4): # Solo 3 épocas para velocidad
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Extraer coeficientes espectrales (fijo)
            data_spec = get_dct_coeffs(data, n=32)
            
            optimizer.zero_grad()
            output = model(data_spec)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx < 5 and epoch == 1:
                print(f"Epoch 1 Batch {batch_idx}: Loss = {loss.item():.4f}")

    # Evaluación
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data_spec = get_dct_coeffs(data, n=32)
            output = model(data_spec)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nFinal Accuracy: {accuracy:.2f}%")
    return accuracy, total_params

# --- 5. Ejecución ---
acc, params = train_mnist()

# Guardar resultados
os.makedirs("results/raw", exist_ok=True)
results = {"v226_mnist_periodic": {"accuracy": acc, "total_params": params}}
with open("results/raw/v226_mnist_periodic.json", "w") as f:
    json.dump(results, f, indent=4)
