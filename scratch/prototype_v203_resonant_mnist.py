"""
scratch/prototype_v203_resonant_mnist.py — Resonant Phase MNIST

Experimento de frontera (V203):
Escalamos la "Neurona de Resonancia" a alta dimensionalidad (MNIST).
Para evitar un OOM (Out Of Memory) al calcular la interferencia de fase
(batch, out_features, in_features), aplicamos una optimización matemática
usando identidades trigonométricas:
  cos(x - w) = cos(x)cos(w) + sin(x)sin(w)
Esto permite calcular la resonancia como dos productos matriciales densos
ultrarrápidos (dot products), haciendo la capa perfectamente escalable.

Input: Pixeles mapeados a Fase (0 a PI).
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

class FastResonantLayer(nn.Module):
    """
    Capa de Resonancia Optimizada Matemáticamente.
    Calcula sum( magnitude * cos(x_phase - w_phase) )
    en O(Batch x In) + O(Batch x Out) en lugar de O(Batch x Out x In).
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Sintonizadores de fase (0 a 2pi) y ganancia
        self.phase_sintonizer = nn.Parameter(torch.rand(out_features, in_features) * 2 * math.pi)
        self.magnitude = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))

    def forward(self, x_phase):
        # 1. Proyectar Input al espacio de Fases Complejas
        x_cos = torch.cos(x_phase)
        x_sin = torch.sin(x_phase)
        
        # 2. Proyectar Pesos al espacio de Fases Complejas
        w_cos = torch.cos(self.phase_sintonizer) * self.magnitude
        w_sin = torch.sin(self.phase_sintonizer) * self.magnitude
        
        # 3. Interferencia (Producto Punto)
        # equivalente a: sum(cos(x-w)*m)
        resonant_sum = F.linear(x_cos, w_cos) + F.linear(x_sin, w_sin)
        
        return F.relu(resonant_sum)

class ResonantMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # Arquitectura simple: 784 -> 128 -> 10
        self.res1 = FastResonantLayer(784, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.res2 = FastResonantLayer(128, 10)

    def forward(self, x):
        # Aplanar imagen
        x = x.view(x.size(0), -1)
        
        # Mapear intensidad [0, 1] a Fase [0, PI]
        # (Asumimos que el input ya está entre 0 y 1 o normalizado,
        # la normalización estándar de MNIST puede generar valores negativos,
        # pero para la fase funciona como un shift).
        x_phase = x * math.pi
        
        x = self.res1(x_phase)
        x = self.bn1(x)
        x = self.res2(x)
        
        return x

def run_mnist_experiment():
    print("🎬 Iniciando Experimento V203: Resonancia en MNIST")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)
    
    model = ResonantMNIST().to(device)
    
    # Conteo de parámetros
    params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros Totales: {params}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 5
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # REGLA DE SUPERVIVENCIA: Fast Feedback
            if epoch == 1 and batch_idx < 5:
                print(f"   [Fast Feedback] Epoch 1 - Batch {batch_idx} | Loss: {loss.item():.4f}")
                
        # Eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = correct / 10000
        print(f"   Época {epoch} Completada | Acc Test: {acc:.4f} | Tiempo total: {time.time()-t0:.1f}s")
        
    print("\n✅ Experimento V203 preparado. Puedes ejecutar más épocas o ajustes en tu terminal.")

if __name__ == "__main__":
    run_mnist_experiment()
