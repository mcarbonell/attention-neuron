import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time
import json
import os

# --- 1. Utilidad DCT-2D Fija ---
def get_dct_matrix(n):
    """Genera la matriz DCT-II de tamaño n x n"""
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                matrix[i, j] = 1.0 / np.sqrt(n)
            else:
                matrix[i, j] = np.sqrt(2.0 / n) * np.cos(np.pi * i * (2 * j + 1) / (2 * n))
    return torch.from_numpy(matrix).float()

class DCTExtractor(nn.Module):
    def __init__(self, k=64):
        super().__init__()
        self.k = k
        # Matriz DCT-II para 28x28
        self.register_buffer('dct_mat', get_dct_matrix(28))
        
    def forward(self, x):
        # x: (batch, 1, 28, 28)
        # 2D DCT: C * X * C^T
        x = x.squeeze(1)
        # Aplicamos DCT por filas y luego por columnas
        res = self.dct_mat @ x @ self.dct_mat.t()
        # Tomamos los k coeficientes del bloque superior izquierdo (bajas frecuencias)
        # Usamos un aplanado simple para prototipo
        return res[:, :8, :8].reshape(-1, 64) # 8x8 = 64 coeficientes

# --- 2. Capa Periódica Rectificada (V225) ---
class StraightPeriodicLayer(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.w_freq = nn.Parameter(torch.randn(1, in_features) * 0.5)
        self.b_phase = nn.Parameter(torch.zeros(1, in_features))
        self.poly = nn.Parameter(torch.tile(torch.tensor([0.0, 0.0, 1.0, 0.0]), (in_features, 1)))

    def forward(self, x):
        z = torch.sigmoid(torch.tan(x * self.w_freq + self.b_phase))
        out = (self.poly[:, 0] * (z**3) + 
               self.poly[:, 1] * (z**2) + 
               self.poly[:, 2] * z + 
               self.poly[:, 3])
        return out

# --- 3. Modelo V227 ---
class PeriodicDCTClassifier(nn.Module):
    def __init__(self, k=64):
        super().__init__()
        self.extractor = DCTExtractor(k=k)
        self.periodic = StraightPeriodicLayer(k)
        self.head = nn.Linear(k, 10)
        
    def forward(self, x):
        coeffs = self.extractor(x)
        features = self.periodic(coeffs)
        return self.head(features)

# --- 4. Entrenamiento ---
def train_v227():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=1000)

    model = PeriodicDCTClassifier(k=64)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"V227: Clasificador DCT Periódico con {total_params} parámetros.")

    for epoch in range(1, 11): # 10 épocas para dar tiempo a sintonizar
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx < 1 and epoch % 2 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # Evaluación
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nFinal Accuracy V227: {accuracy:.2f}%")
    return accuracy, total_params

# --- 5. Ejecución ---
acc, params = train_v227()

# Resultados
os.makedirs("results/raw", exist_ok=True)
results = {"v227_mnist_dct_periodic": {"accuracy": acc, "total_params": params}}
with open("results/raw/v227_mnist_dct_periodic.json", "w") as f:
    json.dump(results, f, indent=4)
