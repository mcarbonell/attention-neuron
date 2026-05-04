import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time
import json
import os

# --- 1. Utilidades DCT ---
def get_dct_matrix(n):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0: matrix[i, j] = 1.0 / np.sqrt(n)
            else: matrix[i, j] = np.sqrt(2.0 / n) * np.cos(np.pi * i * (2 * j + 1) / (2 * n))
    return torch.from_numpy(matrix).float()

class DCTExtractor(nn.Module):
    def __init__(self, k_size=9):
        super().__init__()
        self.k_size = k_size
        self.register_buffer('dct_mat', get_dct_matrix(28))
    def forward(self, x):
        x = x.squeeze(1)
        res = self.dct_mat @ x @ self.dct_mat.t()
        return res[:, :self.k_size, :self.k_size].reshape(-1, self.k_size**2)

# --- 2. Neurona Periódica Rectificada ---
class StraightPeriodicLayer(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.w_freq = nn.Parameter(torch.randn(1, in_features) * 0.5)
        self.b_phase = nn.Parameter(torch.zeros(1, in_features))
        self.poly = nn.Parameter(torch.tile(torch.tensor([0.0, 0.0, 1.0, 0.0]), (in_features, 1)))
    def forward(self, x):
        z = torch.sigmoid(torch.tan(x * self.w_freq + self.b_phase))
        return (self.poly[:, 0]*(z**3) + self.poly[:, 1]*(z**2) + self.poly[:, 2]*z + self.poly[:, 3])

# --- 3. Modelo V228 ---
class PeriodicRecordBreaker(nn.Module):
    def __init__(self, k_size=9):
        super().__init__()
        n_features = k_size**2 # 81
        self.extractor = DCTExtractor(k_size=k_size)
        self.periodic = StraightPeriodicLayer(n_features)
        self.head = nn.Linear(n_features, 10)
    def forward(self, x):
        x = self.extractor(x)
        x = self.periodic(x)
        return self.head(x)

# --- 4. Entrenamiento Intensivo ---
def train_v228():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=1000)

    model = PeriodicRecordBreaker(k_size=9)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"V228: Asalto al 90% con {total_params} parámetros...")

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Programador para ajustar el aprendizaje dinámicamente
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.02, steps_per_epoch=len(train_loader), epochs=20)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 21):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/20 terminada.")

    # Evaluación Final
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\n--- RESULTADO FINAL V228 ---")
    print(f"Accuracy: {accuracy:.2f}% | Params: {total_params}")
    return accuracy, total_params

# --- 5. Ejecución ---
acc, params = train_v228()

# Resultados
os.makedirs("results/raw", exist_ok=True)
results = {"v228_mnist_90_percent": {"accuracy": acc, "total_params": params}}
with open("results/raw/v228_mnist_90_percent.json", "w") as f:
    json.dump(results, f, indent=4)
