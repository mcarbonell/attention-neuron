import torch
import torch.nn as nn
import torch.nn.functional as F
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

# --- GENERADOR DE MATRIZ DCT (DISCRETE COSINE TRANSFORM) ---
def get_dct_matrix(n):
    """ Genera la matriz de la DCT-II de tamaño n x n """
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                matrix[i, j] = np.sqrt(1/n)
            else:
                matrix[i, j] = np.sqrt(2/n) * np.cos(np.pi * i * (2*j + 1) / (2*n))
    return torch.tensor(matrix, dtype=torch.float32).to(device)

# --- CAPA DE NEURONAS DCT CON PROMEDIO ---
class DCTMeanLayer(nn.Module):
    def __init__(self, in_features, out_features, k_spectral=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k_spectral
        
        # Semillas espectrales DCT (K=16)
        self.spectral_params = nn.Parameter(torch.randn(out_features, k_spectral) * 0.1)
        
        # Matriz DCT fija para la síntesis (usamos 1024 para consistencia con FWHT anterior)
        # Pero solo usaremos los primeros 784 componentes
        self.register_buffer('dct_mat', get_dct_matrix(1024))
        
        self.bn = nn.BatchNorm1d(out_features)
        
    def forward(self, x):
        # 1. SINTETIZAR PESOS (IDCT: Espectral -> Espacial)
        # Los pesos espaciales son la combinación lineal de las bases de la DCT
        # w = spectral_params @ DCT_basis_subset
        # Usamos la transpuesta de la matriz DCT para la síntesis
        w = torch.matmul(self.spectral_params, self.dct_mat[:self.k, :self.in_features])
        
        # 2. PROMEDIO DEL PRODUCTO
        # x: (B, 784), w: (256, 784)
        z = F.linear(x, w) / self.in_features
        
        return self.bn(z)

# --- MODELO DCT-MEAN ---
class DCTMeanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = DCTMeanLayer(784, 256, k_spectral=16)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.hidden(x)
        return self.classifier(x)

def run_experiment():
    print(f"\n--- EXPERIMENTO V176: DCT-MEAN NEURONS (COSINE SYNTHESIS) ---")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    model = DCTMeanNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Cerebro de Promedio DCT: 256 Neuronas (K=16).")
    print(f"PARÁMETROS TOTALES: {total_params}")

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
    print(f"RESULTADO DCT-MEAN (V176)")
    print(f"="*55)
    print(f"Precisión Test:  {test_acc:.2f}%")
    print(f"Mecánica:        DCT Synthesis + Element-wise Mean")
    print(f"Tiempo Total:    {wall_time:.2f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v176_dct_mean.json", "w") as f:
        json.dump({"accuracy": test_acc, "params": total_params}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
