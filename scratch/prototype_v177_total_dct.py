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

# --- GENERADOR DE MATRIZ DCT ---
def get_dct_matrix(n, device_target):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                matrix[i, j] = np.sqrt(1/n)
            else:
                matrix[i, j] = np.sqrt(2/n) * np.cos(np.pi * i * (2*j + 1) / (2*n))
    return torch.tensor(matrix, dtype=torch.float32).to(device_target)

# --- CAPA DCT REUTILIZABLE ---
class DCTLayer(nn.Module):
    def __init__(self, in_features, out_features, k_spectral=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k_spectral
        
        # Semillas espectrales
        self.spectral_params = nn.Parameter(torch.randn(out_features, k_spectral) * 0.1)
        
        # Matriz DCT fija (ajustada al tamaño de entrada de esta capa)
        # Usamos una potencia de 2 superior o el tamaño exacto
        n_dct = 1024 if in_features <= 1024 else 2048
        self.register_buffer('dct_mat', get_dct_matrix(n_dct, device))
        
        self.bn = nn.BatchNorm1d(out_features)
        
    def forward(self, x):
        # Síntesis de Pesos
        w = torch.matmul(self.spectral_params, self.dct_mat[:self.k, :self.in_features])
        # Promedio del producto
        z = F.linear(x, w) / self.in_features
        return self.bn(z)

# --- MODELO TOTALMENTE DCT ---
class TotalDCTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Capa 1: 784 -> 256 (DCT K=16)
        self.l1 = DCTLayer(784, 256, k_spectral=16)
        # Capa 2: 256 -> 10 (DCT K=16)
        self.l2 = DCTLayer(256, 10, k_spectral=16)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.l1(x)
        # Añadimos una no-linealidad básica entre capas para que no sea puramente lineal
        x = F.relu(x)
        x = self.l2(x)
        return x

def run_experiment():
    print(f"\n--- EXPERIMENTO V177: TOTAL DCT NETWORK (100% SPECTRAL) ---")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    model = TotalDCTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Red 100% Espectral Lista.")
    print(f"PARÁMETROS TOTALES: {total_params} (¡Compresión de Élite!)")

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
    print(f"RESULTADO TOTAL-DCT (V177)")
    print(f"="*55)
    print(f"Precisión Test:  {test_acc:.2f}%")
    print(f"Parámetros:      {total_params}")
    print(f"Mecánica:        100% DCT Synthesis (L1 + L2)")
    print(f"Tiempo Total:    {wall_time:.2f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v177_total_dct.json", "w") as f:
        json.dump({"accuracy": test_acc, "params": total_params}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
