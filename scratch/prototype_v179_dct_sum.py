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

# --- CAPA DCT CON SUMA (TRADICIONAL) ---
class DCTSumLayer(nn.Module):
    def __init__(self, in_features, out_features, k_spectral=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k_spectral
        
        # Semillas espectrales
        self.spectral_params = nn.Parameter(torch.randn(out_features, k_spectral) * 0.1)
        
        # Matriz DCT fija
        n_dct = 1024 if in_features <= 1024 else 2048
        self.register_buffer('dct_mat', get_dct_matrix(n_dct, device))
        
        # La suma puede generar valores grandes, el BN es crítico aquí
        self.bn = nn.BatchNorm1d(out_features)
        
    def forward(self, x):
        # 1. SINTETIZAR PESOS (IDCT)
        w = torch.matmul(self.spectral_params, self.dct_mat[:self.k, :self.in_features])
        
        # 2. SUMA DEL PRODUCTO (F.linear ya hace la suma)
        z = F.linear(x, w)
        
        return self.bn(z)

# --- MODELO TOTALMENTE DCT (SUM VERSION) ---
class TotalDCTSumNet(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        # Capa 1: 784 -> 1024 (DCT K=16)
        self.l1 = DCTSumLayer(784, hidden_size, k_spectral=16)
        # Capa 2: 1024 -> 10 (DCT K=16)
        self.l2 = DCTSumLayer(hidden_size, 10, k_spectral=16)
        
    def forward(self, x):
        x = x.view(-1, 784)
        # Intuición: Capa 1 (Suma) + ReLU
        x = F.relu(self.l1(x))
        # Clasificación: Capa 2 (Suma)
        return self.l2(x)

def run_experiment():
    print(f"\n--- EXPERIMENTO V179: TOTAL DCT SUM (ENERGY ACCUMULATION) ---")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    # Usamos 1024 neuronas como compromiso entre potencia y eficiencia
    model = TotalDCTSumNet(hidden_size=1024).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Red 100% Espectral con Suma Lista.")
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
    print(f"RESULTADO TOTAL-DCT-SUM (V179)")
    print(f"="*55)
    print(f"Precisión Test:  {test_acc:.2f}%")
    print(f"Mecánica:        100% DCT Synthesis (L1 + L2) + SUM")
    print(f"Parámetros:      {total_params}")
    print(f"Tiempo Total:    {wall_time:.2f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v179_dct_sum.json", "w") as f:
        json.dump({"accuracy": test_acc, "params": total_params}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
