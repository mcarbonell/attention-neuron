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
        self.spectral_params = nn.Parameter(torch.randn(out_features, k_spectral) * 0.1)
        n_dct = 1024 if in_features <= 1024 else 2048
        self.register_buffer('dct_mat', get_dct_matrix(n_dct, device))
        self.bn = nn.BatchNorm1d(out_features)
        
    def forward(self, x):
        w = torch.matmul(self.spectral_params, self.dct_mat[:self.k, :self.in_features])
        z = F.linear(x, w) / self.in_features
        return self.bn(z)

# --- MODELO ESCALABLE ---
class ScalingNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.l1 = DCTLayer(784, hidden_size, k_spectral=16)
        self.l2 = DCTLayer(hidden_size, 10, k_spectral=16)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        return self.l2(x)

def train_and_eval(hidden_size):
    print(f"\n>>> TESTANDO ESCALADO: {hidden_size} NEURONAS DCT")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    model = ScalingNet(hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros Totales: {total_params}")

    model.train()
    for epoch in range(1, 4): # 3 épocas rápidas para comparar tendencia
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Época {epoch} terminada.")

    model.eval()
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_acc = 100. * test_correct / len(test_loader.dataset)
    print(f"RESULTADO ({hidden_size} neuronas): {test_acc:.2f}%")
    return test_acc, total_params

def run_experiment():
    print(f"--- EXPERIMENTO V178: SPECTRAL SCALING BENCHMARK ---")
    results = {}
    for h in [512, 1024, 2048]:
        acc, params = train_and_eval(h)
        results[h] = {"accuracy": acc, "params": params}
    
    print("\n" + "="*55)
    print(f"RESUMEN DE ESCALADO ESPECTRAL (V178)")
    print(f"="*55)
    for h, res in results.items():
        print(f"{h} Neuronas: {res['accuracy']:.2f}% | Params: {res['params']}")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v178_spectral_scaling.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_experiment()
