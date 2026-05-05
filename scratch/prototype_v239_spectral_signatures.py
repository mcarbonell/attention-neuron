import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import os
import json
import numpy as np
import math

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

# --- CAPA ESPECTRAL FLEXIBLE ---
class SpectralSignatureLayer(nn.Module):
    def __init__(self, in_features, out_features, k=8, mode='sum'):
        super().__init__()
        self.k = k
        self.mode = mode # 'sum' o 'signature'
        self.in_features = in_features
        self.out_features = out_features
        
        # Parámetros espectrales (k por cada neurona de salida)
        self.spectral_params = nn.Parameter(torch.randn(out_features, k) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features if mode == 'sum' else out_features * k))
        
        # Matriz DCT fija (se ajusta al tamaño de entrada)
        # Para MNIST (784) usamos 1024. Para capas intermedias usamos el tamaño real.
        n_dct = 1 << (in_features - 1).bit_length() 
        self.register_buffer('dct_mat', get_dct_matrix(n_dct, 'cpu'))

    def forward(self, x):
        # 1. Proyectamos la entrada al dominio de la frecuencia (solo las top-k)
        # x_freq: [Batch, k]
        # x: [Batch, in_features]
        x_freq = torch.matmul(x, self.dct_mat[:self.k, :self.in_features].t())
        
        # 2. Multiplicación por los parámetros aprendidos
        # out_spectral: [Batch, out_features, k]
        out_spectral = x_freq.unsqueeze(1) * self.spectral_params.unsqueeze(0)
        
        if self.mode == 'sum':
            # Salida estándar: Suma de componentes + bias
            z = torch.sum(out_spectral, dim=2) + self.bias
            return z
        else:
            # Salida "Signature": Exponemos los k componentes individuales
            # [Batch, out_features * k]
            z = out_spectral.reshape(x.size(0), -1) + self.bias
            return z

# --- MODELOS ---
class BaselineModel(nn.Module):
    def __init__(self, h1=64, k=8):
        super().__init__()
        self.l1 = SpectralSignatureLayer(784, h1, k=k, mode='sum')
        self.bn1 = nn.BatchNorm1d(h1)
        self.l2 = SpectralSignatureLayer(h1, 10, k=k, mode='sum')
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.bn1(self.l1(x)))
        return self.l2(x)

class SignatureModel(nn.Module):
    def __init__(self, h1=64, k=8):
        super().__init__()
        self.k = k
        self.l1 = SpectralSignatureLayer(784, h1, k=k, mode='signature')
        # La entrada de l2 será h1 * k
        self.bn1 = nn.BatchNorm1d(h1 * k)
        self.l2 = SpectralSignatureLayer(h1 * k, 10, k=k, mode='sum')
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.bn1(self.l1(x)))
        return self.l2(x)

# --- ENTRENAMIENTO ---
def train_and_eval(model_type, epochs=5):
    print(f"\n>>> Entrenando Modelo: {model_type}")
    
    if model_type == "Baseline":
        model = BaselineModel().to(device)
    else:
        model = SignatureModel().to(device)
        
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {total_params}")

    metrics = {
        "model": model_type,
        "params": total_params,
        "history": []
    }

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if epoch == 1 and batch_idx < 5:
                print(f"  [Fast Feedback] B{batch_idx} Loss: {loss.item():.4f}")

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                correct += output.argmax(dim=1).eq(target).sum().item()
        
        acc = 100. * correct / 10000
        print(f"  Epoch {epoch} | Test Acc: {acc:.2f}% | Avg Train Loss: {train_loss/len(train_loader):.4f}")
        metrics["history"].append({"epoch": epoch, "acc": acc, "loss": train_loss/len(train_loader)})

    metrics["wall_clock_time"] = time.time() - t0
    metrics["final_acc"] = acc
    return metrics

def main():
    results = []
    results.append(train_and_eval("Baseline"))
    results.append(train_and_eval("Signature"))
    
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v239_spectral_signatures.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*55)
    print(f"{'Modelo':<15} | {'Acc Final':<10} | {'Parámetros':<12} | {'Tiempo':<8}")
    print("-" * 55)
    for r in results:
        print(f"{r['model']:<15} | {r['final_acc']:<10.2f}% | {r['params']:<12,} | {r['wall_clock_time']:<8.1f}s")
    print("="*55)

if __name__ == "__main__":
    main()
