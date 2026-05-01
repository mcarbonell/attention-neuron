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

# --- TRANSFORMADA DE WALSH-HADAMARD RÁPIDA (FWHT) ---
def fwht(x):
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

# --- CAPA DE NEURONAS DE WALSH CON PROMEDIO ---
class SpectralMeanLayer(nn.Module):
    def __init__(self, in_features, out_features, k_spectral=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k_spectral
        
        # Parámetros comprimidos (16 semillas por neurona)
        self.spectral_params = nn.Parameter(torch.randn(out_features, k_spectral) * 0.1)
        # Usamos BatchNorm para estabilizar los promedios antes del clasificador
        self.bn = nn.BatchNorm1d(out_features)
        
    def forward(self, x):
        # 1. SINTETIZAR PESOS (K=16 -> 1024)
        padded_params = torch.zeros(self.out_features, 1024).to(x.device)
        padded_params[:, :self.k] = self.spectral_params
        synthesized_weights = fwht(padded_params)
        w = synthesized_weights[:, :self.in_features] # (256, 784)
        
        # 2. PROMEDIO DEL PRODUCTO (En lugar de Suma)
        # x: (B, 784), w: (256, 784)
        # Queremos un resultado de (B, 256)
        # z_ij = mean_k (x_ik * w_jk)
        x_expanded = x.unsqueeze(1) # (B, 1, 784)
        w_expanded = w.unsqueeze(0) # (1, 256, 784)
        
        # Multiplicación elemento a elemento y promedio
        # Nota: Esto es equivalente a (F.linear(x, w) / 784)
        z = (x_expanded * w_expanded).mean(dim=-1)
        
        # Estabilizamos con BatchNorm (opcional pero recomendado para promedios)
        return self.bn(z)

# --- MODELO SPECTRAL-MEAN ---
class SpectralMeanNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = SpectralMeanLayer(784, 256, k_spectral=16)
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
    print(f"\n--- EXPERIMENTO V175: SPECTRAL-MEAN NEURONS (STABLE SYNTHESIS) ---")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    model = SpectralMeanNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Cerebro de Promedio Espectral: 256 Neuronas (K=16).")
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
    print(f"RESULTADO SPECTRAL-MEAN (V175)")
    print(f"="*55)
    print(f"Precisión Test:  {test_acc:.2f}%")
    print(f"Mecánica:        Walsh Synthesis + Element-wise Mean")
    print(f"Tiempo Total:    {wall_time:.2f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v175_spectral_mean.json", "w") as f:
        json.dump({"accuracy": test_acc, "params": total_params}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
