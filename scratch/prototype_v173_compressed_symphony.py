import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.special as special
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

# --- CAPA SINFÓNICA COMPRIMIDA (SPECTRAL FILTERING) ---
class CompressedSymphonyLayer(nn.Module):
    def __init__(self, n_coeffs=1024, out_features=128):
        super().__init__()
        self.out_features = out_features
        # Filtro Espectral Aprendido (Solo 1024 ganancias y 1024 fases)
        # ESTO SUSTITUYE A LA MATRIZ DE 200,000 PARÁMETROS
        self.gain = nn.Parameter(torch.ones(n_coeffs) * 0.1)
        self.phase_shift = nn.Parameter(torch.randn(n_coeffs) * 0.1)
        
        # Proyección de mezcla (Pequeña matriz para combinar frecuencias en neuronas)
        # Usamos una proyección dispersa o pequeña para mantener la eficiencia
        self.mixer = nn.Parameter(torch.randn(out_features, n_coeffs) * 0.05)
        self.amp = nn.Parameter(torch.ones(out_features) * 0.1)
        
    def forward(self, x_spec):
        # 1. Aplicamos el Filtro Espectral (Modulación de canal)
        # Filtramos las frecuencias de Walsh directamente
        filtered = x_spec * self.gain + self.phase_shift
        
        # 2. Mezclamos para crear los osciladores
        z = F.linear(filtered, self.mixer)
        
        # 3. Voces Exóticas
        out = torch.zeros_like(z)
        g = self.out_features // 4
        out[:, 0:g] = torch.sin(z[:, 0:g])
        out[:, g:2*g] = torch.sin(z[:, g:2*g]) / (z[:, g:2*g] + 1e-8) # Sinc
        out[:, 2*g:3*g] = special.bessel_j0(z[:, 2*g:3*g])
        out[:, 3*g:] = (2 * z[:, 3*g:]) * torch.exp(-torch.clamp(z[:, 3*g:]**2, max=20) / 2) # Hermite
        
        return out * self.amp

# --- MODELO ULTRA-LIGERO ---
class CompressedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.symphony = CompressedSymphonyLayer(1024, 128)
        self.classifier = nn.Linear(128, 10)
        
    def forward(self, x):
        # 1. Pre-procesamiento Walsh (0 parámetros)
        x = x.view(-1, 784)
        padded = torch.zeros(x.size(0), 1024).to(x.device)
        padded[:, :784] = x
        x_spec = fwht(padded)
        
        # 2. Orquesta Comprimida
        x = self.symphony(x_spec)
        return self.classifier(x)

def run_experiment():
    print(f"\n--- EXPERIMENTO V173: COMPRESSED SYMPHONY (SPECTRAL EFFICIENCY) ---")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    model = CompressedNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Orquesta Comprimida Lista.")
    print(f"PARÁMETROS TOTALES: {total_params} (¡Compresión masiva vs V172!)")

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
    print(f"RESULTADO SINFONÍA COMPRIMIDA (V173)")
    print(f"="*55)
    print(f"Precisión Test:  {test_acc:.2f}%")
    print(f"Parámetros:      {total_params}")
    print(f"Mecánica:        FWHT + Spectral Gain + Exotic Resonance")
    print(f"Tiempo Total:    {wall_time:.2f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v173_compressed_symphony.json", "w") as f:
        json.dump({"accuracy": test_acc, "params": total_params}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
