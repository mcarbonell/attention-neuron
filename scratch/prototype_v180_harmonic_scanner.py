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

# --- CAPA DE ESCÁNER ARMÓNICO (FILAS Y COLUMNAS) ---
class HarmonicScannerLayer(nn.Module):
    def __init__(self, k_per_line=10):
        super().__init__()
        self.k = k_per_line
        
        # Parámetros de los osciladores (Frecuencia y Fase)
        # Para Filas (28 grupos de K osciladores)
        self.row_freq = nn.Parameter(torch.randn(28, k_per_line) * 0.5)
        self.row_phase = nn.Parameter(torch.randn(28, k_per_line) * 2 * np.pi)
        
        # Para Columnas (28 grupos de K osciladores)
        self.col_freq = nn.Parameter(torch.randn(28, k_per_line) * 0.5)
        self.col_phase = nn.Parameter(torch.randn(28, k_per_line) * 2 * np.pi)
        
        # Índices de posición (1 a 28)
        self.register_buffer('pos', torch.linspace(1, 28, 28))
        
    def forward(self, x):
        # x: (B, 28, 28)
        batch_size = x.size(0)
        
        # --- RESONANCIA DE FILAS ---
        # Calculamos la fase total: (pos * freq + phase)
        # pos: (28), row_freq: (28, K), row_phase: (28, K)
        # Queremos una matriz de fase: (28, 28, K)
        row_phases = self.pos.view(1, 28, 1) * self.row_freq.view(28, 1, self.k) + self.row_phase.view(28, 1, self.k)
        row_sin = torch.sin(row_phases) # (28, 28, K)
        
        # Multiplicamos cada fila de la imagen por sus K osciladores y sumamos
        # x: (B, 28, 28) -> view (B, 28, 28, 1)
        row_res = (x.unsqueeze(-1) * row_sin.unsqueeze(0)).sum(dim=2) # (B, 28, K)
        
        # --- RESONANCIA DE COLUMNAS ---
        # Transponemos la imagen para tratar las columnas como filas
        x_t = x.transpose(1, 2)
        col_phases = self.pos.view(1, 28, 1) * self.col_freq.view(28, 1, self.k) + self.col_phase.view(28, 1, self.k)
        col_sin = torch.sin(col_phases)
        col_res = (x_t.unsqueeze(-1) * col_sin.unsqueeze(0)).sum(dim=2) # (B, 28, K)
        
        # Concatenamos los resultados (560 características si K=10)
        features = torch.cat([row_res.view(batch_size, -1), col_res.view(batch_size, -1)], dim=1)
        return features

# --- MODELO ESCÁNER ---
class ScannerNet(nn.Module):
    def __init__(self, k_per_line=10):
        super().__init__()
        self.scanner = HarmonicScannerLayer(k_per_line)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(28 * 2 * k_per_line),
            nn.Linear(28 * 2 * k_per_line, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = x.view(-1, 28, 28)
        x = self.scanner(x)
        return self.classifier(x)

def run_experiment():
    print(f"\n--- EXPERIMENTO V180: HARMONIC ROW-COLUMN SCANNER ---")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    model = ScannerNet(k_per_line=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Escáner Armónico Listo: 28 filas + 28 columnas (K=10).")
    print(f"PARÁMETROS TOTALES: {total_params} (¡Eficiencia Extrema!)")

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
    print(f"RESULTADO ESCÁNER ARMÓNICO (V180)")
    print(f"="*55)
    print(f"Precisión Test:  {test_acc:.2f}%")
    print(f"Mecánica:        Row/Column Resonance (Fourier-like Scanner)")
    print(f"Parámetros:      {total_params}")
    print(f"Tiempo Total:    {wall_time:.2f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v180_harmonic_scanner.json", "w") as f:
        json.dump({"accuracy": test_acc, "params": total_params}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
