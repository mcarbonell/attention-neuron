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

# --- CAPA DE ESCÁNER UNIVERSAL (INVARIANTE A TRASLACIÓN) ---
class UniversalScannerLayer(nn.Module):
    def __init__(self, k=32): # Aumentamos K para compensar que ahora son compartidas
        super().__init__()
        self.k = k
        
        # Frecuencias compartidas por todas las filas/columnas
        # Solo K parámetros en lugar de 28*K
        self.row_freq = nn.Parameter(torch.randn(k) * 0.5)
        self.row_phase = nn.Parameter(torch.randn(k) * 2 * np.pi)
        
        self.col_freq = nn.Parameter(torch.randn(k) * 0.5)
        self.col_phase = nn.Parameter(torch.randn(k) * 2 * np.pi)
        
        self.register_buffer('pos', torch.linspace(1, 28, 28))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # --- PROCESAMIENTO UNIVERSAL DE FILAS ---
        # pos: (28), row_freq: (K), row_phase: (K)
        # row_phases: (28, K) -> las mismas para cada una de las 28 filas
        row_phases = self.pos.view(28, 1) * self.row_freq.view(1, self.k) + self.row_phase.view(1, self.k)
        
        r_sin = torch.sin(row_phases)
        r_cos = torch.cos(row_phases)
        
        # Proyecciones: (B, 28, 28) @ (28, K) -> (B, 28, K)
        # Para cada fila b,r calculamos la resonancia con los K osciladores
        res_r_sin = torch.matmul(x, r_sin)
        res_r_cos = torch.matmul(x, r_cos)
        
        # Magnitud (Invariante a desfases dentro de la fila)
        row_mag = torch.sqrt(res_r_sin**2 + res_r_cos**2 + 1e-8)
        
        # --- PROCESAMIENTO UNIVERSAL DE COLUMNAS ---
        x_t = x.transpose(1, 2)
        col_phases = self.pos.view(28, 1) * self.col_freq.view(1, self.k) + self.col_phase.view(1, self.k)
        
        c_sin = torch.sin(col_phases)
        c_cos = torch.cos(col_phases)
        
        res_c_sin = torch.matmul(x_t, c_sin)
        res_c_cos = torch.matmul(x_t, c_cos)
        
        col_mag = torch.sqrt(res_c_sin**2 + res_c_cos**2 + 1e-8)
        
        # Concatenamos (28*K + 28*K características)
        features = torch.cat([row_mag.view(batch_size, -1), col_mag.view(batch_size, -1)], dim=1)
        return features

# --- MODELO ESCÁNER UNIVERSAL ---
class UniversalScannerNet(nn.Module):
    def __init__(self, k=32):
        super().__init__()
        self.scanner = UniversalScannerLayer(k)
        # El número de características ahora es 28 * 2 * K
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(28 * 2 * k),
            nn.Linear(28 * 2 * k, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        x = x.view(-1, 28, 28)
        x = self.scanner(x)
        return self.classifier(x)

def run_experiment():
    print(f"\n--- EXPERIMENTO V182: UNIVERSAL HARMONIC SCANNER (TRUE INVARIANCE) ---")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    # Usamos K=32 para dar más capacidad expresiva al ser compartida
    model = UniversalScannerNet(k=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Escáner Universal Listo: Frecuencias Compartidas (K=32).")
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
    print(f"RESULTADO ESCÁNER UNIVERSAL (V182)")
    print(f"="*55)
    print(f"Precisión Test:  {test_acc:.2f}%")
    print(f"Mecánica:        Shared Frequencies (Universal Scanning)")
    print(f"Parámetros:      {total_params}")
    print(f"Tiempo Total:    {wall_time:.2f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v182_universal_scanner.json", "w") as f:
        json.dump({"accuracy": test_acc, "params": total_params}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
