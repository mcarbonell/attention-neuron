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

# --- CAPA DE ESCÁNER INVARIANTE A LA FASE (MAGNITUD DE FOURIER) ---
class InvariantScannerLayer(nn.Module):
    def __init__(self, k_per_line=10):
        super().__init__()
        self.k = k_per_line
        
        # Frecuencias aprendidas (La fase ya no es crítica para la invarianza)
        self.row_freq = nn.Parameter(torch.randn(28, k_per_line) * 0.5)
        self.col_freq = nn.Parameter(torch.randn(28, k_per_line) * 0.5)
        
        # Añadimos una fase aprendida por si ayuda a la discriminación, 
        # pero la magnitud la hará menos sensible.
        self.row_phase = nn.Parameter(torch.randn(28, k_per_line) * 2 * np.pi)
        self.col_phase = nn.Parameter(torch.randn(28, k_per_line) * 2 * np.pi)
        
        self.register_buffer('pos', torch.linspace(1, 28, 28))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # --- PROCESAMIENTO DE FILAS ---
        # Calculamos Seno y Coseno para obtener la Magnitud
        row_phases = self.pos.view(1, 28, 1) * self.row_freq.view(28, 1, self.k) + self.row_phase.view(28, 1, self.k)
        
        r_sin = torch.sin(row_phases)
        r_cos = torch.cos(row_phases)
        
        # Proyecciones
        res_r_sin = (x.unsqueeze(-1) * r_sin.unsqueeze(0)).sum(dim=2)
        res_r_cos = (x.unsqueeze(-1) * r_cos.unsqueeze(0)).sum(dim=2)
        
        # Magnitud (Invariante a traslaciones en la fila)
        row_mag = torch.sqrt(res_r_sin**2 + res_r_cos**2 + 1e-8)
        
        # --- PROCESAMIENTO DE COLUMNAS ---
        x_t = x.transpose(1, 2)
        col_phases = self.pos.view(1, 28, 1) * self.col_freq.view(28, 1, self.k) + self.col_phase.view(28, 1, self.k)
        
        c_sin = torch.sin(col_phases)
        c_cos = torch.cos(col_phases)
        
        res_c_sin = (x_t.unsqueeze(-1) * c_sin.unsqueeze(0)).sum(dim=2)
        res_c_cos = (x_t.unsqueeze(-1) * c_cos.unsqueeze(0)).sum(dim=2)
        
        # Magnitud (Invariante a traslaciones en la columna)
        col_mag = torch.sqrt(res_c_sin**2 + res_c_cos**2 + 1e-8)
        
        # Concatenamos (560 características)
        features = torch.cat([row_mag.view(batch_size, -1), col_mag.view(batch_size, -1)], dim=1)
        return features

# --- MODELO ESCÁNER INVARIANTE ---
class InvariantScannerNet(nn.Module):
    def __init__(self, k_per_line=10):
        super().__init__()
        self.scanner = InvariantScannerLayer(k_per_line)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(28 * 2 * k_per_line),
            nn.Linear(28 * 2 * k_per_line, 256),
            nn.ReLU(),
            nn.Dropout(0.2), # Añadimos dropout para evitar el sobreajuste masivo
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        x = x.view(-1, 28, 28)
        x = self.scanner(x)
        return self.classifier(x)

def run_experiment():
    print(f"\n--- EXPERIMENTO V181: PHASE-INVARIANT HARMONIC SCANNER ---")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    model = InvariantScannerNet(k_per_line=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Escáner Invariante Listo: Magnitud de Fourier (Row/Col).")
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
    print(f"RESULTADO ESCÁNER INVARIANTE (V181)")
    print(f"="*55)
    print(f"Precisión Test:  {test_acc:.2f}%")
    print(f"Mecánica:        Fourier Magnitude (Phase Invariant)")
    print(f"Parámetros:      {total_params}")
    print(f"Tiempo Total:    {wall_time:.2f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v181_invariant_scanner.json", "w") as f:
        json.dump({"accuracy": test_acc, "params": total_params}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
