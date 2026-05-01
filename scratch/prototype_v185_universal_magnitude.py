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

# --- CAPA DE ESCÁNER MAGNITUD UNIVERSAL (V185) ---
class UniversalMagnitudeLayer(nn.Module):
    def __init__(self, k=64): # Subimos a 64 para capturar más texturas
        super().__init__()
        self.k = k
        
        # Frecuencias compartidas inicializadas en un rango útil (0.1 a 1.0)
        # Esto evita que empiecen en valores absurdos
        self.row_freq = nn.Parameter(torch.linspace(0.05, 0.8, k))
        self.col_freq = nn.Parameter(torch.linspace(0.05, 0.8, k))
        
        # Las fases iniciales son aleatorias pero el modelo aprenderá a ignorarlas 
        # gracias al cálculo de magnitud.
        self.row_phase = nn.Parameter(torch.randn(k) * 2 * np.pi)
        self.col_phase = nn.Parameter(torch.randn(k) * 2 * np.pi)
        
        self.register_buffer('pos', torch.linspace(1, 28, 28))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # --- RESONANCIA DE FILAS (MAGNITUD) ---
        # Calculamos Seno y Coseno
        row_phases = self.pos.view(28, 1) * self.row_freq.view(1, self.k) + self.row_phase.view(1, self.k)
        r_sin = torch.sin(row_phases)
        r_cos = torch.cos(row_phases)
        
        # Magnitud: sqrt( (sum x*sin)^2 + (sum x*cos)^2 )
        res_r_sin = torch.matmul(x, r_sin)
        res_r_cos = torch.matmul(x, r_cos)
        row_mag = torch.sqrt(res_r_sin**2 + res_r_cos**2 + 1e-8)
        
        # --- RESONANCIA DE COLUMNAS (MAGNITUD) ---
        x_t = x.transpose(1, 2)
        col_phases = self.pos.view(28, 1) * self.col_freq.view(1, self.k) + self.col_phase.view(1, self.k)
        c_sin = torch.sin(col_phases)
        c_cos = torch.cos(col_phases)
        
        res_c_sin = torch.matmul(x_t, c_sin)
        res_c_cos = torch.matmul(x_t, c_cos)
        col_mag = torch.sqrt(res_c_sin**2 + res_c_cos**2 + 1e-8)
        
        # Concatenamos características (28*K*2 = 3584 si K=64)
        features = torch.cat([row_mag.view(batch_size, -1), col_mag.view(batch_size, -1)], dim=1)
        return features

# --- MODELO UNIVERSAL MAGNITUD ---
class UniversalMagNet(nn.Module):
    def __init__(self, k=64):
        super().__init__()
        self.scanner = UniversalMagnitudeLayer(k)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(28 * 2 * k),
            nn.Linear(28 * 2 * k, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = x.view(-1, 28, 28)
        # Normalización local a [0, 1] para consistencia
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = self.scanner(x)
        return self.classifier(x)

def run_experiment():
    print(f"\n--- EXPERIMENTO V185: UNIVERSAL MAGNITUD (PHASE IMMUNITY) ---")
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    model = UniversalMagNet(k=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Escáner de Magnitud Universal Listo (K=64).")
    print(f"PARÁMETROS TOTALES: {total_params}")

    model.train()
    t0 = time.perf_counter()
    for epoch in range(1, 11): # Damos 10 épocas para que el clasificador se asiente
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        acc = 100. * correct / total
        print(f"Época {epoch} - Acc Train: {acc:.2f}%")
        
        # Validación rápida a mitad de camino
        if epoch % 5 == 0:
            model.eval()
            test_correct = 0
            with torch.no_grad():
                for d_t, t_t in test_loader:
                    d_t, t_t = d_t.to(device), t_t.to(device)
                    out_t = model(d_t)
                    p_t = out_t.argmax(dim=1, keepdim=True)
                    test_correct += p_t.eq(t_t.view_as(p_t)).sum().item()
            print(f"   >>> Acc Test Temporal: {100. * test_correct / 10000:.2f}%")
            model.train()

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
    print(f"RESULTADO MAGNITUD UNIVERSAL (V185)")
    print(f"="*55)
    print(f"Precisión Test:  {test_acc:.2f}%")
    print(f"Mecánica:        Universal Power Spectrum (Phase Invariant)")
    print(f"Parámetros:      {total_params}")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v185_universal_mag.json", "w") as f:
        json.dump({"accuracy": test_acc, "params": total_params}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
