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

# --- CAPA DE RESONANCIA PURA (V186 - VISIÓN DEL USUARIO) ---
class PureResonanceLayer(nn.Module):
    def __init__(self, k_per_line=10):
        super().__init__()
        self.k = k_per_line
        
        # INICIALIZACIÓN ORDENADA (No aleatoria)
        # 28 filas, cada una con k frecuencias inicializadas linealmente
        freqs = torch.linspace(0.05, 0.5, k_per_line).repeat(28, 1)
        self.row_freq = nn.Parameter(freqs)
        self.row_phase = nn.Parameter(torch.zeros(28, k_per_line))
        
        self.col_freq = nn.Parameter(freqs.clone())
        self.col_phase = nn.Parameter(torch.zeros(28, k_per_line))
        
        self.register_buffer('pos', torch.linspace(1, 28, 28))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # --- RESONANCIA DE FILAS ---
        # row_phases: (28, 28, K) -> (fila_idx, pos_pixel, k_idx)
        # pos: (28) -> pixel position
        # row_freq: (28, K) -> frequency for each row and oscillator
        row_phases = self.pos.view(1, 28, 1) * self.row_freq.view(28, 1, self.k) + self.row_phase.view(28, 1, self.k)
        r_template = torch.sin(row_phases) # Oscila en [-1, 1]
        
        # x: (B, 28, 28) -> unsqueeze (B, 28, 28, 1)
        # r_template: (1, 28, 28, K)
        # Producto pixel * seno: 
        # Si pixel=1 y seno=1 -> +1
        # Si pixel=1 y seno=-1 -> -1
        # Si pixel=0 -> 0
        row_res = (x.unsqueeze(-1) * r_template.unsqueeze(0)).sum(dim=2) # (B, 28, K)
        
        # --- RESONANCIA DE COLUMNAS ---
        x_t = x.transpose(1, 2)
        col_phases = self.pos.view(1, 28, 1) * self.col_freq.view(28, 1, self.k) + self.col_phase.view(28, 1, self.k)
        c_template = torch.sin(col_phases)
        col_res = (x_t.unsqueeze(-1) * c_template.unsqueeze(0)).sum(dim=2) # (B, 28, K)
        
        features = torch.cat([row_res.view(batch_size, -1), col_res.view(batch_size, -1)], dim=1)
        return features

# --- MODELO RESONANCIA PURA ---
class PureResonanceNet(nn.Module):
    def __init__(self, k_per_line=10):
        super().__init__()
        self.scanner = PureResonanceLayer(k_per_line)
        # Clasificador para interpretar los 560 resultados
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(28 * 2 * k_per_line),
            nn.Linear(28 * 2 * k_per_line, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = x.view(-1, 28, 28)
        # Mapeamos a [0, 1] para que el fondo sea 0 exactamente
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = self.scanner(x)
        return self.classifier(x)

def run_experiment():
    print(f"\n--- EXPERIMENTO V186: PURE HARMONIC RESONANCE (USER VISION) ---")
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    model = PureResonanceNet(k_per_line=10).to(device)
    # Solo entrenamos frecuencias y fases del escáner + el clasificador
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Red de Resonancia Pura Lista. Freq inicializadas de 0.05 a 0.5.")
    print(f"PARÁMETROS TOTALES: {total_params}")

    model.train()
    t0 = time.perf_counter()
    for epoch in range(1, 6):
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
            
        print(f"Época {epoch} - Acc Train: {100. * correct / total:.2f}%")

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
    print(f"RESULTADO RESONANCIA PURA (V186)")
    print(f"="*55)
    print(f"Precisión Test:  {test_acc:.2f}%")
    print(f"Mecánica:        User Logic (Pixel * Sin) + Fixed Init")
    print(f"Parámetros:      {total_params}")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v186_pure_resonance.json", "w") as f:
        json.dump({"accuracy": test_acc, "params": total_params}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
