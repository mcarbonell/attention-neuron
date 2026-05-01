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

# --- CAPA DE RESONANCIA CORREGIDA (V184) ---
class DebuggedResonanceLayer(nn.Module):
    def __init__(self, k=32):
        super().__init__()
        self.k = k
        
        # FRECUENCIAS COMPARTIDAS (Invarianza a traslación vertical/horizontal)
        self.row_freq = nn.Parameter(torch.randn(k) * 0.5)
        self.row_phase = nn.Parameter(torch.randn(k) * 2 * np.pi)
        
        self.col_freq = nn.Parameter(torch.randn(k) * 0.5)
        self.col_phase = nn.Parameter(torch.randn(k) * 2 * np.pi)
        
        self.register_buffer('pos', torch.linspace(1, 28, 28))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # x está en [0, 1] (0=negro, 1=blanco)
        
        # --- RESONANCIA DE FILAS ---
        row_phases = self.pos.view(28, 1) * self.row_freq.view(1, self.k) + self.row_phase.view(1, self.k)
        r_template = torch.sin(row_phases) # Oscila en [-1, 1]
        
        # LÓGICA: x * template
        # Si pixel=1 y template=1 -> +1 (Acierto)
        # Si pixel=1 y template=-1 -> -1 (Error)
        # Si pixel=0 -> 0 (Ignoramos el fondo para no saturar con el vacío)
        row_res = torch.matmul(x, r_template) # (B, 28, K)
        
        # --- RESONANCIA DE COLUMNAS ---
        x_t = x.transpose(1, 2)
        col_phases = self.pos.view(28, 1) * self.col_freq.view(1, self.k) + self.col_phase.view(1, self.k)
        c_template = torch.sin(col_phases)
        col_res = torch.matmul(x_t, c_template) # (B, 28, K)
        
        features = torch.cat([row_res.view(batch_size, -1), col_res.view(batch_size, -1)], dim=1)
        return features

# --- MODELO RESONANCIA DEBUGGED ---
class DebugResonanceNet(nn.Module):
    def __init__(self, k=32):
        super().__init__()
        self.scanner = DebuggedResonanceLayer(k)
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
    print(f"\n--- EXPERIMENTO V184: RESONANCE DEBUGGED (UNIVERSAL & STABLE) ---")
    
    # Normalización estándar para MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    model = DebugResonanceNet(k=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Red de Resonancia Corregida Lista (K=32, Freq Compartidas).")
    print(f"PARÁMETROS TOTALES: {total_params}")

    # ENTRENAMIENTO
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

    # EVALUACIÓN (Mirror exacto del entrenamiento)
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
    print(f"RESULTADO RESONANCIA DEBUGGED (V184)")
    print(f"="*55)
    print(f"Precisión Test:  {test_acc:.2f}%")
    print(f"Mecánica:        Shared Freq + Balanced Resonance")
    print(f"Parámetros:      {total_params}")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v184_resonance_debug.json", "w") as f:
        json.dump({"accuracy": test_acc, "params": total_params}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
