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

# --- CAPA DE RESONANCIA POR PLANTILLA (VISIÓN DEL USUARIO) ---
class TemplateResonanceLayer(nn.Module):
    def __init__(self, k_per_line=10):
        super().__init__()
        self.k = k_per_line
        
        # Frecuencias y Fases para Filas y Columnas
        self.row_freq = nn.Parameter(torch.randn(28, k_per_line) * 0.5)
        self.row_phase = nn.Parameter(torch.randn(28, k_per_line) * 2 * np.pi)
        
        self.col_freq = nn.Parameter(torch.randn(28, k_per_line) * 0.5)
        self.col_phase = nn.Parameter(torch.randn(28, k_per_line) * 2 * np.pi)
        
        self.register_buffer('pos', torch.linspace(1, 28, 28))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Mapeamos la entrada de [0, 1] a [-1, 1]
        # Así, los 0s de la imagen se vuelven -1
        x_mapped = 2 * x - 1
        
        # --- RESONANCIA DE FILAS ---
        row_phases = self.pos.view(1, 28, 1) * self.row_freq.view(28, 1, self.k) + self.row_phase.view(28, 1, self.k)
        r_template = torch.sin(row_phases) # Oscila en [-1, 1]
        
        # Lógica de Usuario:
        # Si x_mapped es -1 (era 0) y r_template es -1 (predice 0) -> Producto = +1 (Acierto)
        # Si x_mapped es 1 (era 1) y r_template es 1 (predice 1) -> Producto = +1 (Acierto)
        # Si x_mapped es 1 y r_template es -1 -> Producto = -1 (Error/Penalización)
        # Si x_mapped es -1 y r_template es 1 -> Producto = -1 (Error/Penalización)
        
        # x_mapped: (B, 28, 28) -> (B, 28, 28, 1)
        # r_template: (28, 28, K) -> (1, 28, 28, K)
        row_res = (x_mapped.unsqueeze(-1) * r_template.unsqueeze(0)).sum(dim=2) # (B, 28, K)
        
        # --- RESONANCIA DE COLUMNAS ---
        x_t_mapped = x_mapped.transpose(1, 2)
        col_phases = self.pos.view(1, 28, 1) * self.col_freq.view(28, 1, self.k) + self.col_phase.view(28, 1, self.k)
        c_template = torch.sin(col_phases)
        col_res = (x_t_mapped.unsqueeze(-1) * c_template.unsqueeze(0)).sum(dim=2) # (B, 28, K)
        
        features = torch.cat([row_res.view(batch_size, -1), col_res.view(batch_size, -1)], dim=1)
        return features

# --- MODELO DE RESONANCIA ---
class ResonanceNet(nn.Module):
    def __init__(self, k_per_line=10):
        super().__init__()
        self.scanner = TemplateResonanceLayer(k_per_line)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(28 * 2 * k_per_line),
            nn.Linear(28 * 2 * k_per_line, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        # Aseguramos que la entrada esté en [0, 1] para la lógica de plantilla
        # MNIST transformado suele estar normalizado, lo re-escalamos a [0, 1]
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = x.view(-1, 28, 28)
        x = self.scanner(x)
        return self.classifier(x)

def run_experiment():
    print(f"\n--- EXPERIMENTO V183: TEMPLATE RESONANCE (PENALIZED ERRORS) ---")
    
    transform = transforms.Compose([transforms.ToTensor()]) # Sin normalización para mantener [0, 1]
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    model = ResonanceNet(k_per_line=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Red de Resonancia Lista (Aciertos suman, Errores penalizan).")
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
    print(f"RESULTADO RESONANCIA DE PLANTILLA (V183)")
    print(f"="*55)
    print(f"Precisión Test:  {test_acc:.2f}%")
    print(f"Mecánica:        Template Matching (Correct adds, Error penalizes)")
    print(f"Tiempo Total:    {wall_time:.2f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v183_template_resonance.json", "w") as f:
        json.dump({"accuracy": test_acc, "params": total_params}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
