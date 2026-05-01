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

# --- CAPA DE VOTACIÓN POR RESONANCIA (V188) ---
class VotingResonanceLayer(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        # 10 grupos de especialistas. Cada grupo tiene (28 filas + 28 cols) osciladores.
        # Total osciladores: 560. Parámetros: 1120.
        
        # Frecuencias y Fases: (Clase, Línea_idx)
        # Inicialización ordenada de 0.05 a 0.5
        init_freqs = torch.linspace(0.05, 0.5, 28).repeat(num_classes, 2) # (10, 56)
        self.freqs = nn.Parameter(init_freqs)
        self.phases = nn.Parameter(torch.zeros(num_classes, 56))
        
        self.register_buffer('pos', torch.linspace(1, 28, 28))
        
    def forward(self, x):
        # x: (B, 28, 28)
        batch_size = x.size(0)
        
        # Preparamos las fases para todas las combinaciones (B, Clase, Línea, Pos)
        # Pero para optimizar lo hacemos por partes
        
        scores = []
        for c in range(self.num_classes):
            # --- ESPECIALISTAS DE LA CLASE C ---
            # Filas (28)
            f_row = self.freqs[c, :28]
            p_row = self.phases[c, :28]
            
            # Matriz de senos para filas: (28 filas, 28 posiciones)
            row_templ = torch.sin(self.pos.view(1, 28) * f_row.view(28, 1) + p_row.view(28, 1))
            res_row = (x * row_templ.unsqueeze(0)).sum(dim=(1, 2)) # Suma total de resonancia en filas
            
            # Columnas (28)
            f_col = self.freqs[c, 28:]
            p_col = self.phases[c, 28:]
            x_t = x.transpose(1, 2)
            col_templ = torch.sin(self.pos.view(1, 28) * f_col.view(28, 1) + p_col.view(28, 1))
            res_col = (x_t * col_templ.unsqueeze(0)).sum(dim=(1, 2)) # Suma total de resonancia en columnas
            
            # La puntuación de la clase C es la SUMA LIMPIA de sus resonancias
            scores.append(res_row + res_col)
            
        return torch.stack(scores, dim=1) # (B, 10)

# --- MODELO VOTACIÓN PURA ---
class VotingResonanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.voter = VotingResonanceLayer()
        # No hay MLP. No hay capas ocultas. Solo la votación.
        
    def forward(self, x):
        x = x.view(-1, 28, 28)
        # Normalización a [0, 1]
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        # Votación directa
        logits = self.voter(x)
        return logits

def run_experiment():
    print(f"\n--- EXPERIMENTO V188: VOTING RESONANCE (PURE PHYSICAL CLASSIFIER) ---")
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    model = VotingResonanceNet().to(device)
    # Entrenamos solo las 1120 variables de afinación
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Cerebro de Votación por Resonancia Listo.")
    print(f"PARÁMETROS TOTALES: {total_params} (La esencia pura)")

    model.train()
    t0 = time.perf_counter()
    for epoch in range(1, 11): # Más épocas porque es un modelo muy rígido
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
    print(f"RESULTADO VOTACIÓN PURA (V188)")
    print(f"="*55)
    print(f"Precisión Test:  {test_acc:.2f}%")
    print(f"Parámetros:      {total_params}")
    print(f"Tiempo Total:    {wall_time:.2f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v188_voting.json", "w") as f:
        json.dump({"accuracy": test_acc, "params": total_params}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
