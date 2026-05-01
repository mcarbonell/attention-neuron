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

# --- CAPA DE VOTACIÓN POR RESONANCIA DE ALTA RESOLUCIÓN (V189) ---
class HighResVotingLayer(nn.Module):
    def __init__(self, num_classes=10, k_per_line=3):
        super().__init__()
        self.num_classes = num_classes
        self.k = k_per_line
        
        # 10 clases. Cada clase tiene 56 líneas (28 rows + 28 cols).
        # Cada línea tiene K osciladores.
        # Total parámetros: 10 * 56 * K * 2
        
        # Inicialización de frecuencias armónicas (0.1, 0.2, 0.3...)
        init_freqs = torch.linspace(0.05, 0.5, k_per_line).view(1, 1, k_per_line).repeat(num_classes, 56, 1)
        self.freqs = nn.Parameter(init_freqs)
        self.phases = nn.Parameter(torch.zeros(num_classes, 56, k_per_line))
        
        self.register_buffer('pos', torch.linspace(1, 28, 28))
        
    def forward(self, x):
        # x: (B, 28, 28)
        batch_size = x.size(0)
        
        scores = []
        for c in range(self.num_classes):
            # --- ESPECIALISTAS DE LA CLASE C ---
            # Frecuencias y Fases de la clase C: (56, K)
            f_c = self.freqs[c] 
            p_c = self.phases[c]
            
            # FILAS (28 líneas x K osciladores)
            # pos: (28), f_row: (28, K), p_row: (28, K)
            # row_templ: (28_lineas, 28_posiciones, K_osciladores)
            row_templ = torch.sin(self.pos.view(1, 28, 1) * f_c[:28].view(28, 1, self.k) + p_c[:28].view(28, 1, self.k))
            # x: (B, 28, 28) -> (B, 28, 28, 1)
            # Resonancia: sumamos sobre posiciones y sobre los K osciladores de cada línea
            res_row = (x.unsqueeze(-1) * row_templ.unsqueeze(0)).sum(dim=(1, 2, 3))
            
            # COLUMNAS (28 líneas x K osciladores)
            x_t = x.transpose(1, 2)
            col_templ = torch.sin(self.pos.view(1, 28, 1) * f_c[28:].view(28, 1, self.k) + p_c[28:].view(28, 1, self.k))
            res_col = (x_t.unsqueeze(-1) * col_templ.unsqueeze(0)).sum(dim=(1, 2, 3))
            
            scores.append(res_row + res_col)
            
        return torch.stack(scores, dim=1)

# --- MODELO VOTACIÓN ALTA RESOLUCIÓN ---
class HighResVotingNet(nn.Module):
    def __init__(self, k_per_line=3):
        super().__init__()
        self.voter = HighResVotingLayer(k_per_line=k_per_line)
        
    def forward(self, x):
        x = x.view(-1, 28, 28)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return self.voter(x)

def run_experiment():
    print(f"\n--- EXPERIMENTO V189: HIGH-RES VOTING RESONANCE ---")
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    # K=3 osciladores por línea (3360 parámetros)
    model = HighResVotingNet(k_per_line=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Cerebro de Votación de Alta Resolución Listo (K=3).")
    print(f"PARÁMETROS TOTALES: {total_params}")

    model.train()
    t0 = time.perf_counter()
    for epoch in range(1, 11):
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
    print(f"RESULTADO VOTACIÓN ALTA RESOLUCIÓN (V189)")
    print(f"="*55)
    print(f"Precisión Test:  {test_acc:.2f}%")
    print(f"Parámetros:      {total_params}")
    print(f"Tiempo Total:    {wall_time:.2f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v189_high_res_voting.json", "w") as f:
        json.dump({"accuracy": test_acc, "params": total_params}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
