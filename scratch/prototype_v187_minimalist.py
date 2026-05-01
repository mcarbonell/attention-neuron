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

# --- GENERADOR DE MATRIZ DCT ---
def get_dct_matrix(n, device_target):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                matrix[i, j] = np.sqrt(1/n)
            else:
                matrix[i, j] = np.sqrt(2/n) * np.cos(np.pi * i * (2*j + 1) / (2*n))
    return torch.tensor(matrix, dtype=torch.float32).to(device_target)

# --- CAPA DCT (PARA CLASIFICACIÓN MINIMALISTA) ---
class DCTClassifierLayer(nn.Module):
    def __init__(self, in_features, out_features, k_spectral=16):
        super().__init__()
        self.k = k_spectral
        self.in_features = in_features
        
        # Semillas espectrales (Muy pocos parámetros)
        self.spectral_params = nn.Parameter(torch.randn(out_features, k_spectral) * 0.1)
        
        # Matriz DCT base
        self.register_buffer('dct_mat', get_dct_matrix(max(in_features, k_spectral), device))
        
    def forward(self, x):
        # Sintetizamos los pesos: (Out, K) @ (K, In) -> (Out, In)
        w = torch.matmul(self.spectral_params, self.dct_mat[:self.k, :self.in_features])
        return F.linear(x, w)

# --- ESCÁNER DE RESONANCIA PURA (DE V186) ---
class PureResonanceLayer(nn.Module):
    def __init__(self, k_per_line=10):
        super().__init__()
        self.k = k_per_line
        freqs = torch.linspace(0.05, 0.5, k_per_line).repeat(28, 1)
        self.row_freq = nn.Parameter(freqs)
        self.row_phase = nn.Parameter(torch.zeros(28, k_per_line))
        self.col_freq = nn.Parameter(freqs.clone())
        self.col_phase = nn.Parameter(torch.zeros(28, k_per_line))
        self.register_buffer('pos', torch.linspace(1, 28, 28))
        
    def forward(self, x):
        batch_size = x.size(0)
        row_phases = self.pos.view(1, 28, 1) * self.row_freq.view(28, 1, self.k) + self.row_phase.view(28, 1, self.k)
        r_template = torch.sin(row_phases)
        row_res = (x.unsqueeze(-1) * r_template.unsqueeze(0)).sum(dim=2)
        
        x_t = x.transpose(1, 2)
        col_phases = self.pos.view(1, 28, 1) * self.col_freq.view(28, 1, self.k) + self.col_phase.view(28, 1, self.k)
        c_template = torch.sin(col_phases)
        col_res = (x_t.unsqueeze(-1) * c_template.unsqueeze(0)).sum(dim=2)
        
        return torch.cat([row_res.view(batch_size, -1), col_res.view(batch_size, -1)], dim=1)

# --- MODELO ESPECTRAL MINIMALISTA (V187) ---
class MinimalistSpectralNet(nn.Module):
    def __init__(self, k_per_line=10):
        super().__init__()
        self.scanner = PureResonanceLayer(k_per_line)
        self.bn = nn.BatchNorm1d(28 * 2 * k_per_line)
        # Clasificador DCT: Proyectamos 560 features a 10 clases con solo 16 semillas
        self.classifier = DCTClassifierLayer(in_features=28 * 2 * k_per_line, out_features=10, k_spectral=16)
        
    def forward(self, x):
        x = x.view(-1, 28, 28)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = self.scanner(x)
        x = self.bn(x)
        # Clasificamos directamente con pesos sintetizados
        return self.classifier(x)

def run_experiment():
    print(f"\n--- EXPERIMENTO V187: MINIMALIST SPECTRAL RESONANCE ---")
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    model = MinimalistSpectralNet(k_per_line=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Red Espectral Minimalista Lista.")
    print(f"PARÁMETROS TOTALES: {total_params} (¡Compresión de Élite!)")

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
    print(f"RESULTADO MINIMALISTA (V187)")
    print(f"="*55)
    print(f"Precisión Test:  {test_acc:.2f}%")
    print(f"Parámetros:      {total_params}")
    print(f"Tiempo Total:    {wall_time:.2f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v187_minimalist.json", "w") as f:
        json.dump({"accuracy": test_acc, "params": total_params}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
