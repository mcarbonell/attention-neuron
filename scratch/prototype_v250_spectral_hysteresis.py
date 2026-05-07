import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import time
import json
import os
import math

# --- CONFIGURACIÓN DE DISPOSITIVO ---
device = torch.device('cpu')
try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using DirectML device: {device}")
except ImportError:
    print("torch-directml not found, using CPU")

# --- UTILIDADES ESPECTRALES ---
def dct_2d(x):
    """Transformada de Coseno Discreta 2D básica para tensores [B, 1, 28, 28]"""
    # Usamos una aproximación simple vía FFT o simplemente aplanamos si el objetivo es eficiencia paramétrica
    # Para este prototipo, trabajaremos en el dominio de la intensidad aplanada (784) 
    # y aplicaremos una matriz de Walsh-Hadamard fija para simular el dominio espectral.
    return x

def fwht(x):
    """Fast Walsh-Hadamard Transform"""
    b, n = x.shape
    # N debe ser potencia de 2
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

# --- MODELOS ---

class HysteresisSpectralLayer(nn.Module):
    def __init__(self, dim, alpha=0.9, beta=0.5):
        super().__init__()
        self.dim = dim
        self.alpha = alpha # Momentum de la memoria (EMA)
        self.beta = beta   # Fuerza de la supresión de fondo (novelty)
        
        # Estado persistente (no es un parámetro, es un buffer)
        self.register_buffer('memory', torch.zeros(1, dim))
        
        # Pesos de la neurona
        self.weight = nn.Parameter(torch.randn(dim, 10) * 0.01)
        self.bias = nn.Parameter(torch.zeros(10))
        
    def forward(self, x, update_memory=True):
        # 1. Transformada Espectral (Walsh-Hadamard)
        x_spec = fwht(x)
        
        # 2. Hysteresis Logic
        # Usamos la memoria del ejemplo anterior (o del batch anterior)
        # Delta = Lo que este ejemplo tiene que NO estaba en el promedio reciente
        x_delta = x_spec - self.beta * self.memory
        
        # 3. Clasificación
        out = torch.mm(x_delta, self.weight) + self.bias
        
        # 4. Actualización de Memoria (EMA)
        if update_memory:
            with torch.no_grad():
                batch_mean = x_spec.mean(dim=0, keepdim=True)
                self.memory.copy_(self.alpha * self.memory + (1 - self.alpha) * batch_mean)
                
        return out

class BaselineMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, 10) * 0.01)
        self.bias = nn.Parameter(torch.zeros(10))
        
    def forward(self, x):
        # Directo en espacio de píxeles (o espectral pero sin memoria)
        x_spec = fwht(x)
        return torch.mm(x_spec, self.weight) + self.bias

# --- EXPERIMENTO ---

def train_and_eval(model, train_loader, test_loader, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    model.train()
    
    # Métricas de primer impacto (Regla de Supervivencia)
    first_batches_loss = []
    
    for epoch in range(epochs):
        correct = 0
        total = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.view(-1, 1024).to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if epoch == 0 and i < 5:
                first_batches_loss.append(loss.item())
                print(f"      Batch {i} Loss: {loss.item():.4f}")
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        print(f"    Epoch {epoch+1} Accuracy: {100. * correct / total:.2f}%")
        
    wall_clock_time = time.time() - start_time
    
    # Eval
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.view(-1, 1024).to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_acc = 100. * test_correct / len(test_loader.dataset)
    return test_acc, wall_clock_time, first_batches_loss

def main():
    print("\n" + "="*70)
    print("PROTOTYPE V250: SPECTRAL HYSTERESIS NEURON (STATEFUL)")
    print("="*70 + "\n")

    # Preparar Datos (MNIST con padding a 32x32 para FWHT)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Pad(2) # 28x28 -> 32x32
    ])
    
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # --- ESCENARIO 1: RANDOM ORDER (DIFÍCIL PARA MEMORIA) ---
    train_loader_rand = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False)
    
    # --- ESCENARIO 2: CLUSTERED ORDER (IDEAL PARA MEMORIA) ---
    # Ordenamos el dataset por etiqueta para simular "persistencia de concepto"
    indices = torch.argsort(train_ds.targets)
    train_ds_clustered = torch.utils.data.Subset(train_ds, indices)
    train_loader_cluster = torch.utils.data.DataLoader(train_ds_clustered, batch_size=64, shuffle=False)

    results = {}

    for name, loader in [("Random Order", train_loader_rand), ("Clustered Order", train_loader_cluster)]:
        print(f"\n>>> Running Scenario: {name}")
        
        # 1. Baseline
        print("  Testing Baseline MLP...")
        model_b = BaselineMLP(1024).to(device)
        acc_b, time_b, _ = train_and_eval(model_b, loader, test_loader)
        
        # 2. Hysteresis
        print("  Testing Hysteresis Spectral Neuron...")
        model_h = HysteresisSpectralLayer(1024, alpha=0.95, beta=0.5).to(device)
        acc_h, time_h, _ = train_and_eval(model_h, loader, test_loader)
        
        # PEI = Accuracy / log10(Params + 1)
        params_b = sum(p.numel() for p in model_b.parameters())
        params_h = sum(p.numel() for p in model_h.parameters())
        pei_b = acc_b / math.log10(params_b + 1)
        pei_h = acc_h / math.log10(params_h + 1)
        
        results[name] = {
            "baseline": {"acc": acc_b, "time": time_b, "pei": pei_b},
            "hysteresis": {"acc": acc_h, "time": time_h, "pei": pei_h}
        }
        
        print(f"\n    Comparison for {name}:")
        print(f"      Baseline:   {acc_b:.2f}% Acc | {pei_b:.2f} PEI")
        print(f"      Hysteresis: {acc_h:.2f}% Acc | {pei_h:.2f} PEI")
        print(f"      Delta Acc:  {acc_h - acc_b:+.2f}%")

    # Guardar resultados
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v250_spectral_hysteresis.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
