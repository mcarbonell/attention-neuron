import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import json

# --- CONFIGURACIÓN DE DISPOSITIVO ---
device = torch.device('cpu')
try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using DirectML device: {device}")
except ImportError:
    print("torch-directml not found, using CPU")

# --- TRANSFORMADA DE WALSH-HADAMARD RÁPIDA (FWHT) ---
def fwht(x):
    """ Fast Walsh-Hadamard Transform vectorizada """
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

# --- CLASIFICADOR HOLOGRÁFICO ESPECTRAL ---
class HolographicClassifier(nn.Module):
    def __init__(self, pattern_dim, num_memories):
        super().__init__()
        # Usamos register_buffer para que no se consideren parámetros entrenables
        # Son recuerdos fijos, no pesos que se ajustan por gradiente.
        self.register_buffer('memory_bank', torch.zeros(num_memories, pattern_dim))
        self.register_buffer('memory_labels', torch.zeros(num_memories, dtype=torch.long))
        
    def inject_memories(self, loader, device):
        """ Copia el set de entrenamiento directamente al dominio espectral """
        idx = 0
        for data, target in loader:
            batch_size = data.size(0)
            data = data.view(batch_size, -1)
            # Pad 28x28 (784) -> 32x32 (1024) para FWHT
            data_padded = torch.zeros(batch_size, 1024).to(device)
            data_padded[:, :784] = data.to(device)
            
            # Transformar y guardar 'recuerdo'
            spec = fwht(data_padded)
            self.memory_bank[idx:idx+batch_size] = spec
            self.memory_labels[idx:idx+batch_size] = target.to(device)
            idx += batch_size

    def forward(self, x):
        batch_size = x.size(0)
        # Pad entrada
        x_padded = torch.zeros(batch_size, 1024).to(x.device)
        x_padded[:, :784] = x.view(batch_size, -1)
        
        # 1. Transformar entrada al dominio espectral
        x_spec = fwht(x_padded)
        
        # 2. Búsqueda Asociativa Masiva (Holográfica)
        # Comparamos la entrada con los 60,000 recuerdos en un solo paso
        similarities = torch.matmul(x_spec, self.memory_bank.t())
        
        # 3. Clasificación por resonancia máxima
        best_match_indices = torch.argmax(similarities, dim=1)
        return self.memory_labels[best_match_indices]

def run_experiment():
    # 1. Carga de datos
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=2000)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)

    # 2. Inicializar Modelo (60k recuerdos de 1024p)
    model = HolographicClassifier(1024, 60000).to(device)
    
    # 3. FASE DE APRENDIZAJE: Inyección (0 Épocas)
    print("\nFase de Inyección: Guardando 60,000 imágenes en la memoria espectral...")
    t0 = time.perf_counter()
    model.inject_memories(train_loader, device)
    t_inject = time.perf_counter() - t0
    print(f"Inyección completada en {t_inject:.2f} segundos.")

    # 4. FASE DE EVALUACIÓN: Test
    print(f"Evaluando precisión en 10,000 imágenes de test...")
    correct = 0
    t_start = time.perf_counter()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            preds = model(data)
            correct += preds.eq(target).sum().item()
            
    t_eval = time.perf_counter() - t_start
    accuracy = correct / 10000
    
    print("\n" + "="*55)
    print(f"RESULTADOS DEL CLASIFICADOR HOLOGRÁFICO MNIST")
    print(f"="*55)
    print(f"Precisión Final: {accuracy*100:.2f}%")
    print(f"Tiempo de Inyección (Aprendizaje): {t_inject:.2f} s")
    print(f"Tiempo de Inferencia (10k imágenes): {t_eval:.2f} s")
    print(f"Latencia por imagen: {(t_eval/10000)*1000:.4f} ms")
    print(f"Entrenamiento Requerido: 0 Épocas (Zero-Backprop)")
    print("="*55)

    # Guardar resultados
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v139_holographic_mnist.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "inject_time": t_inject,
            "test_time": t_eval,
            "model": "HolographicClassifier_V139"
        }, f, indent=4)

if __name__ == "__main__":
    run_experiment()
