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

# --- FUNCIÓN SINC (SENO CARDINAL) ---
def sinc(x):
    """ sinc(x) = sin(x)/x, con límite en 0 """
    # Pequeño epsilon para evitar división por cero
    eps = 1e-8
    return torch.sin(x) / (x + eps)

# --- CAPA SINFÓNICA TRIGONOMÉTRICA ---
class TrigSymphonyLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        # Frecuencia (Pesos), Fase (Bias), Amplitud (Scale)
        self.freq = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.phase = nn.Parameter(torch.randn(out_features) * 2 * np.pi)
        self.amp = nn.Parameter(torch.ones(out_features) * 0.1)
        
        # Repartimos las voces: 1/3 Sin, 1/3 Cos, 1/3 Sinc
        self.group_size = out_features // 3
        
    def forward(self, x):
        # Proyección Lineal (Z = Wx + b)
        # Aquí W actúa como frecuencia y b como fase
        z = F.linear(x, self.freq, self.phase)
        
        # Aplicamos las funciones trigonométricas
        out = torch.zeros_like(z)
        
        # Grupo 1: SENO
        out[:, :self.group_size] = torch.sin(z[:, :self.group_size])
        
        # Grupo 2: COSENO
        out[:, self.group_size:2*self.group_size] = torch.cos(z[:, self.group_size:2*self.group_size])
        
        # Grupo 3: SINC (Foco Local)
        out[:, 2*self.group_size:] = sinc(z[:, 2*self.group_size:])
        
        # Escalamos por la amplitud aprendida
        return out * self.amp

# --- MODELO SINFÓNICO ---
class HarmonicNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Una sola capa oculta de 256 osciladores
        self.harmonic_bank = TrigSymphonyLayer(784, 256)
        # Capa de clasificación final
        self.classifier = nn.Linear(256, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.harmonic_bank(x)
        return self.classifier(x)

def run_experiment():
    print(f"\n--- EXPERIMENTO V171: TRIGONOMETRIC SYMPHONY (HARMONIC RESONANCE) ---")
    
    # 1. CARGA DE DATOS
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=1000, shuffle=False)

    model = HarmonicNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    print(f"Instrumento listo: 256 Osciladores (Sin/Cos/Sinc).")
    print(f"Parámetros Entrenables: {sum(p.numel() for p in model.parameters())}")

    # 2. ENTRENAMIENTO (5 épocas para ver la música)
    model.train()
    t0 = time.perf_counter()
    for epoch in range(1, 6):
        total_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            if batch_idx == 0: # Fast Feedback (Rule 58)
                print(f"Época {epoch} [Batch 0] - Loss: {loss.item():.4f}")

        acc = 100. * correct / len(train_loader.dataset)
        print(f"Fin Época {epoch} - Accuracy Train: {acc:.2f}%")

    wall_time = time.perf_counter() - t0

    # 3. EVALUACIÓN FINAL
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
    print(f"RESULTADO SINFONÍA TRIGONOMÉTRICA (V171)")
    print(f"="*55)
    print(f"Precisión Test:  {test_acc:.2f}%")
    print(f"Mecánica:        Osciladores Armónicos (Sin, Cos, Sinc)")
    print(f"Tiempo Total:    {wall_time:.2f}s")
    print("="*55)

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v171_trig_symphony.json", "w") as f:
        json.dump({"accuracy": test_acc, "wall_time": wall_time}, f, indent=4)

if __name__ == "__main__":
    run_experiment()
