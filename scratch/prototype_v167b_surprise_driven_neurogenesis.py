import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
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

# --- FWHT ---
def fwht(x):
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

# --- MODELO DE NEUROGÉNESIS DINÁMICA ---
class SurpriseMoE:
    def __init__(self, dim=1024, max_experts=10000, surprise_threshold=0.7):
        self.dim = dim
        self.max_experts = max_experts
        self.threshold = surprise_threshold
        # Empezamos con una semilla aleatoria
        self.signatures = torch.randn(1, dim, device=device)
        self.labels = torch.zeros(1, dtype=torch.long, device=device)
        self.num_experts = 1

    def process_sample(self, x_spec, target):
        # Medir Resonancia
        sims = torch.mm(x_spec, F.normalize(self.signatures, p=2, dim=1).t())
        max_sim, _ = torch.max(sims, dim=1)
        
        # Neurogénesis si hay sorpresa
        if max_sim < self.threshold and self.num_experts < self.max_experts:
            self.signatures = torch.cat([self.signatures, x_spec], dim=0)
            self.labels = torch.cat([self.labels, target.view(1)], dim=0)
            self.num_experts += 1
            return True
        return False

    def predict(self, x_spec):
        # Búsqueda holográfica rápida
        sims = torch.mm(x_spec, F.normalize(self.signatures, p=2, dim=1).t())
        best_idx = torch.argmax(sims, dim=1)
        return self.labels[best_idx]

def run_neurogenesis_experiment():
    print(f"\n--- EXPERIMENTO V167b: NEUROGÉNESIS DIRIGIDA POR SORPRESA ---")
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    
    # Umbral de sorpresa 0.8 (Exigente)
    model = SurpriseMoE(dim=1024, max_experts=8192, surprise_threshold=0.8)
    
    print(f"Creciendo sobre 10,000 muestras (Umbral Sorpresa: 0.8)...")
    t0 = time.perf_counter()
    
    for i in range(10000):
        img, target = train_ds[i]
        img_padded = torch.zeros(1, 1024, device=device)
        img_padded[0, :784] = img.view(-1)
        x_spec = F.normalize(fwht(img_padded), p=2, dim=1)
        
        target_tensor = torch.tensor([target], device=device)
        model.process_sample(x_spec, target_tensor)
        
        if (i+1) % 2000 == 0:
            print(f"  Paso {i+1} | Población de Expertos: {model.num_experts}")

    dt_grow = time.perf_counter() - t0
    
    print("\nEvaluando en 2,000 imágenes de test...")
    correct = 0
    for i in range(2000):
        img, target = test_ds[i]
        img_padded = torch.zeros(1, 1024, device=device)
        img_padded[0, :784] = img.view(-1)
        x_spec = F.normalize(fwht(img_padded), p=2, dim=1)
        
        pred = model.predict(x_spec)
        if pred.item() == target: correct += 1
        
    acc = correct / 2000
    
    print("\n" + "="*60)
    print(f"RESULTADO NEUROGÉNESIS (V167b)")
    print(f"="*60)
    print(f"Expertos Finales:   {model.num_experts}")
    print(f"Precisión Zero-Shot: {acc*100:.2f}%")
    print(f"Tiempo Total:       {dt_grow:.2f}s")
    print("="*60)

    # Guardar resultados
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v167b_neurogenesis.json", "w") as f:
        json.dump({"num_experts": model.num_experts, "accuracy": acc}, f, indent=4)

if __name__ == "__main__":
    run_neurogenesis_experiment()
