import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a, b_ = res[:, :, 0, :], res[:, :, 1, :]
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n ** 0.5)

# --- OPTIMIZADOR SWO (Smooth Walsh Optimizer) ---
class SmoothSpectralAdam(optim.Optimizer):
    def __init__(self, params, lr=1e-3, k_ratio=0.25):
        defaults = dict(lr=lr, betas=(0.9, 0.999), eps=1e-8, k_ratio=k_ratio)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Comprimimos estados de Adam para ahorrar VRAM
                    h, w = p.shape if p.dim() == 2 else (1, p.shape[0])
                    kh, kw = max(2, int(h * group['k_ratio'])), max(2, int(w * group['k_ratio']))
                    state['k_size'] = (kh, kw)
                    state['exp_avg'] = torch.zeros((1, 1, kh, kw), device=p.device)
                    state['exp_avg_sq'] = torch.zeros((1, 1, kh, kw), device=p.device)

                state['step'] += 1
                # Normalizar gradiente para estabilidad en el dominio espectral
                grad = p.grad.clamp(-1.0, 1.0)
                g = grad.view(1, 1, *p.shape)
                if p.dim() == 1: g = g.unsqueeze(-2)
                
                # Update Low-Res States
                g_small = F.interpolate(g, size=state['k_size'], mode='bilinear', align_corners=True)
                state['exp_avg'].mul_(group['betas'][0]).add_(g_small, alpha=1-group['betas'][0])
                state['exp_avg_sq'].mul_(group['betas'][1]).addcmul_(g_small, g_small, value=1-group['betas'][1])
                
                # Reconstruct and Apply
                m = F.interpolate(state['exp_avg'], size=p.shape if p.dim()==2 else (1, p.shape[0]), mode='bilinear', align_corners=True).view_as(p)
                v = F.interpolate(state['exp_avg_sq'], size=p.shape if p.dim()==2 else (1, p.shape[0]), mode='bilinear', align_corners=True).view_as(p).clamp(min=0)
                
                # Bias correction
                bc1 = 1 - group['betas'][0]**state['step']
                bc2 = 1 - group['betas'][1]**state['step']
                
                denom = (v.sqrt() / np.sqrt(bc2)).add(group['eps'])
                p.addcdiv_(m, denom, value=-group['lr'] / bc1)

# --- MODELO SPECTRAL-MOE ENTRENABLE ---
class TrainableSpectralMoE(nn.Module):
    def __init__(self, dim, num_experts=4096, top_k=16):
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        # Firmas de los expertos
        self.signatures = nn.Parameter(torch.randn(num_experts, dim) * 0.05)
        # Capa de salida (Clasificador)
        self.classifier = nn.Parameter(torch.randn(num_experts, 10) * 0.01)

    def forward(self, x):
        x_spec = fwht(x)
        # Gating por resonancia (Normalizado para estabilidad)
        sig_norm = F.normalize(self.signatures, p=2, dim=1)
        scores = torch.matmul(x_spec, sig_norm.t())
        
        # Top-K Sparsity
        top_scores, top_indices = torch.topk(scores, k=self.top_k, dim=1)
        weights = F.softmax(top_scores * 10.0, dim=1)
        
        # Selección de expertos y suma ponderada
        # (Batch, TopK, 10)
        selected_experts = self.classifier[top_indices]
        output = (selected_experts * weights.unsqueeze(-1)).sum(dim=1)
        return output

def run_training_experiment():
    print(f"\n--- EXPERIMENTO V165b: ENTRENAMIENTO DE EXPERTOS CON SWO ---")
    
    # 1. DATOS
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)

    # 2. MODELO + OPTIMIZADOR SWO
    model = TrainableSpectralMoE(1024, num_experts=4096).to(device)
    # k_ratio=0.1 -> 100x menos memoria en estados 2D
    optimizer = SmoothSpectralAdam(model.parameters(), lr=0.01, k_ratio=0.1) 
    criterion = nn.CrossEntropyLoss()

    # 3. BUCLE DE ENTRENAMIENTO
    print("Entrenando firmas espectrales (1 época para test rápido)...")
    t0 = time.perf_counter()
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data = data.view(data.size(0), -1)
        x = torch.zeros(data.size(0), 1024, device=device)
        x[:, :784] = data.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"  Batch {i} Loss: {loss.item():.4f}")

    # 4. EVALUACIÓN
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1)
            x = torch.zeros(data.size(0), 1024, device=device)
            x[:, :784] = data.to(device)
            preds = model(x).argmax(dim=1)
            correct += preds.eq(target.to(device)).sum().item()
    
    acc = correct / 10000
    dt = time.perf_counter() - t0

    print("\n" + "="*60)
    print(f"RESULTADO ENTRENAMIENTO SWO (V165b)")
    print(f"="*60)
    print(f"Precisión Final:   {acc*100:.2f}%")
    print(f"Tiempo Total:      {dt:.2f}s")
    print(f"Compresión SWO:    10x (k=0.1)")
    print("="*60)

    # Guardar resultados
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v165b_swo_training.json", "w") as f:
        json.dump({"accuracy": acc, "time": dt}, f, indent=4)

if __name__ == "__main__":
    run_training_experiment()
