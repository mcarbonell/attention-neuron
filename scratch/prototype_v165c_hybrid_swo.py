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

# --- CLUSTERING (Para Inicialización) ---
def get_initial_clanes(feats, k=4096):
    print(f"Generando instinto inicial ({k} clanes)...")
    centers = feats[torch.randperm(feats.shape[0])[:k]]
    for _ in range(3):
        chunk_size = 5000
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(k, device=device)
        for j in range(0, feats.shape[0], chunk_size):
            end = min(j+chunk_size, feats.shape[0])
            batch = feats[j:end]
            sims = torch.mm(batch, centers.t())
            assignments = torch.argmax(sims, dim=1)
            new_centers.index_add_(0, assignments, batch)
            counts.index_add_(0, assignments, torch.ones(batch.size(0), device=device))
        centers = F.normalize(new_centers / (counts.unsqueeze(1) + 1e-8), p=2, dim=1)
    return centers

# --- OPTIMIZADOR SWO ---
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
                    h, w = p.shape if p.dim() == 2 else (1, p.shape[0])
                    kh, kw = max(2, int(h * group['k_ratio'])), max(2, int(w * group['k_ratio']))
                    state['k_size'] = (kh, kw)
                    state['exp_avg'] = torch.zeros((1, 1, kh, kw), device=p.device)
                    state['exp_avg_sq'] = torch.zeros((1, 1, kh, kw), device=p.device)

                state['step'] += 1
                g = p.grad.clamp(-0.1, 0.1).view(1, 1, *p.shape)
                if p.dim() == 1: g = g.unsqueeze(-2)
                
                g_small = F.interpolate(g, size=state['k_size'], mode='bilinear', align_corners=True)
                state['exp_avg'].mul_(group['betas'][0]).add_(g_small, alpha=1-group['betas'][0])
                state['exp_avg_sq'].mul_(group['betas'][1]).addcmul_(g_small, g_small, value=1-group['betas'][1])
                
                bc1, bc2 = 1 - group['betas'][0]**state['step'], 1 - group['betas'][1]**state['step']
                m = F.interpolate(state['exp_avg'], size=p.shape if p.dim()==2 else (1, p.shape[0]), mode='bilinear', align_corners=True).view_as(p)
                v = F.interpolate(state['exp_avg_sq'], size=p.shape if p.dim()==2 else (1, p.shape[0]), mode='bilinear', align_corners=True).view_as(p).clamp(min=0)
                
                denom = (v.sqrt() / np.sqrt(bc2)).add(group['eps'])
                p.addcdiv_(m, denom, value=-group['lr'] / bc1)

# --- MODELO HÍBRIDO ---
class HybridSpectralMoE(nn.Module):
    def __init__(self, dim, initial_clanes, initial_classifier):
        super().__init__()
        self.signatures = nn.Parameter(initial_clanes.clone())
        self.classifier = nn.Parameter(initial_classifier.clone())

    def forward(self, x):
        x_spec = fwht(x)
        # Resonancia (Normalized dot product)
        sig_norm = F.normalize(self.signatures, p=2, dim=1)
        scores = torch.matmul(x_spec, sig_norm.t())
        
        top_scores, top_indices = torch.topk(scores, k=16, dim=1)
        # Activación nítida para aprovechar el fine-tuning
        weights = F.softmax(top_scores * 30.0, dim=1)
        
        expert_outputs = self.classifier[top_indices]
        return (expert_outputs * weights.unsqueeze(-1)).sum(dim=1)

def run_hybrid_experiment():
    print(f"\n--- EXPERIMENTO V165c: HÍBRIDO INSTINTO + SWO ---")
    
    # 1. DATOS + CLUSTERING INICIAL
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)

    # Obtenemos clanes y sus etiquetas mayoritarias para el "instinto"
    print("Preparando instinto inicial...")
    sample_idx = torch.randperm(60000)[:20000]
    sample_data = train_ds.data[sample_idx].float().to(device) / 255.0
    x_sample = torch.zeros(len(sample_idx), 1024, device=device)
    x_sample[:, :784] = sample_data.view(len(sample_idx), -1)
    w_sample = F.normalize(fwht(x_sample), p=2, dim=1)
    
    clanes = get_initial_clanes(w_sample, k=4096)
    
    # Pre-calculamos el clasificador por instinto
    sims = torch.mm(w_sample, clanes.t())
    assigns = torch.argmax(sims, dim=1)
    targets = train_ds.targets[sample_idx]
    
    initial_v = torch.zeros(4096, 10, device=device)
    for i in range(4096):
        mask = assigns == i
        if mask.any():
            counts = torch.bincount(targets[mask], minlength=10).float()
            initial_v[i] = counts / (counts.sum() + 1e-8)
        else:
            initial_v[i] = torch.ones(10) / 10.0

    # 2. MODELO HÍBRIDO
    model = HybridSpectralMoE(1024, clanes, initial_v).to(device)
    optimizer = SmoothSpectralAdam(model.parameters(), lr=0.002, k_ratio=0.2)
    criterion = nn.CrossEntropyLoss()

    # 3. FINE-TUNING
    print("Iniciando Fine-tuning del instinto con SWO...")
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
        
        if i % 100 == 0:
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
    print(f"RESULTADO HÍBRIDO (V165c)")
    print(f"="*60)
    print(f"Precisión Final:   {acc*100:.2f}%")
    print(f"Tiempo Total:      {dt:.2f}s")
    print("="*60)

if __name__ == "__main__":
    run_hybrid_experiment()
