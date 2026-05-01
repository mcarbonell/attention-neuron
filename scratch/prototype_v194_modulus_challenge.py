import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
except ImportError:
    pass

# --- GENERADORES DE DATOS ---
class ModulusBenchmark:
    def __init__(self, name="mod", dim=2):
        self.name = name
        self.dim = dim
        
    def get_ranges(self):
        # x, y. y must be > 0
        return [0.0, 5.0], [0.0, 10.0], [0.0, 20.0]

    def evaluate(self, x):
        # x % y
        val = x[:, 0]
        mod = torch.clamp(torch.abs(x[:, 1]), min=0.5) # Evitar mod por valores muy pequeños
        return torch.remainder(val, mod).unsqueeze(1)

    def generate_data(self, n_samples=3000):
        r_train, r_near, r_far = self.get_ranges()
        x_train = torch.empty(n_samples, self.dim).uniform_(*r_train)
        x_near = torch.empty(n_samples, self.dim).uniform_(*r_near)
        x_far = torch.empty(n_samples, self.dim).uniform_(*r_far)
        return {
            "train": (x_train.to(device), self.evaluate(x_train).to(device)),
            "near": (x_near.to(device), self.evaluate(x_near).to(device)),
            "far": (x_far.to(device), self.evaluate(x_far).to(device))
        }

# --- MODELOS (Misma arquitectura V193) ---

class ResonanceLayer(nn.Module):
    def __init__(self, in_dim, k_oscillators=16): # Más osciladores para capturar discontinuidades
        super().__init__()
        self.freq = nn.Parameter(torch.randn(in_dim, k_oscillators) * 5.0)
        self.phase = nn.Parameter(torch.randn(in_dim, k_oscillators) * math.pi)
    def forward(self, x):
        x_un = x.unsqueeze(-1)
        phases = x_un * self.freq + self.phase
        return torch.sin(phases).view(x.size(0), -1)

class LogInteractionLayer(nn.Module):
    def __init__(self, in_dim, out_dim=8):
        super().__init__()
        self.log_weights = nn.Linear(in_dim, out_dim)
        self.sign_net = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        x_abs = torch.abs(x) + 1e-6
        z_mag = torch.exp(self.log_weights(torch.log(x_abs)))
        z_sign = torch.tanh(self.sign_net(torch.sign(x)))
        return z_mag * z_sign

class PolymorphicStage(nn.Module):
    def __init__(self, in_dim, out_dim=32):
        super().__init__()
        self.res = ResonanceLayer(in_dim, k_oscillators=16)
        self.log = LogInteractionLayer(in_dim, out_dim=8)
        self.total_in = in_dim * 16 + 8 + in_dim * 8
        self.integrator = nn.Linear(self.total_in, out_dim)
        
    def forward(self, x):
        res_f = self.res(x)
        log_f = self.log(x)
        b1, b2, b3, b4 = x, x**2, x**3, torch.abs(x)
        b5, b6, b7 = torch.sin(x), torch.cos(x), 1.0 / (x + 1e-8)
        x_nz = torch.where(x == 0, torch.ones_like(x)*1e-6, x)
        b8 = torch.sin(x_nz) / x_nz
        poly_f = torch.cat([b1, b2, b3, b4, b5, b6, b7, b8], dim=1)
        combined = torch.cat([res_f, log_f, poly_f], dim=1)
        return torch.tanh(self.integrator(combined))

class DeepPolymorphicNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, depth=2):
        super().__init__()
        self.stages = nn.ModuleList()
        current_dim = in_dim
        for i in range(depth):
            self.stages.append(PolymorphicStage(current_dim, hidden_dim))
            current_dim = hidden_dim
        self.head = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return self.head(x)

class BaselineMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, layers=4): # MLP MUY fuerte
        super().__init__()
        model = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(layers - 1):
            model.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        model.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*model)
    def forward(self, x): return self.net(x)

# --- ENGINE ---

def run_modulus_challenge():
    bench = ModulusBenchmark()
    print("\n>>> INICIANDO MODULUS CHALLENGE (x % y)")
    
    models = [
        ("MLP-Huge", BaselineMLP(2, 512, 5)), # MLP masivo para intentar compensar discontinuidad
        ("Poly-Deep-V193", DeepPolymorphicNet(2, 32, 2))
    ]
    
    for m_name, model in models:
        model.to(device)
        p_count = sum(p.numel() for p in model.parameters())
        print(f"\nTraining {m_name} ({p_count} params)...")
        
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        data = bench.generate_data()
        x_train, y_train = data["train"]
        x_far, y_far = data["far"]
        
        t0 = time.time()
        for epoch in range(5000): # Más épocas para esta tarea difícil
            optimizer.zero_grad()
            pred = model(x_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()
            
            if epoch % 1000 == 0:
                print(f"  Epoch {epoch} | Loss: {loss.item():.2e}")
                
        model.eval()
        with torch.no_grad():
            m_train = criterion(model(x_train), y_train).item()
            m_far = criterion(model(x_far), y_far).item()
            
        print(f"  Final Train MSE: {m_train:.4e}")
        print(f"  Final Far OOD MSE: {m_far:.4e}")
        print(f"  Stability Ratio: {m_far / (m_train + 1e-12):.2e}")

if __name__ == "__main__":
    run_modulus_challenge()
