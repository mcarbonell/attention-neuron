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
class FunctionBenchmark:
    def __init__(self, name, dim=1):
        self.name = name
        self.dim = dim
        
    def get_ranges(self):
        if self.name == "1/x":
            return [0.5, 2.0], [0.2, 5.0], [0.1, 10.0]
        if "Rastrigin" in self.name or "Ackley" in self.name:
            return [-2.0, 2.0], [-5.12, 5.12], [-15.0, 15.0]
        return [-2.0, 2.0], [-4.0, 4.0], [-10.0, 10.0]

    def evaluate(self, x):
        if self.name == "Rastrigin":
            A = 10
            n = x.shape[1]
            return (A * n + torch.sum(x**2 - A * torch.cos(2 * math.pi * x), dim=1)).unsqueeze(1)
        if self.name == "Ackley":
            n = x.shape[1]
            sum_sq = torch.sum(x**2, dim=1)
            sum_cos = torch.sum(torch.cos(2 * math.pi * x), dim=1)
            term1 = -20 * torch.exp(-0.2 * torch.sqrt(sum_sq / n))
            term2 = -torch.exp(sum_cos / n)
            return (term1 + term2 + 20 + math.e).unsqueeze(1)
        if self.name == "Schwefel":
            n = x.shape[1]
            return (418.9829 * n - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))), dim=1)).unsqueeze(1)
        if self.name == "prod": return (x[:, 0] * x[:, 1]).unsqueeze(1)
        return x

    def generate_data(self, n_samples=2000):
        r_train, r_near, r_far = self.get_ranges()
        x_train = torch.empty(n_samples, self.dim).uniform_(*r_train)
        x_near = torch.empty(n_samples, self.dim).uniform_(*r_near)
        x_far = torch.empty(n_samples, self.dim).uniform_(*r_far)
        return {
            "train": (x_train.to(device), self.evaluate(x_train).to(device)),
            "near": (x_near.to(device), self.evaluate(x_near).to(device)),
            "far": (x_far.to(device), self.evaluate(x_far).to(device))
        }

# --- COMPONENTES POLIMÓRFICOS ---

class ResonanceLayer(nn.Module):
    def __init__(self, in_dim, k_oscillators=8):
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
    """
    Una etapa que proyecta la entrada a bases estructurales, logarítmicas y resonantes.
    """
    def __init__(self, in_dim, out_dim=16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.res = ResonanceLayer(in_dim, k_oscillators=8)
        self.log = LogInteractionLayer(in_dim, out_dim=8)
        self.n_poly_bases = 8
        
        # Entrada total: in_dim * 8 (res) + 8 (log) + in_dim * 8 (structural)
        self.total_in = in_dim * 8 + 8 + in_dim * 8
        self.integrator = nn.Linear(self.total_in, out_dim)
        
    def forward(self, x):
        res_f = self.res(x)
        log_f = self.log(x)
        
        # Structural bases
        b1, b2, b3, b4 = x, x**2, x**3, torch.abs(x)
        b5, b6, b7 = torch.sin(x), torch.cos(x), 1.0 / (x + 1e-8)
        x_nz = torch.where(x == 0, torch.ones_like(x)*1e-6, x)
        b8 = torch.sin(x_nz) / x_nz
        poly_f = torch.cat([b1, b2, b3, b4, b5, b6, b7, b8], dim=1)
        
        combined = torch.cat([res_f, log_f, poly_f], dim=1)
        z = self.integrator(combined)
        return torch.tanh(z)

class DeepPolymorphicNet(nn.Module):
    """
    V193: Red Polimórfica Profunda.
    """
    def __init__(self, in_dim, hidden_dim=16, depth=2):
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
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x): return self.net(x)

# --- ENGINE ---

def train_and_eval(func_bench, model, epochs=3000):
    data = func_bench.generate_data()
    x_train, y_train = data["train"]
    x_far, y_far = data["far"]
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
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
    return m_train, m_far

def main():
    benchmarks = [
        FunctionBenchmark("Rastrigin", 2),
        FunctionBenchmark("Ackley", 2),
        FunctionBenchmark("Schwefel", 2),
    ]
    
    results = []
    for bench in benchmarks:
        print(f"\n>>> BENCHMARK: {bench.name}")
        models = [
            ("MLP-L", BaselineMLP(bench.dim, 128)),
            ("Poly-V192-Flat", DeepPolymorphicNet(bench.dim, hidden_dim=32, depth=1)),
            ("Poly-V193-Deep", DeepPolymorphicNet(bench.dim, hidden_dim=16, depth=2))
        ]
        
        for m_name, model in models:
            model.to(device)
            p_count = sum(p.numel() for p in model.parameters())
            print(f"  Training {m_name} ({p_count} params)...")
            m_train, m_far = train_and_eval(bench, model)
            results.append({
                "func": bench.name, "model": m_name, "params": p_count,
                "mse_train": m_train, "mse_far": m_far,
                "ratio": m_far / (m_train + 1e-12)
            })
            print(f"    Train: {m_train:.2e} | Far: {m_far:.2e} | Ratio: {results[-1]['ratio']:.2e}")

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v193_deep_poly.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
