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

# --- GENERADORES DE DATOS (Mismos que V190 + Leyes Físicas) ---
class FunctionBenchmark:
    def __init__(self, name, dim=1):
        self.name = name
        self.dim = dim
        
    def get_ranges(self):
        if self.name == "1/x":
            return [0.5, 2.0], [0.2, 5.0], [0.1, 10.0]
        if self.name == "gravity":
            # m1, m2, r. r must be > 0
            return [1.0, 5.0], [0.5, 10.0], [0.1, 20.0]
        # Default
        return [-2.0, 2.0], [-4.0, 4.0], [-10.0, 10.0]

    def evaluate(self, x):
        if self.name == "prod": return (x[:, 0] * x[:, 1]).unsqueeze(1)
        if self.name == "div": 
            x2_safe = torch.where(torch.abs(x[:, 1]) < 0.1, torch.sign(x[:, 1])*0.1, x[:, 1])
            return (x[:, 0] / x2_safe).unsqueeze(1)
        if self.name == "gravity":
            # G * m1 * m2 / r^2. Dim=3: (m1, m2, r)
            m1, m2, r = x[:, 0], x[:, 1], x[:, 2]
            r_safe = torch.clamp(torch.abs(r), min=0.1)
            return (m1 * m2 / (r_safe**2)).unsqueeze(1)
        
        # Copiamos algunas de V190 para comparativa
        if self.name == "x^2": return x**2
        if self.name == "sin": return torch.sin(x)
        
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

# --- MODELOS ---

class LogInteractionLayer(nn.Module):
    """
    V191b: Capa de interacción basada en logaritmos con manejo de signos.
    """
    def __init__(self, in_dim, out_dim=8):
        super().__init__()
        self.in_dim = in_dim
        # Proyectamos a un espacio donde sumamos logs
        self.log_weights = nn.Linear(in_dim, out_dim)
        # Rama para aprender el signo resultante
        self.sign_net = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        # x: (B, D)
        x_sign = torch.sign(x)
        x_abs = torch.abs(x) + 1e-6
        x_log = torch.log(x_abs)
        
        # Magnitud: exp(sum w_i log |x_i|)
        z_log = self.log_weights(x_log)
        z_mag = torch.exp(z_log)
        
        # Signo: Aproximación continua del producto de signos
        # Para prod(x,y), sign(x)*sign(y) es lo que queremos.
        # Una forma es usar tanh de la suma de signos o similar.
        z_sign = torch.tanh(self.sign_net(x_sign))
        
        return z_mag * z_sign

class LogStructuralPoly(nn.Module):
    """
    V191: Neurona Polimórfica con Rama Logarítmica Mejorada.
    """
    def __init__(self, in_dim, hidden_dim=16):
        super().__init__()
        self.in_dim = in_dim
        self.n_bases = 8
        
        # Aumentamos el número de canales logarítmicos
        self.log_branch = LogInteractionLayer(in_dim, out_dim=8)
        
        self.in_features = in_dim * self.n_bases
        self.weights = nn.Parameter(torch.randn(hidden_dim, self.in_features + 8) / math.sqrt(in_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Bases 1D
        b1, b2, b3, b4 = x, x**2, x**3, torch.abs(x)
        b5, b6, b7 = torch.sin(x), torch.cos(x), 1.0 / (x + 1e-8)
        x_nz = torch.where(x == 0, torch.ones_like(x)*1e-6, x)
        b8 = torch.sin(x_nz) / x_nz
        bases_1d = torch.cat([b1, b2, b3, b4, b5, b6, b7, b8], dim=1)
        
        # Log Interaction (Signed)
        log_feats = self.log_branch(x)
        
        combined = torch.cat([bases_1d, log_feats], dim=1)
        
        feat = F.linear(combined, self.weights, self.bias)
        feat = torch.tanh(feat)
        
        return self.head(feat)

class BaselineMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x): return self.net(x)

# --- ENGINE ---

def train_and_eval(func_bench, model, epochs=2000):
    data = func_bench.generate_data()
    x_train, y_train = data["train"]
    x_near, y_near = data["near"]
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
        FunctionBenchmark("prod", 2),
        FunctionBenchmark("div", 2),
        FunctionBenchmark("gravity", 3),
        FunctionBenchmark("x^2", 1),
    ]
    
    results = []
    for bench in benchmarks:
        print(f"\n>>> BENCHMARK: {bench.name}")
        models = [
            ("MLP-M", BaselineMLP(bench.dim, 64)),
            ("Poly-Log-V191", LogStructuralPoly(bench.dim, 16))
        ]
        
        for m_name, model in models:
            model.to(device)
            print(f"  Training {m_name}...")
            m_train, m_far = train_and_eval(bench, model)
            results.append({
                "func": bench.name, "model": m_name, 
                "mse_train": m_train, "mse_far": m_far,
                "ratio": m_far / (m_train + 1e-12)
            })
            print(f"    Train: {m_train:.2e} | Far: {m_far:.2e} | Ratio: {results[-1]['ratio']:.2e}")

    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v191_log_interaction.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
