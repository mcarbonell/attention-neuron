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
    print(f"Using DirectML device: {device}")
except ImportError:
    print("torch-directml not found, using CPU")

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
        if "Schwefel" in self.name:
            return [50, 250], [0, 500], [-500, 1000]
        # Default
        return [-2.0, 2.0], [-4.0, 4.0], [-10.0, 10.0]

    def evaluate(self, x):
        if self.name == "x^2": return x**2
        if self.name == "x^3": return x**3
        if self.name == "1/x": return 1.0 / (x + 1e-8)
        if self.name == "sin": return torch.sin(x)
        if self.name == "tan": return torch.clamp(torch.tan(x), -10, 10)
        if self.name == "sinc": 
            x_nz = torch.where(x == 0, torch.ones_like(x)*1e-6, x)
            return torch.sin(x_nz) / x_nz
        
        if self.name == "sum": return (x[:, 0] + x[:, 1]).unsqueeze(1)
        if self.name == "sub": return (x[:, 0] - x[:, 1]).unsqueeze(1)
        if self.name == "prod": return (x[:, 0] * x[:, 1]).unsqueeze(1)
        if self.name == "div": 
            x2_safe = torch.where(torch.abs(x[:, 1]) < 0.1, torch.sign(x[:, 1])*0.1, x[:, 1])
            return (x[:, 0] / x2_safe).unsqueeze(1)
        if self.name == "mod": return torch.remainder(x[:, 0], x[:, 1] + 1e-6).unsqueeze(1)
        
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
            
        return x

    def generate_data(self, n_samples=2000):
        r_train, r_near, r_far = self.get_ranges()
        
        x_train = torch.empty(n_samples, self.dim).uniform_(*r_train)
        x_near = torch.empty(n_samples, self.dim).uniform_(*r_near)
        x_far = torch.empty(n_samples, self.dim).uniform_(*r_far)
        
        y_train = self.evaluate(x_train)
        y_near = self.evaluate(x_near)
        y_far = self.evaluate(x_far)
        
        return {
            "train": (x_train.to(device), y_train.to(device)),
            "near": (x_near.to(device), y_near.to(device)),
            "far": (x_far.to(device), y_far.to(device))
        }

# --- MODELOS ---

class BaselineMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, layers=2):
        super().__init__()
        model = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(layers - 1):
            model.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        model.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        return self.net(x)

class InteractionLayer(nn.Module):
    """
    Capa que calcula productos cruzados de las características de entrada.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        
    def forward(self, x):
        B, D = x.shape
        if D < 2:
            return torch.zeros(B, 0, device=x.device)
        
        # Para D >= 2, producto circular x_i * x_{i-1}
        rolled = torch.roll(x, shifts=1, dims=1)
        inter = x * rolled
        return inter

class StructuralPolyNeuron(nn.Module):
    """
    V190: Neurona Polimórfica con bases expandidas e interacción.
    """
    def __init__(self, in_dim, hidden_dim=16):
        super().__init__()
        self.in_dim = in_dim
        
        # Bases: Linear, Square, Cube, Abs, Sin, Cos, Inv, Sinc
        self.n_bases = 8
        
        # Capa de interacción (Cross-products)
        self.inter = InteractionLayer(in_dim)
        # d_inter: D para D >= 2, 0 para D < 2
        d_inter = in_dim if in_dim >= 2 else 0
        
        self.in_features = in_dim * self.n_bases + d_inter
        self.weights = nn.Parameter(torch.randn(hidden_dim, self.in_features) / math.sqrt(in_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Bases 1D
        b1 = x # Linear
        b2 = x**2 # Square
        b3 = x**3 # Cube
        b4 = torch.abs(x) # Abs
        b5 = torch.sin(x) # Sin
        b6 = torch.cos(x) # Cos
        b7 = 1.0 / (x + 1e-8) # Inv
        # Sinc
        x_nz = torch.where(x == 0, torch.ones_like(x)*1e-6, x)
        b8 = torch.sin(x_nz) / x_nz
        
        bases_1d = torch.cat([b1, b2, b3, b4, b5, b6, b7, b8], dim=1)
        
        # Interacciones
        interactions = self.inter(x)
        
        combined = torch.cat([bases_1d, interactions], dim=1)
        
        feat = F.linear(combined, self.weights, self.bias)
        feat = torch.tanh(feat)
        
        return self.head(feat)

# --- ENGINE ---

def train_and_eval(func_bench, model, epochs=2000, lr=0.01):
    data = func_bench.generate_data()
    x_train, y_train = data["train"]
    x_near, y_near = data["near"]
    x_far, y_far = data["far"]
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Early batches info (Rule GEMINI.md)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch < 5:
            print(f"  [Init] Epoch {epoch}: Loss {loss.item():.6e}")
            
        if epoch % 500 == 0:
            # Eval progress
            model.eval()
            with torch.no_grad():
                v_near = criterion(model(x_near), y_near).item()
                print(f"  Epoch {epoch} | Train: {loss.item():.6e} | Near: {v_near:.6e}")
            model.train()

    # Final Eval
    model.eval()
    with torch.no_grad():
        mse_train = criterion(model(x_train), y_train).item()
        mse_near = criterion(model(x_near), y_near).item()
        mse_far = criterion(model(x_far), y_far).item()
        
    return mse_train, mse_near, mse_far

def main():
    benchmarks = [
        # 1D
        FunctionBenchmark("x^2", 1),
        FunctionBenchmark("x^3", 1),
        FunctionBenchmark("1/x", 1),
        FunctionBenchmark("sin", 1),
        FunctionBenchmark("sinc", 1),
        # 2D
        FunctionBenchmark("sum", 2),
        FunctionBenchmark("prod", 2),
        FunctionBenchmark("div", 2),
        FunctionBenchmark("mod", 2),
        # ND (3D for speed, can be 10D)
        FunctionBenchmark("Rastrigin", 3),
        FunctionBenchmark("Ackley", 3),
        FunctionBenchmark("Schwefel", 3),
    ]
    
    results = []
    
    for bench in benchmarks:
        print(f"\n>>> BENCHMARK: {bench.name} (Dim: {bench.dim})")
        
        # Modelos
        models = [
            ("MLP-S", BaselineMLP(bench.dim, 16, 1)),
            ("MLP-M", BaselineMLP(bench.dim, 64, 2)),
            ("MLP-L", BaselineMLP(bench.dim, 256, 3)),
            ("Poly-V190", StructuralPolyNeuron(bench.dim, 16))
        ]
        
        for m_name, model in models:
            model.to(device)
            p_count = sum(p.numel() for p in model.parameters())
            print(f"  Training {m_name} ({p_count} params)...")
            
            t0 = time.time()
            m_train, m_near, m_far = train_and_eval(bench, model)
            dt = time.time() - t0
            
            res = {
                "function": bench.name,
                "model": m_name,
                "params": p_count,
                "mse_train": m_train,
                "mse_near": m_near,
                "mse_far": m_far,
                "gen_ratio": m_far / (m_train + 1e-12),
                "time": dt
            }
            results.append(res)
            print(f"    Done. Train: {m_train:.2e} | Far: {m_far:.2e} | Ratio: {res['gen_ratio']:.2e}")

    # Guardar
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v190_ood_generalization.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nBenchmark completed. Results saved to results/raw/v190_ood_generalization.json")

if __name__ == "__main__":
    main()
