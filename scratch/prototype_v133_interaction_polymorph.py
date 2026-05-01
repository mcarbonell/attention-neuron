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

# --- GENERADORES DE DATOS (Mismo que V132) ---
def get_data(func_name, n_samples=2000, train_range=(-2, 2), test_range=(-4, 4)):
    if func_name in ["prod", "div"]:
        x_train = torch.empty(n_samples, 2).uniform_(*train_range)
        x_test = torch.empty(n_samples, 2).uniform_(*test_range)
    else:
        x_train = torch.empty(n_samples, 1).uniform_(*train_range)
        x_test = torch.empty(n_samples, 1).uniform_(*test_range)
        
    def evaluate(x, name):
        if name == "x^2": return x**2
        if name == "1/x": 
            x_safe = torch.where(torch.abs(x) < 0.1, torch.sign(x)*0.1, x)
            return 1.0 / x_safe
        if name == "prod": return (x[:, 0] * x[:, 1]).unsqueeze(1)
        if name == "div": 
            x2_safe = torch.where(torch.abs(x[:, 1]) < 0.1, torch.sign(x[:, 1])*0.1, x[:, 1])
            return (x[:, 0] / x2_safe).unsqueeze(1)
        if name == "sin": return torch.sin(x)
        if name == "cos": return torch.cos(x)
        if name == "tan": return torch.clamp(torch.tan(x), -10, 10)
        if name == "sinc": 
            x_nz = torch.where(x == 0, torch.ones_like(x)*1e-6, x)
            return torch.sin(x_nz) / x_nz
        return x

    y_train = evaluate(x_train, func_name)
    y_test = evaluate(x_test, func_name)
    return x_train.to(device), y_train.to(device), x_test.to(device), y_test.to(device)

# --- MODELOS V133 ---

class InteractionLayer(nn.Module):
    """
    Capa Polimórfica V133: Banco de Lógica con Interacción Cruzada.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 1. Canal Aditivo (Lineal)
        self.w_add = nn.Linear(in_dim, out_dim)
        
        # 2. Canal Multiplicativo (A * B)
        # Proyectamos la entrada a dos espacios y luego multiplicamos elemento a elemento
        self.w_p1 = nn.Linear(in_dim, out_dim)
        self.w_p2 = nn.Linear(in_dim, out_dim)
        
        # 3. Canal de División (A / B)
        self.w_d1 = nn.Linear(in_dim, out_dim)
        self.w_d2 = nn.Linear(in_dim, out_dim)
        
        # 4. Bases No Lineales (Sin/Cos/Square/Inv)
        # Aplicadas de forma independiente a la entrada original
        self.w_base = nn.Parameter(torch.randn(out_dim, in_dim * 4) / math.sqrt(in_dim))
        
        # Mezclador Final (Dial de Atención)
        self.mixer = nn.Parameter(torch.zeros(out_dim, 4)) # [Add, Prod, Div, Base]

    def forward(self, x):
        # x: (B, in_dim)
        
        # Additive
        y_add = self.w_add(x)
        
        # Multiplicative
        y_prod = self.w_p1(x) * self.w_p2(x)
        
        # Division
        denom = self.w_d2(x)
        denom = torch.where(denom >= 0, denom.clamp(min=1e-2), denom.clamp(max=-1e-2))
        y_div = self.w_d1(x) / denom
        y_div = torch.nan_to_num(y_div, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Bases (x^2, sin, cos, 1/x)
        b_sq = x**2
        b_sin = torch.sin(x)
        b_cos = torch.cos(x)
        x_safe = torch.where(x >= 0, x.clamp(min=1e-2), x.clamp(max=-1e-2))
        b_inv = 1.0 / x_safe
        bases = torch.cat([b_sq, b_sin, b_cos, b_inv], dim=1)
        y_base = F.linear(bases, self.w_base)
        
        # Mezclador con Softmax
        # candidatos: (B, out_dim, 4)
        candidates = torch.stack([y_add, y_prod, y_div, y_base], dim=2)
        # Clampar candidatos para evitar que un canal loco arruine todo
        candidates = torch.clamp(candidates, -100, 100)
        
        weights = F.softmax(self.mixer, dim=1) # (out_dim, 4)
        out = (candidates * weights.unsqueeze(0)).sum(dim=2)
        return out

class InteractionNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim=8):
        super().__init__()
        self.poly = InteractionLayer(in_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.poly(x)
        return self.head(x)

# --- SISTEMA DE ENTRENAMIENTO ---

def run_experiment(func_name, model_type="poly_v133", hidden_dim=8, epochs=1000):
    x_train, y_train, x_test, y_test = get_data(func_name)
    in_dim = x_train.shape[1]
    
    model = InteractionNetwork(in_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    t0 = time.time()
    eval_time = 0
    seeds = 5
    results_per_seed = []
    
    for seed in range(seeds):
        torch.manual_seed(seed)
        model = InteractionNetwork(in_dim, hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        for epoch in range(epochs):
            t_eval_start = time.time()
            optimizer.zero_grad()
            pred = model(x_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()
            eval_time += (time.time() - t_eval_start)
            
            if epoch % 200 == 0 and seed == 0:
                print(f"[{func_name}] Epoch {epoch}: Loss {loss.item():.6f}")
        
        model.eval()
        with torch.no_grad():
            pred_train = model(x_train)
            mse_train = criterion(pred_train, y_train).item()
            pred_test = model(x_test)
            mse_test = criterion(pred_test, y_test).item()
        results_per_seed.append(mse_train)
        
    wall_clock = time.time() - t0
    metrics = {
        "func": func_name,
        "model": "poly_v133",
        "final_objective": np.mean(results_per_seed),
        "std_objective": np.std(results_per_seed),
        "mse_test": mse_test,
        "total_evaluations": epochs * len(x_train) * seeds,
        "wall_clock_time": wall_clock,
        "function_evaluation_time": eval_time,
        "internal_overhead_time": wall_clock - eval_time,
        "params": sum(p.numel() for p in model.parameters())
    }
    return metrics

def main():
    functions = ["x^2", "1/x", "prod", "div", "sin", "cos", "tan", "sinc"]
    all_results = []
    
    print("\n" + "="*50)
    print("V133 BENCHMARK: INTERACTION POLYMORPHIC NEURONS")
    print("="*50 + "\n")
    
    for func in functions:
        print(f"\n>>> TESTING FUNCTION: {func}")
        res = run_experiment(func)
        all_results.append(res)
        print(f"  Poly-V133 | MSE Train: {res['final_objective']:.6f} | Test: {res['mse_test']:.6f} | Params: {res['params']}")
        
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v133_interaction_polymorph.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()
