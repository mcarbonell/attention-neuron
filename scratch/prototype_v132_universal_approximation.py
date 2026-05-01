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
# Prioridad: torch-directml (Radeon 780M) > CPU
device = torch.device('cpu')
try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using DirectML device: {device}")
except ImportError:
    print("torch-directml not found, using CPU")

# --- GENERADORES DE DATOS ---
def get_data(func_name, n_samples=2000, train_range=(-2, 2), test_range=(-4, 4)):
    """
    Genera datos para entrenamiento (interpolación) y test (extrapolación/generalización).
    """
    if func_name in ["prod", "div"]:
        # 2D inputs
        x_train = torch.empty(n_samples, 2).uniform_(*train_range)
        x_test = torch.empty(n_samples, 2).uniform_(*test_range)
    else:
        # 1D inputs
        x_train = torch.empty(n_samples, 1).uniform_(*train_range)
        x_test = torch.empty(n_samples, 1).uniform_(*test_range)
        
    def evaluate(x, name):
        if name == "x^2": return x**2
        if name == "1/x": 
            # Evitar división por cero
            x_safe = torch.where(torch.abs(x) < 0.1, torch.sign(x)*0.1, x)
            return 1.0 / x_safe
        if name == "prod": return (x[:, 0] * x[:, 1]).unsqueeze(1)
        if name == "div": 
            x2_safe = torch.where(torch.abs(x[:, 1]) < 0.1, torch.sign(x[:, 1])*0.1, x[:, 1])
            return (x[:, 0] / x2_safe).unsqueeze(1)
        if name == "sin": return torch.sin(x)
        if name == "cos": return torch.cos(x)
        if name == "tan": 
            # Clip para evitar explosiones
            return torch.clamp(torch.tan(x), -10, 10)
        if name == "sinc": 
            x_nz = torch.where(x == 0, torch.ones_like(x)*1e-6, x)
            return torch.sin(x_nz) / x_nz
        return x

    y_train = evaluate(x_train, func_name)
    y_test = evaluate(x_test, func_name)
    
    return x_train.to(device), y_train.to(device), x_test.to(device), y_test.to(device)

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

class PolymorphicLayer(nn.Module):
    """
    Una capa que proyecta la entrada a diferentes bases matemáticas
    y aprende a combinarlas.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Pesos para la combinación lineal de las bases
        # Tenemos 6 bases: Linear, Square, Abs, Sin, Cos, Inv
        self.n_bases = 6
        self.weights = nn.Parameter(torch.randn(out_dim, in_dim * self.n_bases) / math.sqrt(in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
    def forward(self, x):
        # x: (B, in_dim)
        b1 = x # Linear
        b2 = x**2 # Square
        b3 = torch.abs(x) # Abs
        b4 = torch.sin(x) # Sin
        b5 = torch.cos(x) # Cos
        b6 = 1.0 / (x + 1e-6) # Inv
        
        # Concatenamos bases: (B, in_dim * 6)
        bases = torch.cat([b1, b2, b3, b4, b5, b6], dim=1)
        
        # Salida: (B, out_dim)
        return F.linear(bases, self.weights, self.bias)

class PolymorphicNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim=16):
        super().__init__()
        # Usamos menos neuronas ocultas para demostrar eficiencia
        self.poly = PolymorphicLayer(in_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.poly(x)
        # No usamos activación no-lineal fija entre capas, 
        # dejamos que la capa polimórfica decida.
        return self.head(x)

# --- SISTEMA DE ENTRENAMIENTO Y MÉTRICAS ---

def run_experiment(func_name, model_type="mlp", hidden_dim=64, layers=2, epochs=500):
    x_train, y_train, x_test, y_test = get_data(func_name)
    
    in_dim = x_train.shape[1]
    if model_type == "mlp":
        model = BaselineMLP(in_dim, hidden_dim, layers).to(device)
    else:
        model = PolymorphicNetwork(in_dim, hidden_dim).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    t0 = time.time()
    eval_time = 0
    
    # Seeds para robustez
    seeds = 5
    results_per_seed = []
    
    for seed in range(seeds):
        torch.manual_seed(seed)
        # Re-init model
        if model_type == "mlp":
            model = BaselineMLP(in_dim, hidden_dim, layers).to(device)
        else:
            model = PolymorphicNetwork(in_dim, hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        t_seed_start = time.time()
        for epoch in range(epochs):
            t_eval_start = time.time()
            optimizer.zero_grad()
            pred = model(x_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()
            eval_time += (time.time() - t_eval_start)
            
            if epoch % 100 == 0 and seed == 0:
                print(f"[{func_name} - {model_type}] Epoch {epoch}: Loss {loss.item():.6f}")
        
        # Evaluación Final
        model.eval()
        with torch.no_grad():
            pred_train = model(x_train)
            mse_train = criterion(pred_train, y_train).item()
            
            pred_test = model(x_test)
            mse_test = criterion(pred_test, y_test).item()
            
        results_per_seed.append(mse_train)
        
    wall_clock = time.time() - t0
    avg_mse = np.mean(results_per_seed)
    std_mse = np.std(results_per_seed)
    
    metrics = {
        "func": func_name,
        "model": model_type,
        "final_objective": avg_mse,
        "std_objective": std_mse,
        "mse_test": mse_test, # Generalización
        "total_evaluations": epochs * len(x_train) * seeds,
        "wall_clock_time": wall_clock,
        "function_evaluation_time": eval_time,
        "internal_overhead_time": wall_clock - eval_time,
        "params": sum(p.numel() for p in model.parameters())
    }
    
    return metrics

def main():
    functions = ["x^2", "1/x", "prod", "sin", "cos", "tan", "sinc"]
    all_results = []
    
    print("\n" + "="*50)
    print("UNIVERSAL APPROXIMATION BENCHMARK: MLP vs POLYMORPHIC")
    print("="*50 + "\n")
    
    for func in functions:
        print(f"\n>>> TESTING FUNCTION: {func}")
        
        # Test MLP (Small, Medium, Large)
        res_mlp_s = run_experiment(func, "mlp", hidden_dim=16, layers=1, epochs=1000)
        res_mlp_m = run_experiment(func, "mlp", hidden_dim=64, layers=2, epochs=1000)
        
        # Test Polymorphic (Ultra Small)
        res_poly = run_experiment(func, "poly", hidden_dim=8, epochs=1000)
        
        all_results.extend([res_mlp_s, res_mlp_m, res_poly])
        
        # Print mini summary
        print(f"  MLP-Small  | MSE Train: {res_mlp_s['final_objective']:.6f} | Params: {res_mlp_s['params']}")
        print(f"  MLP-Medium | MSE Train: {res_mlp_m['final_objective']:.6f} | Params: {res_mlp_m['params']}")
        print(f"  Poly-Neuron| MSE Train: {res_poly['final_objective']:.6f} | Params: {res_poly['params']}")
        
    # Guardar resultados
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v132_universal_approximation.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    print("\nResultados guardados en results/raw/v132_universal_approximation.json")

if __name__ == "__main__":
    main()
