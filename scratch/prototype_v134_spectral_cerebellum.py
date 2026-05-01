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

# --- UTILIDADES ESPECTRALES ---

def get_walsh_matrix(N):
    if N == 1: return torch.tensor([[1.0]])
    H_prev = get_walsh_matrix(N // 2)
    top = torch.cat([H_prev, H_prev], dim=1)
    bottom = torch.cat([H_prev, -H_prev], dim=1)
    return torch.cat([top, bottom], dim=0)

def get_walsh_matrix_sequency(N):
    H = get_walsh_matrix(N)
    crossings = []
    for i in range(N):
        row = H[i]
        num_crossings = (row[:-1] * row[1:] < 0).sum().item()
        crossings.append((num_crossings, i))
    crossings.sort()
    return H[[idx for _, idx in crossings]]

class SpectralCerebellum(nn.Module):
    """
    Sintetizador espectral 1D basado en Walsh.
    Aproxima f(x) como una combinación lineal de funciones de Walsh.
    """
    def __init__(self, out_dim, K=16):
        super().__init__()
        self.out_dim = out_dim
        self.K = K
        # Matriz de Walsh (K, K)
        self.register_buffer('H', get_walsh_matrix_sequency(K))
        # Coeficientes espectrales aprendidos por neurona de salida
        self.coeffs = nn.Parameter(torch.randn(out_dim, K) * 0.01)

    def forward(self, x):
        # x: (B, in_dim). Si in_dim > 1, promediamos o tomamos la primera.
        if x.shape[1] > 1:
            x_val = x.mean(dim=1, keepdim=True)
        else:
            x_val = x
            
        # Mapeamos x_val de [-2, 2] a [0, K-1] para indexar la base de Walsh
        # Usamos soft-indexing (interpolación lineal) para mantener diferenciabilidad respecto a x
        t = (x_val + 2.0) / 4.0 * (self.K - 1)
        t = t.clamp(0, self.K - 1)
        
        idx_low = t.long()
        idx_high = (idx_low + 1).clamp(0, self.K - 1)
        alpha = t - idx_low.float()
        
        # Obtenemos los valores de las bases de Walsh en los puntos low/high
        # H es (K_bases, K_time). Queremos indexar K_time.
        # w_low: (B, K_bases)
        w_low = self.H[:, idx_low.squeeze(1)].t()
        w_high = self.H[:, idx_high.squeeze(1)].t()
        
        # Interpolación de la base: (B, K)
        w_interp = (1 - alpha) * w_low + alpha * w_high
        
        # Salida: (B, out_dim) = combinación lineal de bases usando los coeficientes de cada neurona
        # (B, K) @ (out_dim, K).t() -> (B, out_dim)
        return F.linear(w_interp, self.coeffs)

# --- MODELO V134 ---

class PolymorphicLayerV134(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 1. SUM
        self.w_add = nn.Linear(in_dim, out_dim)
        # 2. PROD
        self.w_p1 = nn.Linear(in_dim, out_dim)
        self.w_p2 = nn.Linear(in_dim, out_dim)
        # 3. DIV
        self.w_d1 = nn.Linear(in_dim, out_dim)
        self.w_d2 = nn.Linear(in_dim, out_dim)
        # 4. BASES
        self.w_base = nn.Parameter(torch.randn(out_dim, in_dim * 4) / math.sqrt(in_dim))
        # 5. SPECTRAL (Cerebelo)
        self.cerebellum = SpectralCerebellum(out_dim, K=16)
        
        # Mixer Dial (Atención a 5 canales)
        self.mixer = nn.Parameter(torch.zeros(out_dim, 5))

    def forward(self, x):
        y_add = self.w_add(x)
        y_prod = self.w_p1(x) * self.w_p2(x)
        
        denom = self.w_d2(x)
        denom = torch.where(denom >= 0, denom.clamp(min=1e-2), denom.clamp(max=-1e-2))
        y_div = torch.nan_to_num(self.w_d1(x) / denom, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Primitivas fijas
        b_sq = x**2
        b_sin = torch.sin(x)
        b_cos = torch.cos(x)
        x_safe = torch.where(x >= 0, x.clamp(min=1e-2), x.clamp(max=-1e-2))
        b_inv = 1.0 / x_safe
        bases = torch.cat([b_sq, b_sin, b_cos, b_inv], dim=1)
        y_base = F.linear(bases, self.w_base)
        
        # Canal Espectral
        y_spec = self.cerebellum(x)
        
        # Mixing
        candidates = torch.stack([y_add, y_prod, y_div, y_base, y_spec], dim=2)
        candidates = torch.clamp(candidates, -100, 100)
        
        weights = F.softmax(self.mixer, dim=1)
        out = (candidates * weights.unsqueeze(0)).sum(dim=2)
        return out

class PolymorphicNetworkV134(nn.Module):
    def __init__(self, in_dim, hidden_dim=8):
        super().__init__()
        self.poly = PolymorphicLayerV134(in_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.head(self.poly(x))

# --- BENCHMARK SYSTEM (Mismo que V133) ---

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

def run_experiment(func_name, hidden_dim=8, epochs=1000):
    x_train, y_train, x_test, y_test = get_data(func_name)
    in_dim = x_train.shape[1]
    
    seeds = 5
    results_per_seed = []
    t0 = time.time()
    eval_time = 0
    
    for seed in range(seeds):
        torch.manual_seed(seed)
        model = PolymorphicNetworkV134(in_dim, hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
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
            mse_test = criterion(model(x_test), y_test).item()
            mse_train = criterion(model(x_train), y_train).item()
        results_per_seed.append(mse_train)
        
    metrics = {
        "func": func_name,
        "model": "poly_v134",
        "final_objective": np.mean(results_per_seed),
        "std_objective": np.std(results_per_seed),
        "mse_test": mse_test,
        "params": sum(p.numel() for p in model.parameters()),
        "wall_clock_time": time.time() - t0,
        "eval_time": eval_time
    }
    return metrics

def main():
    functions = ["x^2", "1/x", "prod", "div", "sin", "cos", "tan", "sinc"]
    all_results = []
    
    print("\n" + "="*50)
    print("V134 BENCHMARK: SPECTRAL CEREBELLUM POLYMORPH")
    print("="*50 + "\n")
    
    for func in functions:
        print(f"\n>>> TESTING FUNCTION: {func}")
        res = run_experiment(func)
        all_results.append(res)
        print(f"  V134 | Train MSE: {res['final_objective']:.6f} | Test: {res['mse_test']:.6f} | Params: {res['params']}")
        
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v134_spectral_polymorph.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()
