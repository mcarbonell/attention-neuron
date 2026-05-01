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

# --- UTILIDADES ESPECTRALES (Walsh 1D) ---
def get_walsh_matrix_sequency(N):
    def get_walsh(n):
        if n == 1: return torch.tensor([[1.0]])
        h_prev = get_walsh(n // 2)
        return torch.cat([torch.cat([h_prev, h_prev], dim=1),
                          torch.cat([h_prev, -h_prev], dim=1)], dim=0)
    H = get_walsh(N)
    crossings = [( (H[i, :-1] * H[i, 1:] < 0).sum().item(), i) for i in range(N)]
    crossings.sort()
    return H[[idx for _, idx in crossings]]

# --- MÓDULOS DE PENSAMIENTO ---

class FastPolymorphicLayer(nn.Module):
    """ Pensamiento Rápido: Lógica analítica directa (SUM, PROD, BASES) """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w_add = nn.Linear(in_dim, out_dim)
        self.w_p1 = nn.Linear(in_dim, out_dim)
        self.w_p2 = nn.Linear(in_dim, out_dim)
        self.w_base = nn.Parameter(torch.randn(out_dim, in_dim * 4) / math.sqrt(in_dim))
        self.mixer = nn.Parameter(torch.zeros(out_dim, 3)) # [Add, Prod, Base]

    def forward(self, x):
        y_add = self.w_add(x)
        y_prod = self.w_p1(x) * self.w_p2(x)
        
        b_sq = x**2
        b_sin = torch.sin(x)
        b_cos = torch.cos(x)
        x_safe = torch.where(x >= 0, x.clamp(min=1e-2), x.clamp(max=-1e-2))
        b_inv = 1.0 / x_safe
        bases = torch.cat([b_sq, b_sin, b_cos, b_inv], dim=1)
        y_base = F.linear(bases, self.w_base)
        
        candidates = torch.stack([y_add, y_prod, y_base], dim=2)
        weights = F.softmax(self.mixer, dim=1)
        return (candidates * weights.unsqueeze(0)).sum(dim=2)

class SlowSpectralReflection(nn.Module):
    """ Pensamiento Lento: Refinamiento espectral profundo (Walsh) """
    def __init__(self, out_dim, K=32):
        super().__init__()
        self.K = K
        self.register_buffer('H', get_walsh_matrix_sequency(K))
        self.coeffs = nn.Parameter(torch.randn(out_dim, K) * 0.01)

    def forward(self, x):
        if x.shape[1] > 1: x_val = x.mean(dim=1, keepdim=True)
        else: x_val = x
        
        t = (x_val + 2.0) / 4.0 * (self.K - 1)
        t = t.clamp(0, self.K - 1)
        idx_low = t.long()
        idx_high = (idx_low + 1).clamp(0, self.K - 1)
        alpha = t - idx_low.float()
        
        w_low = self.H[:, idx_low.squeeze(1)].t()
        w_high = self.H[:, idx_high.squeeze(1)].t()
        w_interp = (1 - alpha) * w_low + alpha * w_high
        return F.linear(w_interp, self.coeffs)

class CognitiveNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim=8):
        super().__init__()
        self.fast = FastPolymorphicLayer(in_dim, hidden_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        self.slow = SlowSpectralReflection(hidden_dim, K=32)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        y_fast = self.fast(x)
        gate = self.gate_net(x)
        y_slow = self.slow(x)
        
        # Combinación jerárquica
        y_combined = y_fast + gate * y_slow
        return self.head(y_combined), gate

# --- ENTRENAMIENTO ---

def get_data(func_name, n_samples=2000, train_range=(-2, 2), test_range=(-4, 4)):
    if func_name in ["prod", "div"]:
        x_train = torch.empty(n_samples, 2).uniform_(*train_range)
        x_test = torch.empty(n_samples, 2).uniform_(*test_range)
    else:
        x_train = torch.empty(n_samples, 1).uniform_(*train_range)
        x_test = torch.empty(n_samples, 1).uniform_(*test_range)
        
    def evaluate(x, name):
        if name == "x^2": return x**2
        if name == "1/x": return 1.0 / torch.where(torch.abs(x) < 0.1, torch.sign(x)*0.1, x)
        if name == "prod": return (x[:, 0] * x[:, 1]).unsqueeze(1)
        if name == "div": return (x[:, 0] / torch.where(torch.abs(x[:, 1]) < 0.1, torch.sign(x[:, 1])*0.1, x[:, 1])).unsqueeze(1)
        if name == "sin": return torch.sin(x)
        if name == "cos": return torch.cos(x)
        if name == "tan": return torch.clamp(torch.tan(x), -10, 10)
        if name == "sinc": return torch.sin(x+1e-6) / (x+1e-6)
        return x

    y_train = evaluate(x_train, func_name)
    y_test = evaluate(x_test, func_name)
    return x_train.to(device), y_train.to(device), x_test.to(device), y_test.to(device)

def run_experiment(func_name, hidden_dim=8, epochs=1000):
    x_train, y_train, x_test, y_test = get_data(func_name)
    in_dim = x_train.shape[1]
    
    seeds = 3 # Reducido para velocidad en V135
    results_per_seed = []
    gates_per_seed = []
    
    for seed in range(seeds):
        torch.manual_seed(seed)
        model = CognitiveNetwork(in_dim, hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred, gate = model(x_train)
            mse_loss = F.mse_loss(pred, y_train)
            # Penalización por "Pensar demasiado" (Sparsity del gate)
            sparsity_loss = 0.01 * gate.mean()
            loss = mse_loss + sparsity_loss
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            final_pred, final_gate = model(x_test)
            mse_test = F.mse_loss(final_pred, y_test).item()
            results_per_seed.append(mse_test)
            gates_per_seed.append(final_gate.mean().item())
            
    metrics = {
        "func": func_name,
        "model": "cognitive_v135",
        "mse_test": np.mean(results_per_seed),
        "gate_avg": np.mean(gates_per_seed),
        "params": sum(p.numel() for p in model.parameters())
    }
    return metrics

def main():
    functions = ["x^2", "1/x", "prod", "sin", "tan", "sinc"]
    all_results = []
    
    print("\n" + "="*50)
    print("V135 BENCHMARK: COGNITIVE HIERARCHY (FAST vs SLOW)")
    print("="*50 + "\n")
    
    for func in functions:
        res = run_experiment(func)
        all_results.append(res)
        print(f"[{func:5}] Test MSE: {res['mse_test']:.6f} | Gate Avg (Thought): {res['gate_avg']*100:.1f}% | Params: {res['params']}")
        
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v135_cognitive_hierarchy.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()
