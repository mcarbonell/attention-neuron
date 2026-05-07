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
def get_data(func_name, n_samples=5000, train_range=(-10, 10), test_ranges=[(-20, 20), (-40, 40)]):
    if func_name in ["prod", "div"]:
        x_train = torch.empty(n_samples, 2).uniform_(*train_range)
        x_tests = [torch.empty(n_samples, 2).uniform_(*tr) for tr in test_ranges]
    else:
        x_train = torch.empty(n_samples, 1).uniform_(*train_range)
        x_tests = [torch.empty(n_samples, 1).uniform_(*tr) for tr in test_ranges]
        
    def evaluate(x, name):
        if name == "x^2": return x**2
        if name == "x^3": return x**3
        if name == "1/x": 
            x_safe = torch.where(torch.abs(x) < 0.1, torch.sign(x)*0.1, x)
            return 1.0 / x_safe
        if name == "prod": return (x[:, 0] * x[:, 1]).unsqueeze(1)
        if name == "sin": return torch.sin(x)
        if name == "cos": return torch.cos(x)
        if name == "tan": return torch.clamp(torch.tan(x), -10, 10)
        if name == "sinc": 
            x_nz = torch.where(x == 0, torch.ones_like(x)*1e-6, x)
            return torch.sin(x_nz) / x_nz
        return x

    y_train = evaluate(x_train, func_name)
    y_tests = [evaluate(xt, func_name) for xt in x_tests]
    
    return x_train.to(device), y_train.to(device), [(xt.to(device), yt.to(device)) for xt, yt in zip(x_tests, y_tests)]

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

class BasisAugmentor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.eps = 1e-6
        
    def get_basis_names(self):
        names = []
        for i in range(self.in_dim):
            pfx = f"x{i}_" if self.in_dim > 1 else ""
            names.extend([
                f"{pfx}linear", f"{pfx}square", f"{pfx}cubic", f"{pfx}quart",
                f"{pfx}inv", f"{pfx}inv_sq", f"{pfx}log", f"{pfx}exp",
                f"{pfx}sin", f"{pfx}cos", f"{pfx}sin2", f"{pfx}cos2",
                f"{pfx}relu", f"{pfx}relu_neg", f"{pfx}abs", f"{pfx}sgn",
                f"{pfx}saw"
            ])
        if self.in_dim == 2:
            names.extend(["x0*x1", "log|x0*x1|"])
        return names

    def forward(self, x):
        bases = []
        for i in range(self.in_dim):
            xi = x[:, i:i+1]
            bases.append(xi) # linear
            bases.append(xi**2) # square
            bases.append(xi**3) # cubic
            bases.append(xi**4) # quart
            bases.append(1.0 / (xi + self.eps)) # inv
            bases.append(1.0 / (xi**2 + self.eps)) # inv_sq
            bases.append(torch.log(torch.abs(xi) + self.eps)) # log
            bases.append(torch.exp(xi)) # exp
            bases.append(torch.sin(xi)) # sin
            bases.append(torch.cos(xi)) # cos
            bases.append(torch.sin(2*xi)) # sin2
            bases.append(torch.cos(2*xi)) # cos2
            bases.append(F.relu(xi)) # relu
            bases.append(F.relu(-xi)) # relu_neg
            bases.append(torch.abs(xi)) # abs
            bases.append(torch.sign(xi)) # sgn
            bases.append(xi - torch.floor(xi)) # saw
            
        if self.in_dim == 2:
            x0 = x[:, 0:1]
            x1 = x[:, 1:2]
            bases.append(x0 * x1)
            bases.append(torch.log(torch.abs(x0 * x1) + self.eps))
            
        return torch.cat(bases, dim=1)

class SingleNeuronAugmented(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.augmentor = BasisAugmentor(in_dim)
        dummy_x = torch.zeros(1, in_dim)
        aug_dim = self.augmentor(dummy_x).shape[1]
        self.linear = nn.Linear(aug_dim, 1)
        
    def forward(self, x):
        x_aug = self.augmentor(x)
        return self.linear(x_aug)

    def get_learned_weights(self):
        names = self.augmentor.get_basis_names()
        weights = self.linear.weight.data.squeeze().cpu().numpy()
        bias = self.linear.bias.item()
        return list(zip(names, weights)), bias

# --- ENTRENAMIENTO ---

def run_experiment(func_name, model_type="mlp", epochs=4000, lambda_l1=0.01):
    test_ranges = [(-20, 20), (-40, 40)]
    x_train, y_train, test_data = get_data(func_name, train_range=(-10, 10), test_ranges=test_ranges)
    in_dim = x_train.shape[1]
    
    if model_type == "mlp":
        model = BaselineMLP(in_dim, hidden_dim=64, layers=2).to(device)
    else:
        model = SingleNeuronAugmented(in_dim).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    t0 = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x_train)
        mse_loss = criterion(pred, y_train)
        
        # L1 Regularization for augmented neuron
        if model_type == "augmented":
            l1_loss = torch.norm(model.linear.weight, 1)
            loss = mse_loss + lambda_l1 * l1_loss
        else:
            loss = mse_loss
            
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"[{func_name} - {model_type}] Epoch {epoch}: MSE {mse_loss.item():.8f} | Total {loss.item():.8f}")
            
    # Eval
    model.eval()
    mse_tests = []
    with torch.no_grad():
        for xt, yt in test_data:
            mse = criterion(model(xt), yt).item()
            mse_tests.append(mse)
        
    wall_clock = time.time() - t0
    params = sum(p.numel() for p in model.parameters())
    
    result = {
        "func": func_name,
        "model": model_type,
        "mse_test_20": mse_tests[0],
        "mse_test_40": mse_tests[1],
        "params": params,
        "wall_clock": wall_clock
    }
    
    if model_type == "augmented":
        weights, bias = model.get_learned_weights()
        result["weights"] = weights
        result["bias"] = bias
        
    return result

def main():
    functions = ["x^2", "x^3", "1/x", "prod", "sin", "cos", "sinc"]
    all_results = []
    
    print("\n" + "="*70)
    print("EXPERIMENTO V247: SPARSE AUGMENTED FEATURE APPROXIMATOR")
    print("L1 Regularization: lambda = 0.01")
    print("="*70 + "\n")
    
    for func in functions:
        print(f"\n>>> TESTING FUNCTION: {func}")
        
        # MLP Baseline
        res_mlp = run_experiment(func, "mlp")
        # Augmented Neuron with L1
        res_aug = run_experiment(func, "augmented", lambda_l1=0.01)
        
        all_results.extend([res_mlp, res_aug])
        
        print(f"  MLP-Baseline | MSE Test(20): {res_mlp['mse_test_20']:.2e} | MSE Test(40): {res_mlp['mse_test_40']:.2e}")
        print(f"  Sparse-Aug   | MSE Test(20): {res_aug['mse_test_20']:.2e} | MSE Test(40): {res_aug['mse_test_40']:.2e}")
        
        # Mostrar interpretación (filtrando pesos pequeños)
        print("\n  Top Learned Weights (Interpretability > 0.01):")
        sorted_weights = sorted(res_aug["weights"], key=lambda x: abs(x[1]), reverse=True)
        active_bases = 0
        for name, val in sorted_weights:
            if abs(val) > 0.01:
                print(f"    {name:<12}: {val:10.4f}")
                active_bases += 1
        print(f"    {'bias':<12}: {res_aug['bias']:10.4f}")
        print(f"    Active Bases: {active_bases}")
        
    # Guardar resultados
    os.makedirs("results/raw", exist_ok=True)
    json_results = []
    for r in all_results:
        if "weights" in r:
            r["weights"] = {name: float(val) for name, val in r["weights"]}
        json_results.append(r)
        
    with open("results/raw/v247_sparse_augmented.json", "w") as f:
        json.dump(json_results, f, indent=4)
        
    print(f"\nResultados guardados en results/raw/v247_sparse_augmented.json")

if __name__ == "__main__":
    main()
