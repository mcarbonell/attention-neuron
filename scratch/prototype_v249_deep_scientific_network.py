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
def get_data(func_name, n_samples=5000, train_range=(-5, 5), test_ranges=[(-10, 10), (-20, 20)]):
    # Reducimos un poco el rango para estabilidad inicial en Deep
    x_train = torch.empty(n_samples, 1).uniform_(*train_range)
    x_tests = [torch.empty(n_samples, 1).uniform_(*tr) for tr in test_ranges]
        
    def evaluate(x, name):
        if name == "gaussian": return torch.exp(-0.1 * x**2)
        if name == "sin_sq": return torch.sin(0.1 * x**2)
        if name == "quad_plus_sin": return 0.1 * x**2 + torch.sin(x)
        return x

    y_train = evaluate(x_train, func_name)
    y_tests = [evaluate(xt, func_name) for xt in x_tests]
    
    return x_train.to(device), y_train.to(device), [(xt.to(device), yt.to(device)) for xt, yt in zip(x_tests, y_tests)]

# --- MODELOS ---

class BasisAugmentor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.eps = 1e-6
        
    def get_basis_names(self):
        names = []
        for i in range(self.in_dim):
            pfx = f"h{i}_" if self.in_dim > 1 else "x_"
            names.extend([
                f"{pfx}linear", f"{pfx}square", f"{pfx}cubic",
                f"{pfx}inv", f"{pfx}log", f"{pfx}exp",
                f"{pfx}sin", f"{pfx}cos",
                f"{pfx}relu", f"{pfx}abs", f"{pfx}sgn"
            ])
        if self.in_dim >= 2:
            names.append("h0*h1")
        return names

    def forward(self, x):
        # Clamping input for stability
        x = torch.clamp(x, -20, 20)
        bases = []
        for i in range(self.in_dim):
            xi = x[:, i:i+1]
            bases.append(xi)
            bases.append(xi**2)
            bases.append(xi**3)
            # Inverse safe
            bases.append(1.0 / (torch.abs(xi) + 0.1))
            bases.append(torch.log(torch.abs(xi) + 0.1))
            # Exp Clamped to avoid inf
            bases.append(torch.exp(torch.clamp(xi, -10, 5)))
            bases.append(torch.sin(xi))
            bases.append(torch.cos(xi))
            bases.append(F.relu(xi))
            bases.append(torch.abs(xi))
            bases.append(torch.sign(xi))
            
        if self.in_dim >= 2:
            x0 = x[:, 0:1]
            x1 = x[:, 1:2]
            bases.append(x0 * x1)
            
        return torch.cat(bases, dim=1)

class ScientificLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.augmentor = BasisAugmentor(in_dim)
        dummy_x = torch.zeros(1, in_dim)
        aug_dim = self.augmentor(dummy_x).shape[1]
        self.linear = nn.Linear(aug_dim, out_dim)
        # Init weights small for stability
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        x_aug = self.augmentor(x)
        return self.linear(x_aug)

class DeepScientificModel(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=2):
        super().__init__()
        self.layer1 = ScientificLayer(in_dim, hidden_dim)
        self.layer2 = ScientificLayer(hidden_dim, 1)
        
    def forward(self, x):
        h = self.layer1(x)
        # Clamp intermediate for safety
        h = torch.clamp(h, -10, 10)
        y = self.layer2(h)
        return y

# --- ENTRENAMIENTO ---

def run_experiment(func_name, epochs=6000, lambda_l1=0.001, pruning_threshold=0.05):
    test_ranges = [(-10, 10), (-20, 20)]
    x_train, y_train, test_data = get_data(func_name, train_range=(-5, 5), test_ranges=test_ranges)
    
    model = DeepScientificModel(in_dim=1, hidden_dim=2).to(device)
    # Usamos un LR más pequeño para estabilidad en Deep
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    print(f"\n--- Training for {func_name} ---")
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x_train)
        mse_loss = criterion(pred, y_train)
        
        l1_loss = torch.norm(model.layer1.linear.weight, 1) + torch.norm(model.layer2.linear.weight, 1)
        loss = mse_loss + lambda_l1 * l1_loss
        
        if torch.isnan(loss):
            print("  !!! NaN detected, stopping training")
            break
            
        loss.backward()
        # Clip grads
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"  Epoch {epoch}: MSE {mse_loss.item():.8f}")
            
    # PRUNING
    with torch.no_grad():
        mask1 = torch.abs(model.layer1.linear.weight) > pruning_threshold
        model.layer1.linear.weight *= mask1
        mask2 = torch.abs(model.layer2.linear.weight) > pruning_threshold
        model.layer2.linear.weight *= mask2

    # Evaluation
    model.eval()
    mse_tests = []
    with torch.no_grad():
        for xt, yt in test_data:
            mse = criterion(model(xt), yt).item()
            mse_tests.append(mse)
            
    return model, mse_tests

def print_model_formula(model, pruning_threshold=0.01):
    print("\n  [LAYER 1: Discovery of Hidden Laws]")
    l1_names = model.layer1.augmentor.get_basis_names()
    l1_w = model.layer1.linear.weight.data.cpu().numpy()
    l1_b = model.layer1.linear.bias.data.cpu().numpy()
    
    for i in range(l1_w.shape[0]):
        print(f"    h{i} = ", end="")
        terms = []
        for j, name in enumerate(l1_names):
            if abs(l1_w[i, j]) > pruning_threshold:
                terms.append(f"{l1_w[i, j]:.3f}*{name}")
        if abs(l1_b[i]) > pruning_threshold:
            terms.append(f"{l1_b[i]:.3f}")
        print(" + ".join(terms) if terms else "0")

    print("\n  [LAYER 2: Composition]")
    l2_names = model.layer2.augmentor.get_basis_names()
    l2_w = model.layer2.linear.weight.data.squeeze().cpu().numpy()
    l2_b = model.layer2.linear.bias.item()
    
    print("    y = ", end="")
    terms = []
    for j, name in enumerate(l2_names):
        if abs(l2_w[j]) > pruning_threshold:
            terms.append(f"{l2_w[j]:.3f}*{name}")
    if abs(l2_b) > pruning_threshold:
        terms.append(f"{l2_b:.3f}")
    print(" + ".join(terms) if terms else "0")

def main():
    functions = ["gaussian", "sin_sq", "quad_plus_sin"]
    
    print("\n" + "="*70)
    print("EXPERIMENTO V249b: STABLE DEEP SCIENTIFIC NETWORK")
    print("="*70 + "\n")
    
    for func in functions:
        model, mse_tests = run_experiment(func)
        print(f"\n>>> RESULTS FOR {func}:")
        print(f"  MSE Test(10): {mse_tests[0]:.2e} | MSE Test(20): {mse_tests[1]:.2e}")
        print_model_formula(model)
        
    print(f"\nExperimento v249 completado.")

if __name__ == "__main__":
    main()
