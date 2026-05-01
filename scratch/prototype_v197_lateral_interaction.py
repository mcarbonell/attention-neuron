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

# --- DISCONTINUITY OPERATORS (V195) ---
class STEFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return torch.floor(x)
    @staticmethod
    def backward(ctx, grad_output): return grad_output

def ste_mod(x, y):
    # x % y = x - y * floor(x/y)
    y_safe = torch.where(torch.abs(y) < 1e-2, torch.sign(y)*1e-2, y)
    return x - y_safe * STEFloor.apply(x / y_safe)

# --- CAPA DE INTERACCIÓN LATERAL (V197) ---

class LateralInteractionLayer(nn.Module):
    """
    V197: Las neuronas se hablan entre ellas.
    Crea 'Neuronas Hijas' combinando pares de 'Neuronas Padres'.
    """
    def __init__(self, n_parents, n_children=8):
        super().__init__()
        self.n_p = n_parents
        self.n_c = n_children
        
        # Pesos para elegir qué par de padres alimentar a cada hijo
        # Usamos atención simplificada (weights para i y j)
        self.i_select = nn.Parameter(torch.randn(n_children, n_parents))
        self.j_select = nn.Parameter(torch.randn(n_children, n_parents))
        
        # Pesos para elegir la operación (Suma, Resta, Prod, Mod)
        self.op_select = nn.Parameter(torch.randn(n_children, 4))
        
    def forward(self, p):
        # p: (B, N_parents)
        # Soft-selection de padres i y j
        p_i = torch.matmul(F.softmax(self.i_select, dim=1), p.transpose(0, 1)).transpose(0, 1) # (B, N_children)
        p_j = torch.matmul(F.softmax(self.j_select, dim=1), p.transpose(0, 1)).transpose(0, 1) # (B, N_children)
        
        # Operaciones
        op_sum = p_i + p_j
        op_sub = p_i - p_j
        op_mul = p_i * p_j
        op_mod = ste_mod(p_i, p_j)
        
        # Soft-selection de operación
        ops = torch.stack([op_sum, op_sub, op_mul, op_mod], dim=2) # (B, N_children, 4)
        gates = F.softmax(self.op_select, dim=1).unsqueeze(0) # (1, N_children, 4)
        
        children = (ops * gates).sum(dim=2) # (B, N_children)
        return children

# --- NEURONA TOTAL CON INTERACCIÓN LATERAL ---

class ParentNeuron(nn.Module):
    """Rama individual polimórfica (V195)"""
    def __init__(self, in_dim):
        super().__init__()
        # Simplificado para el test
        self.freq = nn.Parameter(torch.randn(in_dim, 8))
        self.log_w = nn.Linear(in_dim, 4)
        
    def forward(self, x):
        res = torch.sin(x.unsqueeze(-1) * self.freq).view(x.size(0), -1)
        log_f = torch.exp(self.log_w(torch.log(torch.abs(x) + 1e-6)))
        structural = torch.cat([x, x**2, 1.0/(x+1e-6)], dim=1)
        return torch.cat([res, log_f, structural], dim=1)

class LateralPolyNet(nn.Module):
    def __init__(self, in_dim, n_children=16):
        super().__init__()
        self.parents = ParentNeuron(in_dim)
        # Calculamos cuántos padres hay (8 res + 4 log + 3 structural * dim)
        n_p = 8 * in_dim + 4 + 3 * in_dim
        self.lateral = LateralInteractionLayer(n_p, n_children)
        
        # Integración Final: Padres + Hijas
        self.head = nn.Linear(n_p + n_children, 1)

    def forward(self, x):
        p = self.parents(x)
        c = self.lateral(p)
        combined = torch.cat([p, c], dim=1)
        return self.head(combined)

# --- ENGINE ---

def run_lateral_test():
    print("\n>>> V197: LATERAL INTERACTION TEST (Neuronas Padres e Hijas)")
    
    # Función de prueba compleja (composición de leyes)
    def target_func(x):
        # (x*y) % (x+y) -> Una composición que requiere interacción horizontal
        x1, x2 = x[:, 0], x[:, 1]
        return torch.remainder(x1 * x2, torch.clamp(x1 + x2, min=0.5)).unsqueeze(1)

    # Datos
    x_train = torch.empty(3000, 2).uniform_(1.0, 5.0).to(device)
    y_train = target_func(x_train).to(device)
    
    model = LateralPolyNet(in_dim=2, n_children=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Modelo con {sum(p.numel() for p in model.parameters())} parámetros.")
    
    for epoch in range(3001):
        optimizer.zero_grad()
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"  Epoch {epoch} | Loss: {loss.item():.2e}")
            # Ver qué operaciones está eligiendo el primer hijo
            with torch.no_grad():
                gates = F.softmax(model.lateral.op_select[0], dim=0)
                print(f"    Child 0 Ops: Sum:{gates[0]:.2f}, Sub:{gates[1]:.2f}, Mul:{gates[2]:.2f}, Mod:{gates[3]:.2f}")

if __name__ == "__main__":
    run_lateral_test()
