"""
scratch/prototype_v212_dge_symbolic.py — Red Simbólica Continua con DGE

Experimento de frontera (V212):
Corrigiendo el paradigma DGE. En V211 forzamos la cuantización de los PARÁMETROS 
con un delta masivo, lo que destruyó la estabilidad.
El DGE Optimizer brilla cuando los PARÁMETROS son continuos y se optimizan con un
delta diminuto (1e-3), pero las ACTIVACIONES pasan por operadores salvajemente 
no-diferenciables (signo, módulo, redondeo).

Al perturbar los pesos continuos por un pequeño delta, algunas muestras del batch
cruzarán los umbrales discontinuos (ej. floor(x)), generando un gradiente limpio
y exacto a través de la caja negra.

Arquitectura:
Construimos un "Symbolic Layer" donde cada neurona extrae dos variables continuas (v1, v2)
y las hace pasar por 4 operadores puros (+, *, %, floor). Luego, un peso continuo
selecciona/mezcla los resultados.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

DGE_PATH = r"C:\Users\mrcm_\Local\proj\algorithms\dge-optimizer"
if DGE_PATH not in sys.path:
    sys.path.append(DGE_PATH)

try:
    from dge.torch_optimizer import TorchDGEOptimizer
except ImportError:
    print(f"❌ Error: No se pudo importar TorchDGEOptimizer. Verifica la ruta: {DGE_PATH}")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModulusBenchmark:
    def __init__(self, name="mod", dim=2):
        self.name = name
        self.dim = dim
        
    def get_ranges(self):
        return [0.0, 5.0], [0.0, 10.0], [0.0, 20.0]

    def evaluate(self, x):
        val = x[:, 0]
        mod = torch.clamp(torch.abs(x[:, 1]), min=0.5)
        return torch.remainder(val, mod).unsqueeze(1)

    def generate_data(self, n_samples=3000):
        r_train, r_near, r_far = self.get_ranges()
        x_train = torch.empty(n_samples, self.dim).uniform_(*r_train)
        x_near = torch.empty(n_samples, self.dim).uniform_(*r_near)
        x_far = torch.empty(n_samples, self.dim).uniform_(*r_far)
        return {
            "train": (x_train.to(device), self.evaluate(x_train).to(device)),
            "near": (x_near.to(device), self.evaluate(x_near).to(device)),
            "far": (x_far.to(device), self.evaluate(x_far).to(device))
        }

# --- SYMBOLIC NEURON LAYER ---
class DGESymbolicLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Extracción de variables continuas
        self.w_mix1 = nn.Linear(in_dim, out_dim)
        self.w_mix2 = nn.Linear(in_dim, out_dim)
        
        # Mezclador de los operadores (Continuous Selector)
        self.w_out = nn.Parameter(torch.randn(out_dim, 4) / math.sqrt(4))
        
    def forward(self, x):
        # x: (batch, in_dim)
        v1 = self.w_mix1(x) # (batch, out_dim)
        v2 = self.w_mix2(x) # (batch, out_dim)
        
        # --- OPERADORES PROHIBIDOS POR BACKPROP ---
        o_add = v1 + v2
        o_mul = v1 * v2
        
        # Módulo puro (protegido contra singuaridades)
        v2_mod = torch.clamp(torch.abs(v2), min=1e-3)
        o_mod = torch.remainder(v1, v2_mod)
        
        # Round / Floor puro
        o_div = torch.floor(v1 / v2_mod)
        
        # Stack: (batch, out_dim, 4)
        ops = torch.stack([o_add, o_mul, o_mod, o_div], dim=-1)
        
        # Combinación lineal: Soft-Selection de operadores
        out = torch.sum(ops * self.w_out.unsqueeze(0), dim=-1)
        
        return out

class DGESymbolicNet(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=16):
        super().__init__()
        self.sym1 = DGESymbolicLayer(in_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Activación no diferenciable implícita en la capa
        h = self.sym1(x)
        return self.head(h)

def run_dge_challenge():
    bench = ModulusBenchmark()
    print("\n🌲 INICIANDO MODULUS CHALLENGE CON RED SIMBÓLICA Y DGE")
    
    model = DGESymbolicNet(in_dim=2, hidden_dim=16).to(device)
    
    flat_params = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    dim = flat_params.numel()
    
    k_blocks = max(1, int(math.sqrt(dim)))
    print(f"  Parámetros a optimizar: {dim}")
    print(f"  DGE Config: k_blocks={k_blocks}")
    
    # Configuración DGE limpia: delta pequeño para no explotar, 
    # dejando que el batch cruce las discontinuidades orgánicamente.
    dge = TorchDGEOptimizer(
        dim=dim,
        k_blocks=k_blocks,
        lr=0.01,
        delta=1e-3,       # Delta clásico y estable
        delta_decay=1.0,  
        total_steps=1000,
        device=device,
        chunk_size=32
    )
    
    data = bench.generate_data()
    x_train_full, y_train_full = data["train"]
    x_far, y_far = data["far"]
    
    criterion = nn.MSELoss(reduction='none')
    
    t0 = time.time()
    epochs = 1000
    
    for epoch in range(epochs):
        idx = torch.randperm(x_train_full.size(0))[:256]
        x_batch_dge = x_train_full[idx]
        y_batch_dge = y_train_full[idx]
        
        def f_batched(P_batch):
            P = P_batch.shape[0]
            losses = torch.empty(P, device=device)
            
            for i in range(P):
                torch.nn.utils.vector_to_parameters(P_batch[i], model.parameters())
                pred = model(x_batch_dge)
                loss_vec = criterion(pred, y_batch_dge)
                loss_vec = torch.nan_to_num(loss_vec, nan=1e6, posinf=1e6, neginf=1e6)
                losses[i] = loss_vec.mean()
                
            return losses

        flat_params, n_evals = dge.step(f_batched, flat_params)
        
        if epoch % 50 == 0:
            torch.nn.utils.vector_to_parameters(flat_params, model.parameters())
            model.eval()
            with torch.no_grad():
                loss_full = F.mse_loss(model(x_train_full), y_train_full)
            print(f"  Paso DGE {epoch} | Train MSE: {loss_full.item():.4f} | Evals: {n_evals}")

    torch.nn.utils.vector_to_parameters(flat_params, model.parameters())
    model.eval()
    with torch.no_grad():
        m_train = F.mse_loss(model(x_train_full), y_train_full).item()
        m_far = F.mse_loss(model(x_far), y_far).item()
        
    print(f"\n  Final Train MSE: {m_train:.4e}")
    print(f"  Final Far OOD MSE: {m_far:.4e}")
    print(f"  Stability Ratio: {m_far / (m_train + 1e-12):.2e}")
    print(f"  Tiempo total: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    run_dge_challenge()
