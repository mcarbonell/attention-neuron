"""
scratch/prototype_v211_dge_modulus.py — DGE Optimizer + Pure Analytic Network

Experimento de frontera (V211):
Cruzando el rubicón hacia la optimización libre de gradientes.
Sustituimos Adam y Backpropagation por el DGE Optimizer (Denoised Gradient Estimation).

CORRECCIÓN DE SINGULARIDAD:
Apilar capas de logaritmos que pueden generar exponentes negativos (ej. 1/x) sobre
las activaciones internas (que cruzan el cero) genera divisiones por cero y explosiones
a infinito (NaN). Por tanto:
1. Usamos solo UNA capa Analítica profunda (es suficiente para resolver el módulo).
2. Clampeamos los exponentes discretos a [-1, 0, 1] para evitar polinomios salvajes.
3. Protegemos la función objetivo contra NaNs estocásticos.

CORRECCIÓN DE RUIDO DGE (V3 FIX):
El error explotaba continuamente porque `f_batched` era llamado múltiples veces
por el mecanismo de `chunk_size` del optimizador DGE, y en CADA llamada generaba
un sub-batch aleatorio nuevo. Esto destruía la coherencia de las diferencias finitas
(comparaba L+ en un batch contra L- en otro batch). Ahora el sub-batch se fija
FUERA de la función f_batched durante cada step completo del DGE.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

# --- IMPORTAR DGE OPTIMIZER ---
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

class PureDiscreteLogLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w_log = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
    def forward(self, x):
        x_abs = torch.abs(x) + 1e-4
        log_x = torch.log(x_abs)
        
        w_discrete = torch.round(self.w_log)
        w_discrete = torch.clamp(w_discrete, min=-1.0, max=1.0)
        
        out = torch.exp(F.linear(log_x, w_discrete))
        return torch.clamp(out, max=1e5)

class PureAnalyticSawtoothLayer(nn.Module):
    def __init__(self, in_dim, k_oscillators):
        super().__init__()
        self.discrete_log = PureDiscreteLogLayer(in_dim, k_oscillators)
        
        total_features = in_dim + k_oscillators
        self.w_phase = nn.Parameter(torch.randn(k_oscillators, total_features) * 0.1)
        self.mag_proj = nn.Linear(total_features, k_oscillators)

    def forward(self, x):
        f_log = self.discrete_log(x)
        features = torch.cat([x, f_log], dim=-1)
        
        w_phase_discrete = torch.clamp(torch.round(self.w_phase), min=-1.0, max=1.0)
        phase = F.linear(features, w_phase_discrete)
        
        mag = self.mag_proj(features)
        sawtooth = phase - torch.floor(phase)
        
        return mag * sawtooth

class PureAnalyticNet(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64):
        super().__init__()
        self.res1 = PureAnalyticSawtoothLayer(in_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x_r = self.res1(x)
        return self.head(x_r)

def run_dge_challenge():
    bench = ModulusBenchmark()
    print("\n🧬 INICIANDO MODULUS CHALLENGE CON OPTIMIZADOR DGE (LIBRE DE GRADIENTES) - V3 FIX")
    
    model = PureAnalyticNet(in_dim=2, hidden_dim=64).to(device)
    
    flat_params = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    dim = flat_params.numel()
    
    print(f"  Parámetros a optimizar: {dim}")
    
    k_blocks = int(math.sqrt(dim))
    print(f"  DGE Config: k_blocks={k_blocks}")
    
    dge = TorchDGEOptimizer(
        dim=dim,
        k_blocks=k_blocks,
        lr=0.05,          # LR más bajo y estable
        delta=0.8,        # Delta alto para saltos enteros
        delta_decay=0.8,  
        total_steps=1000,
        device=device,
        chunk_size=32     # División en chunks para no ahogar la RAM
    )
    
    data = bench.generate_data()
    x_train_full, y_train_full = data["train"]
    x_far, y_far = data["far"]
    
    criterion = nn.MSELoss(reduction='none')
    
    t0 = time.time()
    epochs = 1000
    
    for epoch in range(epochs):
        # Generar el batch estocástico UNA SOLA VEZ por cada paso del DGE
        # Esto garantiza que las diferencias finitas (L+ vs L-) sean consistentes
        idx = torch.randperm(x_train_full.size(0))[:256]
        x_batch_dge = x_train_full[idx]
        y_batch_dge = y_train_full[idx]
        
        def f_batched(P_batch):
            P = P_batch.shape[0]
            losses = torch.empty(P, device=device)
            
            # Usamos el mismo x_batch_dge para todas las llamadas del chunk
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
