"""
scratch/prototype_v240_hybrid_moe.py — Mixture of Experts Híbrido (Analítico + DGE)

Este experimento (V240) valida la arquitectura de Diferenciabilidad Mixta:
1. La mayoría de la red es derivable (AnalyticExpert) y se optimiza con Adam.
2. Una minoría crítica es no-derivable (SymbolicExpert) y se optimiza con DGE.

El objetivo es resolver el "Modulus Challenge" (x % y), que es discontinuo y 
difícil para redes neuronales estándar, pero trivial para lógica simbólica.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import math
import json

# --- CONFIGURACIÓN DE RUTAS ---
DGE_PATH = r"C:\Users\mrcm_\Local\proj\algorithms\dge-optimizer"
if DGE_PATH not in sys.path:
    sys.path.append(DGE_PATH)

try:
    from dge.torch_optimizer import TorchDGEOptimizer
except ImportError:
    print(f"Error: No se pudo importar TorchDGEOptimizer. Verifica la ruta: {DGE_PATH}")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- BENCHMARK: MODULUS CHALLENGE ---
class ModulusBenchmark:
    def __init__(self, dim=2):
        self.dim = dim
        
    def evaluate(self, x):
        val = x[:, 0]
        mod = torch.clamp(torch.abs(x[:, 1]), min=0.5)
        return torch.remainder(val, mod).unsqueeze(1)

    def generate_data(self, n_samples=5000):
        # Rangos: Train [0,5], Near [0,10], Far [0,20]
        x_train = torch.empty(n_samples, self.dim).uniform_(0, 5)
        x_near = torch.empty(n_samples, self.dim).uniform_(0, 10)
        x_far = torch.empty(n_samples, self.dim).uniform_(0, 20)
        return {
            "train": (x_train.to(device), self.evaluate(x_train).to(device)),
            "near": (x_near.to(device), self.evaluate(x_near).to(device)),
            "far": (x_far.to(device), self.evaluate(x_far).to(device))
        }

# --- EXPERTO ANALÍTICO (ADAM) ---
class AnalyticExpert(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

# --- EXPERTO SIMBÓLICO (DGE) ---
class SymbolicExpert(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # Parámetros que serán optimizados por DGE
        # Proyectamos la entrada a dos valores que se usarán en el módulo
        self.w_v1 = nn.Parameter(torch.randn(1, in_dim) * 0.1)
        self.w_v2 = nn.Parameter(torch.randn(1, in_dim) * 0.1)
        # Mezclador de operadores
        self.w_mix = nn.Parameter(torch.randn(1, 2)) # Selecciona entre modulo y floor
        
    def forward(self, x):
        v1 = F.linear(x, self.w_v1)
        v2 = F.linear(x, self.w_v2)
        v2_safe = torch.clamp(torch.abs(v2), min=0.5)
        
        o_mod = torch.remainder(v1, v2_safe)
        o_floor = torch.floor(v1 / v2_safe)
        
        ops = torch.cat([o_mod, o_floor], dim=-1)
        out = torch.sum(ops * self.w_mix, dim=-1, keepdim=True)
        return out

# --- MODELO MOE HÍBRIDO ---
class HybridMoE(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64):
        super().__init__()
        self.analytic = AnalyticExpert(in_dim, hidden_dim)
        self.symbolic = SymbolicExpert(in_dim)
        
        # Gating Network (Analítica)
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        g = self.gate(x) # (batch, 2)
        
        out_a = self.analytic(x)
        out_s = self.symbolic(x)
        
        # Mezcla ponderada
        return g[:, 0:1] * out_a + g[:, 1:2] * out_s

def train_hybrid():
    bench = ModulusBenchmark()
    data = bench.generate_data()
    x_train, y_train = data["train"]
    x_far, y_far = data["far"]
    
    model = HybridMoE(in_dim=2, hidden_dim=64).to(device)
    
    # --- SEPARAR PARÁMETROS PARA OPTIMIZADORES ---
    # Parámetros DGE (solo los del experto simbólico)
    dge_params = list(model.symbolic.parameters())
    flat_dge = torch.nn.utils.parameters_to_vector(dge_params).detach().clone()
    
    # Parámetros Adam (el resto)
    adam_params = [p for n, p in model.named_parameters() if "symbolic" not in n]
    optimizer_adam = optim.Adam(adam_params, lr=0.005)
    
    # Configurar DGE
    dge_dim = flat_dge.numel()
    dge = TorchDGEOptimizer(
        dim=dge_dim,
        k_blocks=max(1, int(math.sqrt(dge_dim))),
        lr=0.02,
        delta=1e-3,
        total_steps=1000,
        device=device
    )
    
    criterion = nn.MSELoss()
    
    t0 = time.time()
    epochs = 1000
    batch_size = 256
    
    print(f"--- Iniciando Entrenamiento Hibrido (V240) ---")
    print(f"   Analytic Params: {sum(p.numel() for p in adam_params)}")
    print(f"   Symbolic Params (DGE): {dge_dim}")
    print("-" * 50)

    for epoch in range(epochs):
        # 1. PREPARAR BATCH
        idx = torch.randperm(x_train.size(0))[:batch_size]
        xb, yb = x_train[idx], y_train[idx]
        
        # 2. PASO DGE (Actualizar solo SymbolicExpert)
        def f_dge(P_batch):
            P = P_batch.shape[0]
            losses = torch.empty(P, device=device)
            # Desactivar gradientes para DGE evaluations
            with torch.no_grad():
                for i in range(P):
                    torch.nn.utils.vector_to_parameters(P_batch[i], dge_params)
                    pred = model(xb)
                    losses[i] = F.mse_loss(pred, yb)
            return losses

        flat_dge, _ = dge.step(f_dge, flat_dge)
        torch.nn.utils.vector_to_parameters(flat_dge, dge_params)
        
        # 3. PASO ADAM (Actualizar AnalyticExpert + Gate)
        model.train()
        optimizer_adam.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        
        # Asegurarse de que los gradientes de los parámetros DGE sean nulos
        # (PyTorch ya lo hace porque no están en optimizer_adam, pero por claridad)
        optimizer_adam.step()
        
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(x_train), y_train)
            print(f"  Paso {epoch} | Loss: {val_loss.item():.4f} | DGE active")

    # --- MÉTRICAS FINALES ---
    wall_clock_time = time.time() - t0
    model.eval()
    with torch.no_grad():
        final_loss = criterion(model(x_train), y_train).item()
        far_loss = criterion(model(x_far), y_far).item()
        
    total_params = sum(p.numel() for p in model.parameters())
    pei = (1.0 / (final_loss + 1e-6)) / math.log10(total_params + 1)

    results = {
        "final_objective": final_loss,
        "far_ood_mse": far_loss,
        "total_evaluations": epochs * (dge.total_k + 1), # Estimacion
        "wall_clock_time": wall_clock_time,
        "PEI": pei,
        "total_params": total_params,
        "dge_params_ratio": dge_dim / total_params
    }

    print("\n" + "="*50)
    print(f"RESULTADOS V240 HYBRID MoE")
    print(f"="*50)
    print(f"Train MSE: {final_loss:.4e}")
    print(f"Far OOD MSE: {far_loss:.4e}")
    print(f"PEI: {pei:.2f}")
    print(f"DGE Ratio: {results['dge_params_ratio']*100:.2f}%")
    print(f"Time: {wall_clock_time:.1f}s")
    print("="*50)

    # Almacenar resultados
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v240_hybrid_moe.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_hybrid_test = True
    if run_hybrid_test:
        train_hybrid()
