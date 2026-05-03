"""
scratch/prototype_v213_activation_bridge.py — The Activation Bridge

Experimento de frontera (V213):
Demostrando que una red neuronal profunda clásica (Adam) puede aprender a hablar
el lenguaje de una "Calculadora Simbólica" no-diferenciable.

V2 FIX: Evitar explosión de gradiente sintético.
Al hacer diferencias finitas sobre el operador Módulo (`%`), cuando la variable
cruza exactamente el umbral (ej. 4.999 -> 5.001 con módulo 5), el valor se desploma
de 5 a 0. Esto genera un pseudo-gradiente monstruoso de `-5 / (2*delta)` que
destruye los momentos de Adam cuando la red empieza a converger.
Solución: Gradient Clipping estricto sobre el Jacobiano estimado.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

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

class AdamFeatureExtractor(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, x):
        return self.net(x)

def fixed_symbolic_calculator(Z):
    num = Z[:, 0]
    den = torch.clamp(torch.abs(Z[:, 1]), min=1e-3)
    return torch.remainder(num, den).unsqueeze(1)

def compute_dge_activation_gradient(Z, y_batch, delta=1e-3):
    Z_base = Z.detach()
    dZ = torch.zeros_like(Z_base)
    
    criterion = nn.MSELoss(reduction='none')
    
    Z_p0 = Z_base.clone(); Z_p0[:, 0] += delta
    Z_m0 = Z_base.clone(); Z_m0[:, 0] -= delta
    L_p0 = criterion(fixed_symbolic_calculator(Z_p0), y_batch)
    L_m0 = criterion(fixed_symbolic_calculator(Z_m0), y_batch)
    dZ[:, 0] = (L_p0.squeeze() - L_m0.squeeze()) / (2 * delta)
    
    Z_p1 = Z_base.clone(); Z_p1[:, 1] += delta
    Z_m1 = Z_base.clone(); Z_m1[:, 1] -= delta
    L_p1 = criterion(fixed_symbolic_calculator(Z_p1), y_batch)
    L_m1 = criterion(fixed_symbolic_calculator(Z_m1), y_batch)
    dZ[:, 1] = (L_p1.squeeze() - L_m1.squeeze()) / (2 * delta)
    
    # CLIPPING CRÍTICO: Previene las ondas de choque cuando la diferencia finita cruza
    # el acantilado discontinuo del operador % (donde la función cae en picado a 0).
    return torch.clamp(dZ, min=-5.0, max=5.0)

def run_bridge_experiment():
    bench = ModulusBenchmark()
    print("\n🌉 INICIANDO EXPERIMENTO V213: EL PUENTE NEURO-SIMBÓLICO (V2 FIX CLIPPING)")
    
    model = AdamFeatureExtractor(in_dim=2, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005) # LR un poco más suave
    
    data = bench.generate_data(n_samples=5000)
    x_train, y_train = data["train"]
    x_far, y_far = data["far"]
    
    t0 = time.time()
    batch_size = 256
    epochs = 4000
    
    for epoch in range(epochs):
        idx = torch.randperm(x_train.size(0))[:batch_size]
        x_batch = x_train[idx]
        y_batch = y_train[idx]
        
        optimizer.zero_grad()
        Z = model(x_batch) 
        
        dZ = compute_dge_activation_gradient(Z, y_batch, delta=1e-3)
        Z.backward(gradient=dZ)
        
        # Opcional pero sano: Gradient clipping interno de PyTorch para los pesos del MLP
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 500 == 0:
            model.eval()
            with torch.no_grad():
                Z_all = model(x_train)
                pred_all = fixed_symbolic_calculator(Z_all)
                loss_full = F.mse_loss(pred_all, y_train)
            model.train()
            print(f"  Época {epoch} | Train MSE: {loss_full.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred_train = fixed_symbolic_calculator(model(x_train))
        pred_far = fixed_symbolic_calculator(model(x_far))
        
        m_train = F.mse_loss(pred_train, y_train).item()
        m_far = F.mse_loss(pred_far, y_far).item()
        
    print(f"\n  Final Train MSE: {m_train:.4e}")
    print(f"  Final Far OOD MSE: {m_far:.4e}")
    print(f"  Stability Ratio: {m_far / (m_train + 1e-12):.2e}")
    print(f"  Tiempo total: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    run_bridge_experiment()
