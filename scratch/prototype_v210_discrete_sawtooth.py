"""
scratch/prototype_v210_discrete_sawtooth.py — Discrete Analytic Sawtooth Neuron

Experimento de frontera (V210):
Fusionamos la discontinuidad perfecta (Sawtooth) con la estabilidad OOD
absoluta de los Logaritmos Discretos (Integers).

Concepto Clave: Un error continuo en la Amplitud (ej. 0.98 * y) produce
un pequeño error constante. Pero un error continuo en la FASE (ej. 0.98 * x/y)
provoca una desincronización acumulativa masiva en extrapolación.
Por tanto, la FASE debe construirse con pesos ESTRICTAMENTE ENTEROS,
mientras que la MAGNITUD puede ser un hiperplano continuo suave.

Usamos un Straight-Through Estimator (STE) para obligar a la red a usar
enteros limpios durante el forward pass, pero permitiendo gradientes en el backward.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import math

# --- CONFIGURACIÓN DE DISPOSITIVO ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- GENERADORES DE DATOS ---
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

# --- STRAIGHT-THROUGH DISCRETE LOG LAYER ---
class DiscreteLogLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Inicializamos cerca de 0 para que empiece eligiendo exponentes 0 o 1
        self.w_log = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
    def forward(self, x):
        x_abs = torch.abs(x) + 1e-6
        log_x = torch.log(x_abs)
        
        # STE: En forward es un entero estricto, en backward pasa el gradiente intacto
        w_discrete = self.w_log - self.w_log.detach() + torch.round(self.w_log).detach()
        
        # exp(W * log(x)) = prod(x_i ^ W_i)
        return torch.exp(F.linear(log_x, w_discrete))

# --- ANALYTIC SAWTOOTH NEURON ---
class AnalyticSawtoothLayer(nn.Module):
    def __init__(self, in_dim, k_oscillators):
        super().__init__()
        self.discrete_log = DiscreteLogLayer(in_dim, k_oscillators)
        
        # Fase estrictamente entera (para evitar Phase Drift OOD)
        total_features = in_dim + k_oscillators
        self.w_phase = nn.Parameter(torch.randn(k_oscillators, total_features) * 0.1)
        
        # Magnitud continua suave (los errores de amplitud no acumulan desincronización)
        self.mag_proj = nn.Linear(total_features, k_oscillators)

    def forward(self, x):
        f_log = self.discrete_log(x)
        features = torch.cat([x, f_log], dim=-1)
        
        # STE Round para la Fase
        w_phase_discrete = self.w_phase - self.w_phase.detach() + torch.round(self.w_phase).detach()
        phase = F.linear(features, w_phase_discrete)
        
        mag = self.mag_proj(features)
        
        # Sawtooth No Centrado: p - floor(p)
        # Esto va de 0 a 1 de forma asimétrica, perfecto para x % y = y * (x/y - floor(x/y))
        sawtooth = phase - torch.floor(phase)
        
        return mag * sawtooth

# --- RED ANALÍTICA PROFUNDA ---
class AnalyticSawtoothNet(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=32):
        super().__init__()
        self.res1 = AnalyticSawtoothLayer(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) # Ojo, bn en features lógicos puede distorsionar, probamos
        
        self.res2 = AnalyticSawtoothLayer(hidden_dim, hidden_dim)
        
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x_r = self.res1(x)
        # BatchNorm1d no afecta la fase, solo la amplitud que pasa a la sig capa
        x_r = self.bn1(x_r) 
        x_r = self.res2(x_r)
        
        return self.head(x_r)

# --- ENGINE ---
def run_analytic_challenge():
    bench = ModulusBenchmark()
    print("\n🔬 INICIANDO MODULUS CHALLENGE CON NEURONA ANALÍTICA DISCRETA")
    
    models = [
        ("Analytic-Sawtooth-V210", AnalyticSawtoothNet(in_dim=2, hidden_dim=32))
    ]
    
    for m_name, model in models:
        model.to(device)
        p_count = sum(p.numel() for p in model.parameters())
        print(f"\nTraining {m_name} ({p_count} params)...")
        
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        data = bench.generate_data()
        x_train, y_train = data["train"]
        x_far, y_far = data["far"]
        
        t0 = time.time()
        for epoch in range(5000):
            optimizer.zero_grad()
            pred = model(x_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()
            
            if epoch % 1000 == 0:
                print(f"  Epoch {epoch} | Loss: {loss.item():.2e}")
                
        model.eval()
        with torch.no_grad():
            m_train = criterion(model(x_train), y_train).item()
            m_far = criterion(model(x_far), y_far).item()
            
        print(f"  Final Train MSE: {m_train:.4e}")
        print(f"  Final Far OOD MSE: {m_far:.4e}")
        print(f"  Stability Ratio: {m_far / (m_train + 1e-12):.2e}")

if __name__ == "__main__":
    run_analytic_challenge()
