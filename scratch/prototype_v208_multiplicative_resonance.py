"""
scratch/prototype_v208_multiplicative_resonance.py — Multiplicative Phase Resonance

Experimento de frontera (V208):
Destrozando el Muro de la Discontinuidad (x % y).
Basado en el análisis de V207, la fase w1*x + w2*y es insuficiente para x % y,
ya que la frecuencia real depende de 1/y. Necesitamos modulación multiplicativa.

Implementamos la "Neurona Resonante Polimórfica":
1. Genera características multiplicativas usando la identidad logarítmica: exp(W * log(x)).
   Esto permite a la red descubrir relaciones como x/y o x*y.
2. Utiliza estas características tanto para la FASE (frecuencia dinámica)
   como para la MAGNITUD (amplitud dinámica).
3. Pasa estas dinámicas por los osciladores armónicos (cosenos).

Si la teoría es correcta, esta red no solo aprenderá el módulo, sino que
descubrirá la ecuación exacta y extrapolará con un OOD MSE casi nulo.
"""

import torch
import torch.nn as nn
import torch.optim as optim
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

# --- NEURONA DE RESONANCIA MULTIPLICATIVA ---
class MultiplicativeResonantLayer(nn.Module):
    def __init__(self, in_dim, k_oscillators, k_multipliers):
        super().__init__()
        # Generador de términos multiplicativos (ej: x/y)
        self.log_weights = nn.Linear(in_dim, k_multipliers, bias=False)
        
        # Total de características: Originales + Multiplicativas
        total_features = in_dim + k_multipliers
        
        # Sintonizador de Fase (Frecuencias dinámicas)
        self.phase_weights = nn.Linear(total_features, k_oscillators)
        
        # Sintonizador de Magnitud (Amplitudes dinámicas)
        self.mag_weights = nn.Linear(total_features, k_oscillators)

    def forward(self, x):
        # 1. Extracción Multiplicativa: exp(W * log(x))
        x_abs = torch.abs(x) + 1e-6
        log_x = torch.log(x_abs)
        multiplicative_features = torch.exp(self.log_weights(log_x))
        
        # Sign preservation for odd powers (simple approximation for positive data like Modulus)
        # En el dataset del módulo x e y son positivos, así que esto basta.
        
        # Combinamos variables lineales y variables polinómicas/racionales
        features = torch.cat([x, multiplicative_features], dim=-1)
        
        # 2. Modulación de Fase
        phase = self.phase_weights(features)
        
        # 3. Modulación de Magnitud
        magnitude = self.mag_weights(features)
        
        # 4. Resonancia Armónica
        resonance = magnitude * torch.cos(phase)
        
        return resonance

class MultiplicativeResonantNet(nn.Module):
    def __init__(self, in_dim=2, k_oscillators=32, k_multipliers=8):
        super().__init__()
        self.res_layer = MultiplicativeResonantLayer(in_dim, k_oscillators, k_multipliers)
        # Proyección lineal final que suma los armónicos (Serie de Fourier)
        self.head = nn.Linear(k_oscillators, 1)

    def forward(self, x):
        resonance = self.res_layer(x)
        return self.head(resonance)

# --- ENGINE ---
def run_multiplicative_challenge():
    bench = ModulusBenchmark()
    print("\n🎸 INICIANDO MODULUS CHALLENGE CON RESONANCIA MULTIPLICATIVA")
    
    models = [
        ("Multi-Resonant-V208", MultiplicativeResonantNet(in_dim=2, k_oscillators=32, k_multipliers=8))
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
    run_multiplicative_challenge()
