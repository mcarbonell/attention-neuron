"""
scratch/prototype_v207_resonant_modulus.py — Modulus Challenge con Resonancia de Fase

Experimento de frontera (V207):
Ponemos a prueba el "Cerebro Rítmico" (Neuronas de Resonancia) contra el
mayor enemigo de las redes neuronales continuas: La función Módulo (x % y).

La hipótesis: Dado que las Neuronas de Fase operan de forma nativa con
periodicidad (ondas, cosenos), deberían poder aprender la "sierra" del
módulo de manera mucho más natural y estable (Extrapolación OOD) que un
MLP masivo o una red polimórfica, utilizando una fracción de los parámetros.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import math

# --- CONFIGURACIÓN DE DISPOSITIVO ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- GENERADORES DE DATOS (Mismo que V194) ---
class ModulusBenchmark:
    def __init__(self, name="mod", dim=2):
        self.name = name
        self.dim = dim
        
    def get_ranges(self):
        # x, y. y must be > 0
        return [0.0, 5.0], [0.0, 10.0], [0.0, 20.0]

    def evaluate(self, x):
        # x % y
        val = x[:, 0]
        mod = torch.clamp(torch.abs(x[:, 1]), min=0.5) # Evitar mod por valores muy pequeños
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

# --- CAPAS DE RESONANCIA (V205) ---
class FastResonantLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.phase_sintonizer = nn.Parameter(torch.rand(out_features, in_features) * 2 * math.pi)
        self.magnitude = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))

    def forward(self, x_phase):
        x_cos = torch.cos(x_phase)
        x_sin = torch.sin(x_phase)
        w_cos = torch.cos(self.phase_sintonizer) * self.magnitude
        w_sin = torch.sin(self.phase_sintonizer) * self.magnitude
        return F.relu(F.linear(x_cos, w_cos) + F.linear(x_sin, w_sin))

# --- RED RESONANTE PROFUNDA ---
class DeepResonantNet(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, phase_scale=math.pi):
        super().__init__()
        self.phase_scale = phase_scale
        # Arquitectura con ~17k parámetros para ser justa contra Poly-Deep-V193 (~28k)
        self.res1 = FastResonantLayer(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.res2 = FastResonantLayer(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.res3 = FastResonantLayer(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Mapear inputs a Fase
        x_phase = x * self.phase_scale
        
        # Resonancias en cascada (Ondas acopladas)
        x_r = self.res1(x_phase)
        x_r = self.bn1(x_r)
        
        x_r = self.res2(x_r)
        x_r = self.bn2(x_r)
        
        x_r = self.res3(x_r)
        x_r = self.bn3(x_r)
        
        # Integración lineal final de las amplitudes resonantes
        return self.head(x_r)

# --- ENGINE ---
def run_modulus_challenge():
    bench = ModulusBenchmark()
    print("\n🌊 INICIANDO MODULUS CHALLENGE CON REDES DE RESONANCIA (x % y)")
    
    # MLP y Poly ya los corriste en V194. Traemos la ResonantNet.
    models = [
        ("Resonant-Phase-V207", DeepResonantNet(2, 64, phase_scale=math.pi/2))
    ]
    
    for m_name, model in models:
        model.to(device)
        p_count = sum(p.numel() for p in model.parameters())
        print(f"\nTraining {m_name} ({p_count} params)...")
        
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        data = bench.generate_data()
        x_train, y_train = data["train"]
        x_far, y_far = data["far"]
        
        t0 = time.time()
        for epoch in range(5000): # 5000 épocas como en V194
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
    run_modulus_challenge()
