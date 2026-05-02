"""
scratch/prototype_v209_sawtooth_resonance.py — Sawtooth Resonant Neuron

Experimento de frontera (V209):
Eliminamos el "Fenómeno de Gibbs" sustituyendo el oscilador armónico continuo (Coseno)
por un oscilador discontinuo nativo (Sawtooth/Sierra).
La base matemática cambia de cos(Phase) a (Phase - floor(Phase)).

Esta neurona está diseñada específicamente para dominios lógicos y aritméticos
donde se requieren saltos abruptos. Comparamos directamente contra la red V207
(que usaba cosenos) bajo la misma arquitectura y conteo de parámetros.
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

# --- NEURONA DE SIERRA (SAWTOOTH) ---
class SawtoothResonantLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Sintonizadores lineales simples (como en V207, sin multiplicativos explosivos)
        self.phase_sintonizer = nn.Linear(in_features, out_features)
        self.magnitude = nn.Linear(in_features, out_features)

    def forward(self, x):
        phase = self.phase_sintonizer(x)
        mag = self.magnitude(x)
        
        # Onda de Sierra: Periódica entre -1 y 1
        # phase - floor(phase + 0.5) extrae el residuo fraccionario centrado en 0
        sawtooth = 2.0 * (phase - torch.floor(phase + 0.5))
        
        # Opcional: El gradiente de floor es 0, por lo que backprop pasará a través de 'phase'
        # con gradiente constante 1. Esto actúa como un Straight-Through Estimator nativo.
        
        return mag * sawtooth

# --- RED DE SIERRA PROFUNDA ---
class DeepSawtoothNet(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64):
        super().__init__()
        # Arquitectura idéntica a V207 para comparación justa (~17k parámetros)
        self.res1 = SawtoothResonantLayer(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.res2 = SawtoothResonantLayer(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.res3 = SawtoothResonantLayer(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x_r = self.res1(x)
        x_r = self.bn1(x_r)
        
        x_r = self.res2(x_r)
        x_r = self.bn2(x_r)
        
        x_r = self.res3(x_r)
        x_r = self.bn3(x_r)
        
        return self.head(x_r)

# --- ENGINE ---
def run_sawtooth_challenge():
    bench = ModulusBenchmark()
    print("\n🪚 INICIANDO MODULUS CHALLENGE CON NEURONAS SAWTOOTH (x % y)")
    
    models = [
        ("Sawtooth-Resonant-V209", DeepSawtoothNet(in_dim=2, hidden_dim=64))
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
    run_sawtooth_challenge()
