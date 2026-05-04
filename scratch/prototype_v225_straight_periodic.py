import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os

# --- 1. Modelo Periódico "Enderezado" (V225) ---
class StraightPeriodicNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Base Periódica (4 parámetros)
        self.w_freq = nn.Parameter(torch.tensor([[3.14159 / 1.0]])) # Inicializado cerca de pi
        self.b_phase = nn.Parameter(torch.zeros(1, 1))
        self.w_amp = nn.Parameter(torch.ones(1, 1))
        self.b_bias = nn.Parameter(torch.zeros(1, 1))
        
        # 2. Corrección Polinómica (4 parámetros adicionales: a*x^3 + b*x^2 + c*x + d)
        self.poly = nn.Parameter(torch.tensor([0.0, 0.0, 1.0, 0.0])) # Inicia como identidad y = x
        
    def forward(self, x):
        # Fase periódica
        z = torch.sigmoid(torch.tan(x @ self.w_freq + self.b_phase))
        
        # Corrección polinómica para enderezar la "S"
        # y = a*z^3 + b*z^2 + c*z + d
        out = self.poly[0] * (z**3) + self.poly[1] * (z**2) + self.poly[2] * z + self.poly[3]
        
        return out

# --- 2. Datos ---
def get_batch(batch_size=128):
    x = torch.rand(batch_size, 1) * 5.0
    y = x % 1.0
    return x, y

# --- 3. Entrenamiento ---
def train():
    model = StraightPeriodicNeuron()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    steps = 5000
    start_time = time.time()
    
    for step in range(steps):
        x, y_target = get_batch()
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y_target)
        loss.backward()
        optimizer.step()
        
        if step < 5 or step % 1000 == 0:
            print(f"[V225] Step {step}: Loss = {loss.item():.6f}")
            
    wall_clock = time.time() - start_time
    total_params = sum(p.numel() for p in model.parameters())
    
    return model, loss.item(), wall_clock, total_params

# --- 4. Ejecución ---
print("Entrenando V225: Straight Periodic Neuron...")
model_v225, final_loss, duration, params = train()

# Visualización
x_test = torch.linspace(0, 5, 1000).view(-1, 1)
y_gt = x_test % 1.0
y_pred = model_v225(x_test).detach()

plt.figure(figsize=(12, 6), facecolor='#0f172a')
ax = plt.gca()
ax.set_facecolor('#0f172a')

plt.plot(x_test, y_gt, 'w--', alpha=0.3, label='Ground Truth')
plt.plot(x_test, y_pred, '#10b981', label=f'Straight Periodic (Loss: {final_loss:.5f}, 8 params)', linewidth=3)

plt.title('V225: "Straightened" Periodic Neuron (Polynomial Correction)', color='white', fontsize=16)
plt.legend()
plt.grid(color='#1e293b', linestyle=':')
plt.savefig('results/figures/v225_straight_modulo.png', dpi=300)

# Guardar Resultados
results = {
    "v225_straight_periodic": {
        "final_objective": final_loss,
        "total_params": params,
        "wall_clock_time": duration,
        "pei": (1 - final_loss) / np.log10(params + 1)
    }
}

with open("results/raw/v225_straight_bench.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"\n--- RESULTADOS V225 ---")
print(f"Final Loss: {final_loss:.6f}")
print(f"Total Parámetros: {params}")
print(f"PEI: {results['v225_straight_periodic']['pei']:.4f}")
