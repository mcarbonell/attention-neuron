import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os

# --- 1. Definición de la Activación Periódica ---
class PeriodicSigmoid(nn.Module):
    def forward(self, x):
        # f(x) = sigmoid(tan(x))
        # Añadimos un pequeño epsilon para evitar NaNs en el punto exacto del infinito si ocurre
        return torch.sigmoid(torch.tan(x))

# --- 2. Modelos ---
class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

class PeriodicNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        # Usamos solo 4 parámetros para competir contra los 2200 del MLP
        self.w1 = nn.Parameter(torch.randn(1, 1) * 2)
        self.b1 = nn.Parameter(torch.zeros(1, 1))
        self.w2 = nn.Parameter(torch.randn(1, 1))
        self.b2 = nn.Parameter(torch.zeros(1, 1))
        self.activation = PeriodicSigmoid()
        
    def forward(self, x):
        # y = w2 * sigma(tan(w1*x + b1)) + b2
        x = x @ self.w1 + self.b1
        x = self.activation(x)
        return x @ self.w2 + self.b2

# --- 3. Datos de Entrenamiento ---
def get_batch(batch_size=128):
    x = torch.rand(batch_size, 1) * 5.0  # Rango [0, 5]
    y = x % 1.0
    return x, y

# --- 4. Loop de Entrenamiento ---
def train_model(model, name, steps=2000):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    losses = []
    start_time = time.time()
    
    for step in range(steps):
        x, y_target = get_batch()
        
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y_target)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step < 5:
            print(f"[{name}] Step {step}: Loss = {loss.item():.6f}")
            
    wall_clock = time.time() - start_time
    total_params = sum(p.numel() for p in model.parameters())
    
    return losses, wall_clock, total_params

# --- 5. Ejecución del Experimento ---
print("Iniciando Benchmarks...")
mlp = BaselineMLP()
periodic = PeriodicNeuron()

# Entrenamiento
losses_mlp, time_mlp, params_mlp = train_model(mlp, "Baseline-ReLU", steps=3000)
losses_periodic, time_periodic, params_periodic = train_model(periodic, "Periodic-Sigmoid", steps=3000)

# Métricas Finales
final_loss_mlp = np.mean(losses_mlp[-100:])
final_loss_periodic = np.mean(losses_periodic[-100:])

# Parametric Efficiency Index (Accuracy aproximada como 1 - Loss)
pei_mlp = (1 - final_loss_mlp) / np.log10(params_mlp + 1)
pei_periodic = (1 - final_loss_periodic) / np.log10(params_periodic + 1)

results = {
    "baseline_mlp": {
        "final_objective": final_loss_mlp,
        "total_params": params_mlp,
        "wall_clock_time": time_mlp,
        "pei": pei_mlp
    },
    "periodic_neuron": {
        "final_objective": final_loss_periodic,
        "total_params": params_periodic,
        "wall_clock_time": time_periodic,
        "pei": pei_periodic
    }
}

# --- 6. Guardar Resultados y Gráficas ---
os.makedirs("results/raw", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

with open("results/raw/v224_periodic_bench.json", "w") as f:
    json.dump(results, f, indent=4)

# Plot Comparison
x_test = torch.linspace(0, 5, 500).view(-1, 1)
y_gt = x_test % 1.0
y_mlp = mlp(x_test).detach()
y_periodic = periodic(x_test).detach()

plt.figure(figsize=(12, 8), facecolor='#0f172a')
ax = plt.gca()
ax.set_facecolor('#0f172a')

plt.plot(x_test, y_gt, 'w--', alpha=0.5, label='Ground Truth (Modulo)')
plt.plot(x_test, y_mlp, '#f43f5e', label='ReLU MLP (2.2k params)', linewidth=2)
plt.plot(x_test, y_periodic, '#38bdf8', label='Periodic Neuron (4 params)', linewidth=3)

plt.title('V224: Learning Modulo Function (x % 1)', color='white', fontsize=16)
plt.legend()
plt.grid(color='#1e293b', linestyle=':')
plt.savefig('results/figures/v224_modulo_comparison.png', dpi=300)

print("\n--- RESULTADOS ---")
print(f"Baseline MLP Loss: {final_loss_mlp:.6f} | PEI: {pei_mlp:.4f}")
print(f"Periodic Neuron Loss: {final_loss_periodic:.6f} | PEI: {pei_periodic:.4f}")
print(f"Ventaja de Eficiencia: {pei_periodic / pei_mlp:.1f}x")
