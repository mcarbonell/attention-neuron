"""
scratch/prototype_v214_dual_moe.py — Dual-Space Mixture of Experts

Experimento de frontera (V214):
Validar si un Enrutador entrenado con descenso de gradiente (Adam) puede descubrir
que el Espacio Lineal es óptimo para Sumas/Restas, y que el Espacio Logarítmico
es óptimo para Multiplicaciones/Divisiones.

Se crea un "Mixed Arithmetic Benchmark" con 4 operaciones (0:+, 1:-, 2:*, 3:/).
El modelo tiene dos expertos y una puerta (router).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DATASET ---
class MixedArithmeticBenchmark:
    def __init__(self, dim=2):
        self.dim = dim
        
    def get_ranges(self):
        # Evitamos el cero y negativos estrictamente para proteger el espacio Log
        return [1.0, 5.0], [1.0, 10.0], [1.0, 20.0]

    def generate_batch(self, n_samples, r_min, r_max):
        x = torch.empty(n_samples, 2).uniform_(r_min, r_max)
        op_ids = torch.randint(0, 4, (n_samples,))
        op_onehot = F.one_hot(op_ids, num_classes=4).float()
        
        y = torch.zeros(n_samples, 1)
        
        m_add = (op_ids == 0)
        m_sub = (op_ids == 1)
        m_mul = (op_ids == 2)
        m_div = (op_ids == 3)
        
        y[m_add, 0] = x[m_add, 0] + x[m_add, 1]
        y[m_sub, 0] = x[m_sub, 0] - x[m_sub, 1]
        y[m_mul, 0] = x[m_mul, 0] * x[m_mul, 1]
        y[m_div, 0] = x[m_div, 0] / x[m_div, 1]
        
        return x.to(device), op_onehot.to(device), op_ids.to(device), y.to(device)

    def generate_data(self, n_samples=3000):
        r_tr, r_nr, r_fr = self.get_ranges()
        
        tr_data = self.generate_batch(n_samples, *r_tr)
        nr_data = self.generate_batch(n_samples, *r_nr)
        fr_data = self.generate_batch(n_samples, *r_fr)
        
        return {"train": tr_data, "near": nr_data, "far": fr_data}


# --- EXPERTOS Y ROUTER ---

class LinearExpert(nn.Module):
    """Opera puramente en el espacio Lineal"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

class LogExpert(nn.Module):
    """Opera puramente en el espacio Logarítmico"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        # Salto al espacio logarítmico
        log_x = torch.log(torch.clamp(x, min=1e-6))
        
        # Procesamiento Lineal en el espacio Logarítmico (equivale a Muls/Divs/Pows)
        log_y = self.net(log_x)
        
        # Salto de vuelta al espacio normal
        # Clampeamos para que no explote a infinitos/NaNs antes de la pérdida
        log_y = torch.clamp(log_y, max=10.0) 
        return torch.exp(log_y)

class MoERouter(nn.Module):
    """Enrutador de conocimiento: Reparte los pesos entre los expertos"""
    def __init__(self):
        super().__init__()
        # Input = 2 variables (x,y) + 4 OneHot (operador) = 6
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 2) # P_lineal, P_log
        )
    def forward(self, x, op_onehot):
        features = torch.cat([x, op_onehot], dim=1)
        logits = self.net(features)
        return F.softmax(logits, dim=1)

class DualSpaceMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert_linear = LinearExpert()
        self.expert_log = LogExpert()
        self.router = MoERouter()
        
    def forward(self, x, op_onehot):
        probs = self.router(x, op_onehot)
        
        pred_lin = self.expert_linear(x)
        pred_log = self.expert_log(x)
        
        out = probs[:, 0:1] * pred_lin + probs[:, 1:2] * pred_log
        return out, probs

# --- ENTRENAMIENTO ---
def analyze_routing(probs, op_ids):
    """Analiza y printea la probabilidad media del Experto Logarítmico por Operación"""
    ops = ["Suma (+)", "Resta (-)", "Multiplicación (*)", "División (/)"]
    print("\n  🔍 ANÁLISIS DEL ROUTER (A quién pide ayuda para cada tarea):")
    for i in range(4):
        mask = (op_ids == i)
        if mask.sum() > 0:
            p_log_mean = probs[mask, 1].mean().item()
            p_lin_mean = probs[mask, 0].mean().item()
            print(f"    {ops[i]:<18} -> Experto Lineal: {p_lin_mean:.1%} | Experto Log: {p_log_mean:.1%}")

def run_moe():
    bench = MixedArithmeticBenchmark()
    print("\n⚖️ INICIANDO EXPERIMENTO V214: DUAL-SPACE MoE (Lineal vs Logarítmico)")
    
    model = DualSpaceMoE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    data = bench.generate_data(n_samples=5000)
    x_tr, op_tr, id_tr, y_tr = data["train"]
    x_far, op_far, id_far, y_far = data["far"]
    
    t0 = time.time()
    epochs = 4000
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        pred, probs = model(x_tr, op_tr)
        loss = criterion(pred, y_tr)
        
        # Opcional: Load balancing loss (opcional, pero ayuda a que los expertos no colapsen)
        # Aquí no lo usamos para ver el comportamiento crudo y natural.
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"  Época {epoch:<4} | Train MSE: {loss.item():.4e}")

    # Evaluación Final
    model.eval()
    with torch.no_grad():
        pred_tr, probs_tr = model(x_tr, op_tr)
        pred_far, probs_far = model(x_far, op_far)
        
        m_train = criterion(pred_tr, y_tr).item()
        m_far = criterion(pred_far, y_far).item()
        
    print(f"\n  Final Train MSE: {m_train:.4e}")
    print(f"  Final Far OOD MSE: {m_far:.4e}")
    print(f"  Stability Ratio: {m_far / (m_train + 1e-12):.2e}")
    print(f"  Tiempo total: {time.time() - t0:.1f}s")
    
    # Inspeccionar las entrañas del Enrutador
    analyze_routing(probs_tr, id_tr)

if __name__ == "__main__":
    run_moe()
