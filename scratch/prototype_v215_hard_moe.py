"""
scratch/prototype_v215_hard_moe.py — Hard-Routing Mixture of Experts

Experimento de frontera (V215):
Corregir el "Expert Collusion" del V214 mediante Gumbel-Softmax (hard=True).
Forzamos a que el Router elija UN solo experto por muestra, eliminando la 
posibilidad de que colaboren para compensar errores.
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
        return [1.0, 5.0], [1.0, 10.0], [1.0, 20.0]

    def generate_batch(self, n_samples, r_min, r_max):
        x = torch.empty(n_samples, 2).uniform_(r_min, r_max)
        op_ids = torch.randint(0, 4, (n_samples,))
        op_onehot = F.one_hot(op_ids, num_classes=4).float()
        y = torch.zeros(n_samples, 1)
        
        m_add, m_sub, m_mul, m_div = (op_ids == 0), (op_ids == 1), (op_ids == 2), (op_ids == 3)
        y[m_add, 0] = x[m_add, 0] + x[m_add, 1]
        y[m_sub, 0] = x[m_sub, 0] - x[m_sub, 1]
        y[m_mul, 0] = x[m_mul, 0] * x[m_mul, 1]
        y[m_div, 0] = x[m_div, 0] / x[m_div, 1]
        
        return x.to(device), op_onehot.to(device), op_ids.to(device), y.to(device)

    def generate_data(self, n_samples=3000):
        r_tr, r_nr, r_fr = self.get_ranges()
        return {
            "train": self.generate_batch(n_samples, *r_tr),
            "far": self.generate_batch(n_samples, *r_fr)
        }

# --- EXPERTOS ---

class LinearExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

class LogExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        log_x = torch.log(torch.clamp(x, min=1e-6))
        log_y = self.net(log_x)
        return torch.exp(torch.clamp(log_y, max=10.0))

# --- HARD ROUTER ---

class HardMoERouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x, op_onehot, tau=1.0, hard=True):
        features = torch.cat([x, op_onehot], dim=1)
        logits = self.net(features)
        return F.gumbel_softmax(logits, tau=tau, hard=hard)

class HardSpaceMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert_linear = LinearExpert()
        self.expert_log = LogExpert()
        self.router = HardMoERouter()
        
    def forward(self, x, op_onehot, tau=1.0):
        probs = self.router(x, op_onehot, tau=tau, hard=True)
        pred_lin = self.expert_linear(x)
        pred_log = self.expert_log(x)
        out = probs[:, 0:1] * pred_lin + probs[:, 1:2] * pred_log
        return out, probs

# --- ENGINE ---

def run_hard_moe():
    bench = MixedArithmeticBenchmark()
    print("\n🔨 INICIANDO EXPERIMENTO V215: HARD-ROUTING MoE (Gumbel-Softmax)")
    
    model = HardSpaceMoE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    
    data = bench.generate_data(n_samples=5000)
    x_tr, op_tr, id_tr, y_tr = data["train"]
    x_far, op_far, id_far, y_far = data["far"]
    
    epochs = 6000
    t0 = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        tau = max(0.1, 1.0 - epoch/4000)
        pred, probs = model(x_tr, op_tr, tau=tau)
        mse_loss = criterion(pred, y_tr)
        
        importance = probs.mean(dim=0)
        load_balance_loss = torch.sum(importance * torch.log(importance + 1e-6))
        total_loss = mse_loss + 0.01 * load_balance_loss
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"  Época {epoch:<4} | MSE: {mse_loss.item():.4e} | Tau: {tau:.2f}")

    model.eval()
    with torch.no_grad():
        pred_tr, probs_tr = model(x_tr, op_tr, tau=0.1)
        pred_fr, probs_fr = model(x_far, op_far, tau=0.1)
        m_tr = criterion(pred_tr, y_tr).item()
        m_fr = criterion(pred_fr, y_far).item()
        
    print(f"\n  Final Train MSE: {m_tr:.4e}")
    print(f"  Final Far OOD MSE: {m_fr:.4e}")
    print(f"  Stability Ratio: {m_fr / (m_tr + 1e-12):.2e}")
    
    ops = ["Suma (+)", "Resta (-)", "Multiplicación (*)", "División (/)"]
    print("\n  🔍 ANÁLISIS DE ESPECIALIZACIÓN (Decisión Binaria):")
    for i in range(4):
        mask = (id_tr == i)
        if mask.sum() > 0:
            p_lin = probs_tr[mask, 0].mean().item()
            p_log = probs_tr[mask, 1].mean().item()
            winner = "LINEAL" if p_lin > 0.5 else "LOG"
            print(f"    {ops[i]:<18} -> Elegido: {winner:<7} (P_lin: {p_lin:.1%}, P_log: {p_log:.1%})")

if __name__ == "__main__":
    run_hard_moe()
