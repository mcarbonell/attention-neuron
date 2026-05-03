"""
scratch/prototype_v217_honest_moe.py — Honest Mixture of Experts

Experimento de frontera (V217):
Sustituir el Router neuronal por una selección basada en el error mínimo
instantáneo. Cada experto compite por el derecho a aprender de la muestra.
El que tiene menos error se queda con el gradiente (Winner-Takes-All).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DATASET ---
class MixedArithmeticBenchmark:
    def __init__(self, dim=2): self.dim = dim
    def get_ranges(self): return [1.0, 1000.0], [1.0, 10.0]
    def generate_batch(self, n_samples, r_min, r_max):
        x = torch.empty(n_samples, 2).uniform_(r_min, r_max)
        op_ids = torch.randint(0, 4, (n_samples,))
        op_onehot = F.one_hot(op_ids, num_classes=4).float()
        y = torch.zeros(n_samples, 1)
        m_add, m_sub, m_mul, m_div = (op_ids==0), (op_ids==1), (op_ids==2), (op_ids==3)
        y[m_add,0] = x[m_add,0] + x[m_add,1]
        y[m_sub,0] = x[m_sub,0] - x[m_sub,1]
        y[m_mul,0] = x[m_mul,0] * x[m_mul,1]
        y[m_div,0] = x[m_div,0] / x[m_div,1]
        return x.to(device), op_onehot.to(device), op_ids.to(device), y.to(device)

# --- EXPERTOS ---
class LinearExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x)

class LogExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        log_x = torch.log(torch.clamp(x, min=1e-6))
        return torch.exp(torch.clamp(self.net(log_x), max=10.0))

class HonestMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert_linear = LinearExpert()
        self.expert_log = LogExpert()
        
    def forward(self, x, y_target):
        p_lin = self.expert_linear(x)
        p_log = self.expert_log(x)
        
        # --- SELECCIÓN COMPETITIVA POR ERROR ---
        with torch.no_grad():
            err_lin = torch.abs(p_lin - y_target)
            err_log = torch.abs(p_log - y_target)
            mask_lin = (err_lin < err_log).float()
            mask_log = 1.0 - mask_lin
            
        out = mask_lin * p_lin + mask_log * p_log
        return out, mask_lin

# --- ENGINE ---
def run_honest_moe():
    bench = MixedArithmeticBenchmark()
    print("\n⚔️ INICIANDO EXPERIMENTO V217: HONEST MoE (Competencia por Error)")
    
    model = HonestMoE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    
    tr_data = bench.generate_batch(5000, 1.0, 5.0)
    x_tr, _, id_tr, y_tr = tr_data
    
    epochs = 8000
    t0 = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pred, mask_lin = model(x_tr, y_tr)
        loss = criterion(pred, y_tr)
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"  Época {epoch:<4} | Train MSE: {loss.item():.4e}")

    # Análisis
    model.eval()
    with torch.no_grad():
        _, mask_lin = model(x_tr, y_tr)
        
    ops = ["Suma (+)", "Resta (-)", "Multiplicación (*)", "División (/)"]
    print("\n  🔍 GANADORES NATURALES (Quién tuvo menos error al final):")
    for i in range(4):
        mask = (id_tr == i)
        if mask.sum() > 0:
            p_lin_win = mask_lin[mask].mean().item()
            winner = "LINEAL" if p_lin_win > 0.5 else "LOG"
            print(f"    {ops[i]:<18} -> Ganador: {winner:<7} (Frecuencia Lineal: {p_lin_win:.1%})")

if __name__ == "__main__":
    run_honest_moe()
