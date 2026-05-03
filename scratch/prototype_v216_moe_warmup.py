"""
scratch/prototype_v216_moe_warmup.py — MoE con Warm-up Curriculado

Experimento de frontera (V216):
Implementar una fase de entrenamiento inicial donde ambos expertos son forzados
a aprender todas las tareas (Warm-up). Esto evita que el Router descarte a un 
experto antes de que este haya tenido oportunidad de aprender su especialidad.
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
        return [1.0, 5.0], [1.0, 20.0]
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

# --- MODELO ---
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
        log_y = self.net(log_x)
        return torch.exp(torch.clamp(log_y, max=10.0))

class HardMoERouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x, op_onehot, tau=1.0, hard=True):
        logits = self.net(torch.cat([x, op_onehot], dim=1))
        return F.gumbel_softmax(logits, tau=tau, hard=hard)

class WarmupMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert_linear = LinearExpert()
        self.expert_log = LogExpert()
        self.router = HardMoERouter()
        
    def forward(self, x, op_onehot, tau=1.0, force_uniform=False):
        if force_uniform:
            batch_size = x.shape[0]
            probs = torch.ones(batch_size, 2, device=x.device) * 0.5
        else:
            probs = self.router(x, op_onehot, tau=tau, hard=True)
            
        pred_lin = self.expert_linear(x)
        pred_log = self.expert_log(x)
        out = probs[:, 0:1] * pred_lin + probs[:, 1:2] * pred_log
        return out, probs

# --- TRAINING ---
def run_warmup_moe():
    bench = MixedArithmeticBenchmark()
    print("\n🔥 INICIANDO EXPERIMENTO V216: MoE CON WARM-UP CURRICULADO")
    
    model = WarmupMoE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    
    tr_data = bench.generate_batch(5000, 1.0, 5.0)
    x_tr, op_tr, id_tr, y_tr = tr_data
    
    epochs = 8000
    t0 = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        if epoch < 2000:
            force_uniform = True
            tau = 1.0
        else:
            force_uniform = False
            tau = max(0.1, 1.0 - (epoch-2000)/4000)
            
        pred, probs = model(x_tr, op_tr, tau=tau, force_uniform=force_uniform)
        loss = criterion(pred, y_tr)
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            phase = "WARMUP" if force_uniform else "ROUTING"
            print(f"  Época {epoch:<4} | [{phase}] MSE: {loss.item():.4e} | Tau: {tau:.2f}")

    model.eval()
    with torch.no_grad():
        pred_tr, probs_tr = model(x_tr, op_tr, tau=0.1, force_uniform=False)
        m_tr = criterion(pred_tr, y_tr).item()
        
    print(f"\n  Final Train MSE: {m_tr:.4e}")
    ops = ["Suma (+)", "Resta (-)", "Multiplicación (*)", "División (/)"]
    print("\n  🔍 MAPA MATEMÁTICO FINAL (Decisión del Router):")
    for i in range(4):
        mask = (id_tr == i)
        if mask.sum() > 0:
            p_lin = probs_tr[mask, 0].mean().item()
            winner = "LINEAL" if p_lin > 0.5 else "LOG"
            print(f"    {ops[i]:<18} -> Especialista: {winner:<7} (P_lin: {p_lin:.1%}, P_log: {1-p_lin:.1%})")

if __name__ == "__main__":
    run_warmup_moe()
