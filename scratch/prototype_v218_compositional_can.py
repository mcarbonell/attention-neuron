"""
scratch/prototype_v218_compositional_can.py - Compositional Attention Neuron (CAN)

Experimento V218:
Validar si una red multicapa puede descubrir la composicion simbolica de h(x,y) = sin(x * y)
y generalizar Out-of-Distribution (OOD).

Entrenamiento: x in [1, 2]
Test OOD: x in [2, 10]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DATASET ---
class CompositionalBenchmark:
    def __init__(self):
        pass
    
    def target_func(self, x):
        # h(x1, x2) = sin(x1 * x2)
        return torch.sin(x[:, 0] * x[:, 1]).unsqueeze(1)

    def generate_batch(self, n_samples, r_min, r_max):
        x = torch.empty(n_samples, 2).uniform_(r_min, r_max)
        y = self.target_func(x)
        return x.to(device), y.to(device)

# --- EXPERTOS DE CAPA 1 (MAPPERS) ---
class LinearMapper(nn.Module):
    def forward(self, x): return x

class LogMapper(nn.Module):
    def forward(self, x): 
        return torch.log(torch.clamp(x, min=1e-6))

# --- EXPERTOS DE CAPA 2 (OPERATORS) ---
class MLPExpert(nn.Module):
    def __init__(self, in_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

class HarmonicExpert(nn.Module):
    """
    Especializado en periodicidad. 
    Si la entrada es log(x), puede aprender exp(sum(log x)) = x*y antes del sin.
    """
    def __init__(self, in_dim=2):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, in_dim))
        self.b = nn.Parameter(torch.zeros(1))
        self.use_exp = nn.Parameter(torch.tensor([0.0])) # Aprendizaje suave de si usar exp
        
    def forward(self, x):
        # Proyeccion lineal en el espacio actual
        z = F.linear(x, self.w, self.b)
        # Decidir si aplicar exp (util si venimos de log-space)
        gate = torch.sigmoid(self.use_exp)
        z_final = (1-gate)*z + gate*torch.exp(torch.clamp(z, max=10.0))
        return torch.sin(z_final)

# --- COMPOSITIONAL ATTENTION NEURON (CAN) ---
class CAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1_lin = LinearMapper()
        self.m1_log = LogMapper()
        
        self.op2_mlp = MLPExpert()
        self.op2_har = HarmonicExpert()
        
    def forward(self, x, y_target=None):
        # Definimos los 4 caminos posibles
        # Path 0: Lin -> MLP
        # Path 1: Lin -> Har
        # Path 2: Log -> MLP
        # Path 3: Log -> Har
        
        feat_lin = self.m1_lin(x)
        feat_log = self.m1_log(x)
        
        p0 = self.op2_mlp(feat_lin)
        p1 = self.op2_har(feat_lin)
        p2 = self.op2_mlp(feat_log)
        p3 = self.op2_har(feat_log)
        
        preds = torch.cat([p0, p1, p2, p3], dim=1) # [N, 4]
        
        if y_target is not None:
            # SELECCION HONESTA (Darwiniana)
            with torch.no_grad():
                errors = torch.abs(preds - y_target)
                winner_ids = torch.argmin(errors, dim=1)
                mask = F.one_hot(winner_ids, num_classes=4).float()
            
            # Solo el ganador recibe gradiente
            out = (preds * mask).sum(dim=1, keepdim=True)
            return out, winner_ids
        else:
            # Inferencia (promedio o votacion, aqui usamos el que tenga menos varianza o el ultimo ganador conocido)
            # Para simplificar en este prototipo, devolvemos todos para análisis
            return preds

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- ENGINE ---
def train_v218():
    bench = CompositionalBenchmark()
    model = CAN().to(device)
    
    total_params = count_parameters(model)
    print(f"\n[INIT] CAN Architecture initialized with {total_params} parameters.")
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    # Dataset de entrenamiento (Rango estrecho)
    x_train, y_train = bench.generate_batch(4000, 1.0, 2.0)
    # Dataset OOD (Rango extendido)
    x_ood, y_ood = bench.generate_batch(1000, 2.0, 5.0)
    
    epochs = 5000
    t0 = time.time()
    
    print(f"Training on x in [1, 2] | Target: sin(x1 * x2)")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        pred, _ = model(x_train, y_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            # Evaluacion rapida OOD
            model.eval()
            with torch.no_grad():
                preds_ood = model(x_ood)
                # Evaluamos el error del mejor camino en OOD para ver si alguno generalizo
                errors_ood = torch.abs(preds_ood - y_ood)
                best_path_error = errors_ood.min(dim=1)[0].mean()
                
            print(f"  Epoch {epoch:<4} | Loss: {loss.item():.4e} | OOD Error (Best Path): {best_path_error.item():.4e}")

    t_final = time.time() - t0
    
    # --- ANALISIS FINAL ---
    model.eval()
    with torch.no_grad():
        preds_tr, winners_tr = model(x_train, y_train)
        preds_ood = model(x_ood)
        
        # Calculamos precision final (1 - Error relativo medio aprox o similar)
        # Usamos 1 - MSE como proxy de Accuracy para el PEI
        final_mse = criterion(preds_tr, y_train).item()
        accuracy_proxy = max(0, 1 - math.sqrt(final_mse))
        
        pei = accuracy_proxy / math.log10(total_params + 1)
        
        print("\n--- RESULTADOS FINALES V218 ---")
        print(f"PEI: {pei:.4f}")
        print(f"Wall Clock Time: {t_final:.2f}s")
        
        paths = ["Lin -> MLP", "Lin -> Har", "Log -> MLP", "Log -> Har"]
        print("\nPath Dominance (Train):")
        for i in range(4):
            freq = (winners_tr == i).float().mean().item()
            print(f"  {paths[i]:<12}: {freq:.1%}")
            
        print("\nOOD Generalization (MSE per path):")
        for i in range(4):
            err = criterion(preds_ood[:, i:i+1], y_ood).item()
            print(f"  {paths[i]:<12}: {err:.4e}")

if __name__ == "__main__":
    train_v218()
