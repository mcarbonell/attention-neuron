import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

# --- CONFIGURACIÓN ---
device = torch.device('cpu')
try:
    import torch_directml
    device = torch_directml.device()
except ImportError:
    pass

# --- EXPERTOS (Polymorphic Neurons simplificados) ---

class Expert(nn.Module):
    def __init__(self, in_dim, name="Expert"):
        super().__init__()
        self.name = name
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# --- GATER (El "Adivino" de Competencia) ---

class CompetenceGater(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # 1: Prefiere A, 0: Prefiere B
        )
    def forward(self, x):
        return self.net(x)

# --- ENGINE ---

def run_competence_moe():
    print("\n>>> V199: COMPETENCE-BASED MOE (El Adivino de Errores)")
    
    # 1. Función Híbrida: Seno en negativos, Parabólica en positivos
    def target_func(x):
        res = torch.where(x < 0, torch.sin(x*5), x**2)
        return res
    
    x_train = torch.linspace(-3, 3, 2000).unsqueeze(1).to(device)
    y_train = target_func(x_train)
    
    # 2. Inicializar Expertos y Gater
    expert_a = Expert(1, "Expert-A").to(device)
    expert_b = Expert(1, "Expert-B").to(device)
    gater = CompetenceGater(1).to(device)
    
    # Optimizadores
    opt_a = optim.Adam(expert_a.parameters(), lr=0.01)
    opt_b = optim.Adam(expert_b.parameters(), lr=0.01)
    opt_g = optim.Adam(gater.parameters(), lr=0.01)
    
    criterion = nn.MSELoss(reduction='none') # Para calcular error por muestra
    
    print("\nFase 1: Entrenamiento de Expertos (Paralelo)")
    for epoch in range(1001):
        # Expertos aprenden la función completa (cada uno llegará a una solución distinta)
        opt_a.zero_grad()
        opt_b.zero_grad()
        
        pred_a = expert_a(x_train)
        pred_b = expert_b(x_train)
        
        loss_a = criterion(pred_a, y_train).mean()
        loss_b = criterion(pred_b, y_train).mean()
        
        loss_a.backward()
        loss_b.backward()
        
        opt_a.step()
        opt_b.step()
        
        if epoch % 500 == 0:
            print(f"  Epoch {epoch} | LossA: {loss_a.item():.2e} | LossB: {loss_b.item():.2e}")

    print("\nFase 2: Entrenamiento del Gater (Adivinando quién es mejor)")
    for epoch in range(1001):
        opt_g.zero_grad()
        
        with torch.no_grad():
            # Calculamos errores reales de cada experto
            err_a = criterion(expert_a(x_train), y_train)
            err_b = criterion(expert_b(x_train), y_train)
            # Target para el gater: 1 si A es mejor (menor error), 0 si B es mejor
            better_a = (err_a < err_b).float()
        
        w = gater(x_train)
        loss_g = F.binary_cross_entropy(w, better_a)
        
        loss_g.backward()
        opt_g.step()
        
        if epoch % 500 == 0:
            print(f"  Epoch {epoch} | Gater Loss: {loss_g.item():.2e}")

    # 3. Evaluación del Ensamble
    expert_a.eval()
    expert_b.eval()
    gater.eval()
    
    with torch.no_grad():
        w = gater(x_train)
        pred_moe = w * expert_a(x_train) + (1 - w) * expert_b(x_train)
        final_mse = F.mse_loss(pred_moe, y_train).item()
        
        print(f"\nResultados Finales:")
        print(f"  MSE Expert A: {F.mse_loss(expert_a(x_train), y_train).item():.2e}")
        print(f"  MSE Expert B: {F.mse_loss(expert_b(x_train), y_train).item():.2e}")
        print(f"  MSE MoE Ensemble: {final_mse:.2e}")
        
        # Ver si el gater aprendió la frontera x=0
        w_neg = w[x_train < 0].mean().item()
        w_pos = w[x_train > 0].mean().item()
        print(f"\nComportamiento del Gater:")
        print(f"  Preferencia en x < 0 (Seno): {'A' if w_neg > 0.5 else 'B'} (w={w_neg:.2f})")
        print(f"  Preferencia en x > 0 (Parab): {'A' if w_pos > 0.5 else 'B'} (w={w_pos:.2f})")

if __name__ == "__main__":
    run_competence_moe()
