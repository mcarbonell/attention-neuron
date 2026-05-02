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

# --- CAPA CON PARÁMETROS ACOTADOS [-1, 1] (V200) ---

class BoundedLinear(nn.Module):
    """
    Una capa lineal donde los pesos y bias están garantizados en [-1, 1].
    Utiliza un factor de escala externo para recuperar el rango dinámico.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Parámetros 'crudos' (sin acotar)
        self.raw_weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.raw_bias = nn.Parameter(torch.randn(out_dim))
        
        # Factor de escala aprendible (un escalar por neurona)
        self.scale = nn.Parameter(torch.ones(out_dim))

    def get_bounded_params(self):
        # Forzamos los pesos y bias a [-1, 1] mediante tanh o clamp
        w = torch.tanh(self.raw_weight)
        b = torch.tanh(self.raw_bias)
        return w, b

    def forward(self, x):
        w, b = self.get_bounded_params()
        # Aplicamos la operación lineal
        z = F.linear(x, w, b)
        # Re-escalamos el resultado
        return z * self.scale

# --- NEURONA POLIMÓRFICA ACOTADA ---

class BoundedPolyNeuron(nn.Module):
    def __init__(self, in_dim, hidden_dim=32):
        super().__init__()
        # En V200, todas nuestras proyecciones son Bounded
        self.proj_res = BoundedLinear(in_dim, 16) # Osciladores
        self.proj_log = BoundedLinear(in_dim, 8)  # Interacciones
        
        # Integración final acotada
        total_in = 16 + 8 + 3 * in_dim # Res + Log + Structural
        self.integrator = BoundedLinear(total_in, hidden_dim)
        self.head = BoundedLinear(hidden_dim, 1)

    def forward(self, x):
        # Rama Resonante (acotada)
        res_f = torch.sin(self.proj_res(x))
        # Rama Logarítmica (acotada)
        log_f = torch.exp(self.proj_log(torch.log(torch.abs(x) + 1e-6)))
        
        # Bases Estructurales (estas son fijas, no necesitan acotarse)
        b1, b2, b3 = x, x**2, 1.0/(x+1e-6)
        p_f = torch.cat([b1, b2, b3], dim=1) # 3 * in_dim
        
        combined = torch.cat([res_f, log_f, p_f], dim=1)
        # Aseguramos que el combined no explote antes del integrator
        combined = torch.clamp(combined, -10, 10)
        
        z = torch.tanh(self.integrator(combined))
        return self.head(z)

# --- ENGINE ---

def run_bounded_test():
    print("\n>>> V200: BOUNDED PARAMETERS CHALLENGE (Lego Units)")
    
    # Función objetivo compleja
    def target_func(x):
        return torch.sin(x*3) + x**2

    x_train = torch.linspace(-2, 2, 2000).unsqueeze(1).to(device)
    y_train = target_func(x_train)

    model = BoundedPolyNeuron(1, 32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Entrenando modelo con pesos restringidos a [-1, 1]...")
    
    for epoch in range(2001):
        optimizer.zero_grad()
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            # Monitorizar el factor de escala más grande
            max_s = torch.max(torch.abs(model.head.scale)).item()
            print(f"  Epoch {epoch} | Loss: {loss.item():.2e} | Max Scale: {max_s:.2f}")

    # Verificación de rangos
    with torch.no_grad():
        w, b = model.integrator.get_bounded_params()
        print(f"\nVerificación de Rangos (Capa Integrador):")
        print(f"  Min W: {w.min().item():.4f} | Max W: {w.max().item():.4f}")
        print(f"  Min B: {b.min().item():.4f} | Max B: {b.max().item():.4f}")
        print(f"  Scale (media): {model.integrator.scale.abs().mean().item():.2f}")

if __name__ == "__main__":
    run_bounded_test()
