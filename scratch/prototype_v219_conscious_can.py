"""
scratch/prototype_v219_conscious_can.py - Conscious Attention Neuron (CAN)

Experimento V219:
La red emite un par (Valor, Confianza) y entrena ambos.
La Confianza es la prediccion del propio Loss (MSE).
Validamos si la red "sabe que no sabe" en rangos Out-of-Distribution (OOD).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import math
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DATASET ---
class CompositionalBenchmark:
    def target_func(self, x):
        return torch.sin(x[:, 0] * x[:, 1]).unsqueeze(1)

    def generate_batch(self, n_samples, r_min, r_max):
        x = torch.empty(n_samples, 2).uniform_(r_min, r_max)
        y = self.target_func(x)
        return x.to(device), y.to(device)

# --- COMPONENTES ---
class LinearMapper(nn.Module):
    def forward(self, x): return x

class LogMapper(nn.Module):
    def forward(self, x): 
        return torch.log(torch.clamp(x, min=1e-6))

class ConsciousExpert(nn.Module):
    """
    Experto que devuelve (Valor, Log_Loss_Previsto).
    Usamos log_loss para asegurar que el loss previsto sea siempre positivo.
    """
    def __init__(self, in_dim=2, type='mlp'):
        super().__init__()
        self.type = type
        if type == 'mlp':
            self.backbone = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
            self.value_head = nn.Linear(32, 1)
            self.conf_head = nn.Linear(32, 1)
        else: # Harmonic
            self.w = nn.Parameter(torch.randn(1, in_dim))
            self.b = nn.Parameter(torch.zeros(1))
            self.conf_head = nn.Linear(in_dim, 1) # Cabeza simple para confianza

    def forward(self, x):
        if self.type == 'mlp':
            feat = self.backbone(x)
            val = self.value_head(feat)
            conf = self.conf_head(feat) # Prediccion del log(MSE)
            return val, conf
        else:
            z = F.linear(x, self.w, self.b)
            val = torch.sin(z)
            conf = self.conf_head(x) # Confianza basada en el input directamente
            return val, conf

# --- CONSCIOUS CAN ---
class ConsciousCAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1_lin = LinearMapper()
        self.m1_log = LogMapper()
        
        # 4 Caminos como en V218
        self.paths = nn.ModuleList([
            ConsciousExpert(2, 'mlp'), # Lin -> MLP
            ConsciousExpert(2, 'har'), # Lin -> Har
            ConsciousExpert(2, 'mlp'), # Log -> MLP
            ConsciousExpert(2, 'har')  # Log -> Har
        ])
        
    def forward(self, x, y_target=None):
        feat_lin = self.m1_lin(x)
        feat_log = self.m1_log(x)
        
        p0_v, p0_c = self.paths[0](feat_lin)
        p1_v, p1_c = self.paths[1](feat_lin)
        p2_v, p2_c = self.paths[2](feat_log)
        p3_v, p3_c = self.paths[3](feat_log)
        
        vals = torch.cat([p0_v, p1_v, p2_v, p3_v], dim=1)
        confs = torch.cat([p0_c, p1_c, p2_c, p3_c], dim=1) # log(MSE) previstos
        
        if y_target is not None:
            with torch.no_grad():
                errors = (vals - y_target)**2 # MSE por muestra
                winner_ids = torch.argmin(errors, dim=1)
                mask = F.one_hot(winner_ids, num_classes=4).float()
            
            val_out = (vals * mask).sum(dim=1, keepdim=True)
            conf_out = (confs * mask).sum(dim=1, keepdim=True)
            actual_error = (errors * mask).sum(dim=1, keepdim=True)
            
            return val_out, conf_out, actual_error, winner_ids
        else:
            return vals, confs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- ENGINE ---
def train_v219():
    bench = CompositionalBenchmark()
    model = ConsciousCAN().to(device)
    total_params = count_parameters(model)
    print(f"[INIT] Conscious CAN initialized with {total_params} parameters.")
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    x_train, y_train = bench.generate_batch(4000, 1.0, 2.0)
    x_ood, y_ood = bench.generate_batch(1000, 2.0, 5.0)
    
    epochs = 6000
    print(f"Training 'Self-Confidence' on x in [1, 2]...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        val_pred, conf_pred, actual_mse, _ = model(x_train, y_train)
        
        # Loss de la Tarea
        task_loss = F.mse_loss(val_pred, y_train)
        
        # Loss de Confianza: predecir el log(actual_mse)
        # Usamos clamp para evitar log(0)
        target_log_mse = torch.log(actual_mse + 1e-8)
        conf_loss = F.mse_loss(conf_pred, target_log_mse.detach())
        
        total_loss = task_loss + 0.1 * conf_loss
        total_loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"  Epoch {epoch:<4} | Task Loss: {task_loss.item():.4e} | Conf Loss: {conf_loss.item():.4e}")

    # --- ANALISIS ---
    model.eval()
    with torch.no_grad():
        # Evaluación en Train
        _, conf_tr, actual_err_tr, winners_tr = model(x_train, y_train)
        
        # Evaluación en OOD
        vals_ood, confs_ood = model(x_ood)
        actual_err_ood = (vals_ood - y_ood)**2
        
        # Metrica: Correlacion entre log(Actual) y Predicho
        def get_corr(pred, actual):
            p = pred.cpu().numpy().flatten()
            a = torch.log(actual + 1e-8).cpu().numpy().flatten()
            return np.corrcoef(p, a)[0, 1]

        print("\n--- RESULTADOS FINALES V219 ---")
        path_names = ["Lin->MLP", "Lin->Har", "Log->MLP", "Log->Har"]
        
        for i in range(4):
            # Filtramos por donde ese camino fue el ganador o mas relevante
            mask = (winners_tr == i)
            if mask.sum() > 50:
                corr = get_corr(conf_tr[mask], actual_err_tr[mask])
                print(f"Path {path_names[i]:<8} | Train Corr: {corr:.4f}")
        
        # Analisis OOD (Global)
        # ¿Sube la confianza prevista en OOD?
        avg_conf_tr = torch.exp(conf_tr).mean().item()
        # Escogemos el camino Log->Har para OOD por ser el mas robusto
        avg_conf_ood = torch.exp(confs_ood[:, 3]).mean().item()
        actual_mse_ood = actual_err_ood[:, 3].mean().item()
        
        print(f"\nUbicacion | MSE Real | Confianza (MSE Previsto)")
        print(f"----------|----------|-------------------------")
        print(f"Train     | {actual_err_tr.mean().item():.4e} | {avg_conf_tr:.4e}")
        print(f"OOD       | {actual_mse_ood:.4e} | {avg_conf_ood:.4e}")

        accuracy_proxy = max(0, 1 - math.sqrt(actual_err_tr.mean().item()))
        pei = accuracy_proxy / math.log10(total_params + 1)
        print(f"\nPEI: {pei:.4f}")

if __name__ == "__main__":
    train_v219()
