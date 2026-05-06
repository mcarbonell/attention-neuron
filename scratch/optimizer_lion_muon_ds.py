import torch
from torch.optim import Optimizer

def orthogonalize(G, steps=5):
    """
    Newton-Schulz iteration para ortogonalizar la matriz G.
    Hace que las filas de G sean ortonormales entre sí.
    """
    if G.ndim < 2:
        return G
    
    # Transponer si es necesario (queremos ortogonalizar las filas)
    rows, cols = G.shape
    if rows > cols:
        G = G.T
    
    # Normalización inicial (Spectral norm approximation)
    X = G / (G.norm() + 1e-7)
    
    # Iteraciones de Newton-Schulz (X = 1.5X - 0.5 * X * X^T * X)
    for _ in range(steps):
        XXT = torch.mm(X, X.t())
        X = 1.5 * X - 0.5 * torch.mm(XXT, X)
        
    return X if rows <= cols else X.T

class LionMuonDS(Optimizer):
    """
    Lion-Muon-DS: El optimizador definitivo.
    - Ortogonalización Muon para matrices 2D.
    - Sign-Momentum para normalización global.
    - Estabilidad Direccional (DS) para adaptabilidad de LR.
    - Memoria ultra-baja (5 bytes/p).
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99, 0.99), weight_decay=0.0, alpha=0.5):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, alpha=alpha)
        super(LionMuonDS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sign'] = torch.zeros_like(p, dtype=torch.int8)

                exp_avg = state['exp_avg']
                exp_avg_sign = state['exp_avg_sign']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # 1. Update Momentum (común para ambos)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

                # 2. Determinar dirección de actualización según dimensiones
                if p.ndim == 2 and p.shape[0] > 1 and p.shape[1] > 1:
                    # MATRICES: Usamos Muon (Ortogonalización del momentum)
                    # El momentum ya 'limpia' el gradiente, Muon lo ortogonaliza
                    update_dir = orthogonalize(exp_avg)
                    scale_factor = 1.0 # Muon ya tiene su escala
                else:
                    # VECTORES: Usamos Lion (Signo del momentum)
                    # Escalamos hacia abajo (0.1) para que no sea tan agresivo comparado con Muon
                    update_dir = torch.sign(exp_avg)
                    scale_factor = 0.1 

                # 3. Estabilidad Direccional (DS-EMA Int8)
                current_sign = torch.sign(grad)
                prev_sign = torch.sign(exp_avg)
                consistency = (current_sign == prev_sign).float() * 2.0 - 1.0
                
                stability_f = exp_avg_sign.to(torch.float32) / 127.0
                stability_f.mul_(beta3).add_(consistency, alpha=1 - beta3)
                exp_avg_sign.copy_((stability_f * 127.0).to(torch.int8))

                # 4. Aplicar paso modulado por estabilidad
                bias_correction3 = 1 - beta3 ** state['step']
                stability = stability_f / bias_correction3
                gain = torch.exp(group['alpha'] * stability)
                
                p.add_(update_dir * (gain * scale_factor), alpha=-group['lr'])

        return loss
