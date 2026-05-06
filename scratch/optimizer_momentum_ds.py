import torch
from torch.optim import Optimizer
import math

class MomentumDS(Optimizer):
    """
    Momentum with Directional Stability (Momentum-DS)
    Versión ultra-ligera: ELIMINA la varianza (v) de Adam.
    Solo usa Momento (4 bytes) + Estabilidad Int8 (1 byte).
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta3=0.99, weight_decay=0, alpha=1.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        defaults = dict(lr=lr, beta1=beta1, beta3=beta3, weight_decay=weight_decay, alpha=alpha)
        super(MomentumDS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1 = group['beta1']
            beta3 = group['beta3']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # Inicialización de estados
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p) # Momento (Float32)
                    state['exp_avg_sign'] = torch.zeros_like(p, dtype=torch.int8) # Estabilidad (Int8)

                exp_avg = state['exp_avg']
                exp_avg_sign = state['exp_avg_sign']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # 1. Actualizar Momento (Media de gradientes)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 2. Actualizar Estabilidad Direccional (Int8)
                current_sign = torch.sign(grad)
                prev_sign = torch.sign(exp_avg)
                consistency = (current_sign == prev_sign).float() * 2.0 - 1.0
                
                # Descomprimir -> Actualizar -> Comprimir
                stability_f = exp_avg_sign.to(torch.float32) / 127.0
                stability_f.mul_(beta3).add_(consistency, alpha=1 - beta3)
                exp_avg_sign.copy_((stability_f * 127.0).to(torch.int8))

                # 3. Calcular Ganancia y Actualizar
                # Sin varianza abajo, usamos el momento directamente escalado por la estabilidad
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction3 = 1 - beta3 ** state['step']
                
                stability = stability_f / bias_correction3
                gain = torch.exp(group['alpha'] * stability)
                
                # Aplicamos el paso: El momento 'limpia' el gradiente, la estabilidad lo escala
                update = (exp_avg / bias_correction1) * gain
                p.add_(update, alpha=-group['lr'])

        return loss
