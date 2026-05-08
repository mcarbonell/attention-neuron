import torch
from torch.optim import Optimizer

class SignDS(Optimizer):
    """
    Sign-DS: Sign-based SGD with Directional Stability
    - Memoria: Solo 2 bytes por parámetro (Estabilidad Int8 + Signo Previo Int8).
    - Update: Usa sign(grad) para normalización máxima.
    - Estabilidad: Modula el LR según la consistencia temporal de los signos.
    """
    def __init__(self, params, lr=1e-4, betas=(0.99, 0.99), weight_decay=0.0, alpha=0.5):
        # beta1: Inercia de estabilidad, beta2: Inercia de signo previo (no aplica aquí, es binario)
        # Pero usamos una estructura similar a otros DS
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, alpha=alpha)
        super(SignDS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta_stab, _ = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    # 1 byte para estabilidad
                    state['exp_avg_sign'] = torch.zeros_like(p, dtype=torch.int8)
                    # 1 byte para recordar el signo anterior
                    state['prev_sign'] = torch.zeros_like(p, dtype=torch.int8)

                exp_avg_sign = state['exp_avg_sign']
                prev_sign_tensor = state['prev_sign']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # 1. Dirección: Sign-SGD puro
                current_sign_f = torch.sign(grad)
                
                # 2. Estabilidad Direccional (DS-EMA Int8)
                # Comparamos signo actual con el anterior
                prev_sign_f = prev_sign_tensor.to(torch.float32)
                consistency = (current_sign_f == prev_sign_f).float() * 2.0 - 1.0
                
                stability_f = exp_avg_sign.to(torch.float32) / 127.0
                stability_f.mul_(beta_stab).add_(consistency, alpha=1 - beta_stab)
                exp_avg_sign.copy_((stability_f * 127.0).to(torch.int8))
                
                # Guardamos el signo actual para el siguiente paso (cuantizado)
                prev_sign_tensor.copy_(current_sign_f.to(torch.int8))

                # 3. Cálculo de Ganancia
                bias_correction = 1 - beta_stab ** state['step']
                stability = stability_f / bias_correction
                gain = torch.exp(group['alpha'] * stability)
                
                # 4. Update
                p.add_(current_sign_f * gain, alpha=-group['lr'])

        return loss
