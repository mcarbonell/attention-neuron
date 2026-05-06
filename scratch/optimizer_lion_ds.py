import torch
from torch.optim import Optimizer

class LionDS(Optimizer):
    """
    Lion with Directional Stability (Lion-DS)
    - Memoria: Solo 5 bytes por parámetro (Momento + Estabilidad Int8).
    - Normalización: Usa sign(momentum) para ignorar la escala del gradiente.
    - Adaptabilidad: Usa ganancia exponencial por estabilidad de signo.
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99, 0.99), weight_decay=0.0, alpha=0.5):
        # Lion suele usar un LR 10 veces más bajo que Adam (ej: 1e-4)
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, alpha=alpha)
        super(LionDS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
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

                # 1. Lógica Lion: Update step con el signo del momento mezclado
                # c = beta1 * exp_avg + (1 - beta1) * grad
                update_direction = exp_avg.clone().mul_(beta1).add_(grad, alpha=1 - beta1).sign()
                
                # 2. Update Momentum para el siguiente paso
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

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
                
                # Lion usa el signo, lo cual normaliza por defecto
                p.add_(update_direction * gain, alpha=-group['lr'])

        return loss
