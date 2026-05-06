import torch
from torch.optim import Optimizer
import math

class AdamDS(Optimizer):
    """
    Adam with Directional Stability (Adam-DS)
    Ajusta el learning rate basándose en la consistencia del signo del gradiente
    a lo largo del tiempo (DS-EMA).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.99), eps=1e-8, 
                 weight_decay=0, alpha=0.5):
        # beta3 es para la estabilidad de signo (0.99 por defecto)
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, alpha=alpha)
        super(AdamDS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avg_signs = [] # El nuevo estado: Consistencia de Signo
            state_steps = []

            beta1, beta2, beta3 = group['betas']
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)

                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Inicializamos como Int8 (1 byte por parámetro)
                        state['exp_avg_sign'] = torch.zeros_like(p, dtype=torch.int8)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    exp_avg_signs.append(state['exp_avg_sign'])
                    state['step'] += 1
                    state_steps.append(state['step'])

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                exp_avg_sign = exp_avg_signs[i]
                step = state_steps[i]

                if group['weight_decay'] != 0:
                    grad = grad.add(param, alpha=group['weight_decay'])

                # 1. Update standard Adam moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 2. Update Directional Stability (DS-EMA) con Cuantización Int8
                current_sign = torch.sign(grad)
                prev_sign = torch.sign(exp_avg)
                consistency = (current_sign == prev_sign).float() * 2.0 - 1.0
                
                # Descomprimimos Int8 -> Float32 (-1.0 a 1.0)
                stability_f = exp_avg_sign.to(torch.float32) / 127.0
                
                # Actualizamos el EMA en float
                stability_f.mul_(beta3).add_(consistency, alpha=1 - beta3)
                
                # Comprimimos de nuevo Float32 -> Int8
                exp_avg_sign.copy_((stability_f * 127.0).to(torch.int8))

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                bias_correction3 = 1 - beta3 ** step

                # 3. Calculate gain based on stability
                # Usamos la versión corregida (bias correction) de la estabilidad
                stability = stability_f / bias_correction3
                gain = torch.exp(group['alpha'] * stability)

                # 4. Final update
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                
                # Modificamos la actualización para soportar el gain per-parámetro
                update = (exp_avg / denom) * gain
                param.add_(update, alpha=-step_size)

        return loss
