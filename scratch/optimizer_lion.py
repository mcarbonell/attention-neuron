import torch
from torch.optim import Optimizer

class Lion(Optimizer):
    """
    Lion: Optimized sign-based optimizer by Google.
    Memory: 4 bytes/p (Momentum).
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(Lion, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Lion update
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1 - beta1).sign()
                p.add_(update, alpha=-group['lr'])

                # Momentum update
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
