import torch
from torch.optim import Optimizer

def orthogonalize(G, steps=5):
    """ Newton-Schulz para ortogonalizar matrices G (rows <= cols) """
    if G.ndim < 2: return G
    rows, cols = G.shape
    do_transpose = rows > cols
    if do_transpose: G = G.T
    
    # Normalización inicial más suave
    X = G / (G.norm() + 1e-7)
    
    for _ in range(steps):
        XXT = torch.mm(X, X.t())
        X = 1.5 * X - 0.5 * torch.mm(XXT, X)
    
    # En redes pequeñas, el escalado por sqrt(dim) es demasiado agresivo.
    # Usamos un escalado unitario o muy pequeño para MNIST.
    return X.T if do_transpose else X

class MuonClean(Optimizer):
    """ Muon original: Momentum + Orthogonalization """
    def __init__(self, params, lr=0.02, momentum=0.9, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(MuonClean, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p)

                mom = state['momentum']
                state['step'] += 1

                # 1. Update Momentum
                mom.mul_(group['momentum']).add_(grad, alpha=1 - group['momentum'])

                # 2. Orthogonalize (solo matrices 2D)
                if p.ndim == 2 and p.shape[0] > 1 and p.shape[1] > 1:
                    update = orthogonalize(mom)
                else:
                    update = mom

                # 3. Apply Update
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                p.add_(update, alpha=-group['lr'])

        return loss
