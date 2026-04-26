import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
from torchvision import datasets, transforms

# --- FWHT Core ---
def fwht(x):
    original_shape = x.shape
    if len(original_shape) == 1: x = x.unsqueeze(0)
    B, N = x.shape
    h = 1
    while h < N:
        x = x.view(B, N // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        h *= 2
    res = x.view(B, N)
    return res if len(original_shape) > 1 else res.squeeze(0)

def ifwht(x):
    N = x.shape[-1]
    return fwht(x) / N

# --- Stabilized Seismic Walsh Optimizer ---

class StabilizedSeismicOptimizer:
    """
    V38: THE STABILIZED SEISMIC WALSH OPTIMIZER.
    Implements Seismic Descent with an aggressive amplitude decay (cooling).
    """
    def __init__(self, parameters, total_steps, base_lr=0.01, a0=0.02, freq0=0.2):
        self.params = list(parameters)
        self.lr = base_lr
        self.a0 = a0
        self.freq0 = freq0
        self.total_steps = total_steps
        self.t = 0
        
        self.seismic_energies = []
        for p in self.params:
            n = p.numel()
            n_pow2 = 2**math.ceil(math.log2(n))
            energy = torch.randn(n_pow2, device=p.device) * 0.1
            self.seismic_energies.append(energy)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.t += 1
        
        # 1. Seismic Cooling Schedule
        # Amplitud decae linealmente hacia cero
        decay = max(0.0, 1.0 - (self.t / self.total_steps))
        current_freq = self.freq0 
        amplitude = self.a0 * math.sin(self.t * current_freq) * decay
        
        with torch.no_grad():
            for p, energy in zip(self.params, self.seismic_energies):
                if p.grad is None: continue
                
                # A. Gradient Descent
                p.data.add_(p.grad, alpha=-self.lr)
                
                # B. Stabilized Seismic Kick (only if amplitude > 0)
                if abs(amplitude) > 1e-6:
                    n = p.numel()
                    vibration_spatial = ifwht(energy * math.cos(self.t * current_freq * 0.5))
                    seismic_kick = vibration_spatial[:n].view(p.shape)
                    p.data.add_(seismic_kick, alpha=amplitude)
                
                # C. Energy Drift
                energy.add_(torch.randn_like(energy) * 0.01)
                energy.div_(energy.norm() + 1e-8).mul_(math.sqrt(energy.shape[0]))

        return loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V38 'STABILIZED SEISMIC WALSH' on MNIST: {device}")
    
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 10)
    ).to(device)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000, shuffle=False)
    
    EPOCHS = 10
    total_steps = len(train_loader) * EPOCHS
    optimizer = StabilizedSeismicOptimizer(model.parameters(), total_steps=total_steps, base_lr=0.0001, a0=0.002, freq0=0.2)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.view(-1, 784).to(device), target.to(device)
            optimizer.step(closure=lambda: criterion(model(data), target).backward() or True)
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.view(-1, 784).to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = correct / 10000
        if acc > best_acc: best_acc = acc
        
        decay = max(0.0, 1.0 - (optimizer.t / optimizer.total_steps))
        print(f"Epoch {epoch:2d}/10 | Acc: {acc:.4f} | Best: {best_acc:.4f} | Seismic Decay: {decay:.2f} | Time: {time.time()-t_start:.1f}s")

if __name__ == "__main__":
    main()
