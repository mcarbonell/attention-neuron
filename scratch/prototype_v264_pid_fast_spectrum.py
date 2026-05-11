import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os
import math

# --- PID Optimizer Implementation ---
class PID(optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, derivative=0.1, kp=1.0, ki=1.0, kd=1.0, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, derivative=derivative, 
                        kp=kp, ki=ki, kd=kd, weight_decay=weight_decay)
        super(PID, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, momentum, derivative = group['lr'], group['momentum'], group['derivative']
            kp, ki, kd = group['kp'], group['ki'], group['kd']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['integral'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.clone(grad).detach()
                    state['derivative'] = torch.zeros_like(p)
                integral, prev_grad, deriv_ema = state['integral'], state['prev_grad'], state['derivative']
                integral.mul_(momentum).add_(grad, alpha=1 - momentum)
                current_deriv = grad - prev_grad
                deriv_ema.mul_(derivative).add_(current_deriv, alpha=1 - derivative)
                update = grad.mul(kp).add(integral, alpha=ki).add(deriv_ema, alpha=kd)
                p.add_(update, alpha=-lr)
                state['prev_grad'].copy_(grad)
        return loss

# --- Fast Architecture Components ---
def get_hadamard_matrix(n):
    if n == 1: return torch.tensor([[1.0]])
    h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    h_prev = get_hadamard_matrix(n // 2)
    return torch.kron(h2, h_prev) / math.sqrt(2)

class TernaryLinearGated(nn.Module):
    def __init__(self, in_features, out_features, init_val=1.0):
        super().__init__()
        weights = torch.randint(-1, 2, (out_features, in_features)).float()
        self.register_buffer("weight", weights)
        self.gate = nn.Parameter(torch.full((out_features,), float(init_val)))
    def forward(self, x):
        return torch.matmul(x, self.weight.t()) * self.gate

class FastSpectrumBlock(nn.Module):
    def __init__(self, hidden_dim, hadamard_matrix):
        super().__init__()
        self.register_buffer("hadamard", hadamard_matrix)
        self.proj = TernaryLinearGated(hidden_dim, hidden_dim, init_val=0.0)
        self.silu = nn.SiLU()
    def forward(self, x):
        return x + self.silu(self.proj(self.hadamard @ x))

class FastPSGT(nn.Module):
    def __init__(self, patch_size=4, hidden_dim=128):
        super().__init__()
        self.patch_size = patch_size
        n_patches = (28 // patch_size) ** 2 # 49 patches (7x7)
        # Note: 49 is not power of 2. We use 64 by padding to 32x32 image or just use nearest power of 2.
        # Let's use 32x32 padding and 4x4 patches -> 64 patches.
        n_seq = 64
        self.register_buffer("hadamard", get_hadamard_matrix(n_seq))
        self.patch_embed = TernaryLinearGated(patch_size * patch_size, hidden_dim, init_val=1.0)
        self.block1 = FastSpectrumBlock(hidden_dim, self.hadamard)
        self.classifier = TernaryLinearGated(hidden_dim, 10, init_val=1.0)
        
    def forward(self, x):
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(b, -1, p*p)
        x = self.patch_embed(x)
        x = self.block1(x)
        return self.classifier(x.mean(dim=1))

def run_fast_exp(name, optimizer_class, optimizer_kwargs, train_loader, test_loader, device, epochs=5):
    model = FastPSGT().to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n--- Testing Optimizer: {name} ---")
    learnable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.buffers())
    print(f"Learnable Gates: {learnable:,} | Frozen Params: {frozen:,}")
    
    for epoch in range(epochs):
        model.train()
        start = time.time()
        for i, (data, target) in enumerate(train_loader):
            data = torch.nn.functional.pad(data, (2, 2, 2, 2))
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient Clipping for Stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            if i % 100 == 0:
                print(f"  [{time.strftime('%H:%M:%S')}] Batch {i} | Loss: {loss.item():.4f}")
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = torch.nn.functional.pad(data, (2, 2, 2, 2))
                output = model(data.to(device))
                correct += output.argmax(dim=1).eq(target.to(device)).sum().item()
        
        duration = time.time() - start
        print(f"Epoch {epoch+1}/{epochs} | Acc: {100.*correct/10000:.2f}% | Time: {duration:.2f}s")
    return 100.*correct/10000

def train_fast():
    device = torch.device("cpu")
    torch.manual_seed(42)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=256, shuffle=False)
    
    # 1. PID
    run_fast_exp("PID (Kp=1, Ki=1, Kd=1)", PID, {"lr": 1e-3, "ki": 1.0}, train_loader, test_loader, device)
    run_fast_exp("PID (Kp=1, Ki=5, Kd=1)", PID, {"lr": 1e-3, "ki": 5.0}, train_loader, test_loader, device)
    run_fast_exp("PID (Kp=1, Ki=15, Kd=1)", PID, {"lr": 1e-3, "ki": 15.0}, train_loader, test_loader, device)
    run_fast_exp("PID (Kp=1, Ki=50, Kd=1)", PID, {"lr": 1e-3, "ki": 50.0}, train_loader, test_loader, device)
    run_fast_exp("PID (Kp=1, Ki=100, Kd=1)", PID, {"lr": 1e-3, "ki": 100.0}, train_loader, test_loader, device)
    run_fast_exp("PID (Kp=1, Ki=150, Kd=1)", PID, {"lr": 1e-3, "ki": 150.0}, train_loader, test_loader, device)
    
    # 2. Adam
    run_fast_exp("Adam (Standard)", optim.Adam, {"lr": 1e-3}, train_loader, test_loader, device)

if __name__ == "__main__":
    train_fast()
