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
            lr = group['lr']
            momentum = group['momentum']
            derivative = group['derivative']
            kp, ki, kd = group['kp'], group['ki'], group['kd']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                if wd != 0: grad = grad.add(p, alpha=wd)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['integral'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.clone(grad).detach()
                    state['derivative'] = torch.zeros_like(p)
                integral, prev_grad, deriv_ema = state['integral'], state['prev_grad'], state['derivative']
                state['step'] += 1
                integral.mul_(momentum).add_(grad, alpha=1 - momentum)
                current_deriv = grad - prev_grad
                deriv_ema.mul_(derivative).add_(current_deriv, alpha=1 - derivative)
                update = grad.mul(kp).add(integral, alpha=ki).add(deriv_ema, alpha=kd)
                p.add_(update, alpha=-lr)
                state['prev_grad'].copy_(grad)
        return loss

# --- Architecture Components ---
def get_hadamard_matrix(n):
    if n == 1: return torch.tensor([[1.0]])
    h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    h_prev = get_hadamard_matrix(n // 2)
    return torch.kron(h2, h_prev) / math.sqrt(2)

def get_sinusoidal_embeddings(n_seq, d_model):
    pe = torch.zeros(n_seq, d_model)
    position = torch.arange(0, n_seq, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class TernaryLinearGated(nn.Module):
    def __init__(self, in_features, out_features, init_val=1.0):
        super().__init__()
        weights = torch.randint(-1, 2, (out_features, in_features)).float()
        self.register_buffer("weight", weights)
        self.gate = nn.Parameter(torch.full((out_features,), float(init_val)))
    def forward(self, x):
        res = torch.matmul(x, self.weight.t())
        return res * self.gate

class ResidualSpectrumBlock(nn.Module):
    def __init__(self, hidden_dim, seq_len, hadamard_matrix):
        super().__init__()
        self.register_buffer("hadamard", hadamard_matrix)
        self.proj = TernaryLinearGated(hidden_dim, hidden_dim, init_val=0.0)
        self.silu = nn.SiLU()
    def forward(self, x):
        residual = x
        x = self.hadamard @ x
        x = self.proj(x)
        x = self.silu(x)
        return residual + x

class HighResPSGT_PID(nn.Module):
    def __init__(self, patch_size=2, hidden_dim=256, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        n_patches = (32 // patch_size) ** 2
        self.register_buffer("hadamard", get_hadamard_matrix(n_patches))
        self.register_buffer("pos_encoding", get_sinusoidal_embeddings(n_patches, hidden_dim))
        self.patch_embed = TernaryLinearGated(patch_size * patch_size, hidden_dim, init_val=1.0)
        self.silu_embed = nn.SiLU()
        self.block1 = ResidualSpectrumBlock(hidden_dim, n_patches, self.hadamard)
        self.block2 = ResidualSpectrumBlock(hidden_dim, n_patches, self.hadamard)
        self.block3 = ResidualSpectrumBlock(hidden_dim, n_patches, self.hadamard)
        self.block4 = ResidualSpectrumBlock(hidden_dim, n_patches, self.hadamard)
        self.classifier = TernaryLinearGated(hidden_dim, num_classes, init_val=1.0)
    def forward(self, x):
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(b, -1, p*p)
        x = self.patch_embed(x)
        x = x + self.pos_encoding
        x = self.silu_embed(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# --- Training Loop ---
def train():
    device = torch.device("cpu")
    torch.manual_seed(42)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=128, shuffle=False)
    
    model = HighResPSGT_PID().to(device)
    # Using the "Beast" configuration (Ki=150)
    optimizer = PID(model.parameters(), lr=1e-3, kp=1.0, ki=5.0, kd=1.0)
    criterion = nn.CrossEntropyLoss()
    
    print("\n--- Training High-Res PSGT with PID-5 ---")
    for epoch in range(15):
        model.train()
        for i, (data, target) in enumerate(train_loader):
            data = torch.nn.functional.pad(data, (2, 2, 2, 2))
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if i < 25 and epoch == 0:
                print(f"  Batch {i}: Loss {loss.item():.4f}")
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = torch.nn.functional.pad(data, (2, 2, 2, 2))
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        acc = 100. * correct / len(test_loader.dataset)
        print(f"Epoch {epoch+1}/15 | Acc: {acc:.2f}%")

if __name__ == "__main__":
    train()
