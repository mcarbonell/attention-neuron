import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os

# --- PID Optimizer ---
class PID(optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, derivative=0.1, kp=1.0, ki=1.0, kd=1.0, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, derivative=derivative, 
                        kp=kp, ki=ki, kd=kd, weight_decay=weight_decay)
        super(PID, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
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

# --- Standard MLP ---
class StandardMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, num_classes)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

def run_standard_exp(name, optimizer_class, optimizer_kwargs, train_loader, test_loader, epochs=10):
    device = torch.device("cpu")
    model = StandardMLP().to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n--- Testing Optimizer: {name} ---")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # THE KEY: Relaxed Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            
            optimizer.step()
            if i % 200 == 0 and epoch == 0:
                print(f"  Batch {i} | Loss: {loss.item():.4f}")
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data.to(device))
                correct += output.argmax(dim=1).eq(target.to(device)).sum().item()
        
        acc = 100. * correct / 10000
        print(f"Epoch {epoch+1}/{epochs} | Acc: {acc:.2f}% | Time: {time.time()-start_time:.2f}s")
    return acc

def main():
    torch.manual_seed(42)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=128, shuffle=False)
    
    # PID with Relaxed Clipping
    pid_acc = run_standard_exp("PID (Kp=1, Ki=100, Kd=0.1) + Clip 100", PID, {"lr": 1e-3, "kp": 1.0, "ki": 100.0, "kd": 0.1}, train_loader, test_loader)
    pid_acc = run_standard_exp("PID (Kp=1, Ki=100, Kd=1) + Clip 100", PID, {"lr": 1e-3, "kp": 1.0, "ki": 100.0, "kd": 1}, train_loader, test_loader)
    pid_acc = run_standard_exp("PID (Kp=1, Ki=100, Kd=10) + Clip 100", PID, {"lr": 1e-3, "kp": 1.0, "ki": 100.0, "kd": 10}, train_loader, test_loader)
    
    # Adam Baseline
    adam_acc = run_standard_exp( "Adam (Standard)", optim.Adam, {"lr": 1e-3}, train_loader, test_loader)
    
    # print(f"\nFINAL VERDICT (Standard MLP): PID {pid_acc:.2f}% vs Adam {adam_acc:.2f}%")

if __name__ == "__main__":
    main()
