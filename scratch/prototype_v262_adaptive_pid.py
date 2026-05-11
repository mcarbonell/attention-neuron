import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os
import math

# --- Adaptive-PID Optimizer Implementation ---
class AdaptivePID(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 kp=1.0, ki=1.0, kd=1.0, derivative_beta=0.1, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, kp=kp, ki=ki, kd=kd, 
                        derivative_beta=derivative_beta, weight_decay=weight_decay)
        super(AdaptivePID, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            kp, ki, kd = group['kp'], group['ki'], group['kd']
            d_beta = group['derivative_beta']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Integral (Adam's m_t)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Variance (Adam's v_t)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # For Derivative
                    state['prev_grad'] = torch.clone(grad).detach()
                    state['exp_avg_deriv'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                prev_grad, exp_avg_deriv = state['prev_grad'], state['exp_avg_deriv']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # 1. Update Integral (I) - Adam's 1st moment
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # 2. Update Variance (Adam's 2nd moment)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 3. Update Derivative (D) - EMA of gradient change
                current_deriv = grad - prev_grad
                exp_avg_deriv.mul_(d_beta).add_(current_deriv, alpha=1 - d_beta)
                
                # 4. Adaptive PID Signal
                # Signal = Kp*grad + Ki*integral + Kd*derivative
                # We use bias-corrected integral
                i_term = exp_avg / bias_correction1
                signal = grad.mul(kp).add(i_term, alpha=ki).add(exp_avg_deriv, alpha=kd)
                
                # 5. Adaptive Normalization (Adam style)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                step_size = lr
                p.addcdiv_(signal, denom, value=-step_size)
                
                # Store for next step
                state['prev_grad'].copy_(grad)

        return loss

# --- Benchmark Setup ---
class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_classes=10):
        super(SimpleMLP, self).__init__()
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
        x = x.view(x.size(0), -1)
        return self.net(x)

def run_experiment(name, optimizer_class, optimizer_kwargs, train_loader, test_loader, epochs=10):
    device = torch.device("cpu")
    model = SimpleMLP().to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n--- Testing Optimizer: {name} ---")
    start_time = time.time()
    history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if i < 5 and epoch == 0:
                print(f"  Batch {i}: Loss {loss.item():.4f}")
        
        # Eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = 100. * correct / len(test_loader.dataset)
        history.append(acc)
        print(f"Epoch {epoch+1}/{epochs} | Acc: {acc:.2f}% | Loss: {total_loss/len(train_loader):.4f}")
        
    return acc, time.time() - start_time

def main():
    torch.manual_seed(42)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=128, shuffle=False)
    
    # Adaptive-PID (Fusing Adam's variance with PID logic)
    apid_acc, apid_time = run_experiment(
        "Adaptive-PID (Kp=1, Ki=1, Kd=1)", 
        AdaptivePID, 
        {"lr": 1e-3, "kp": 1.0, "ki": 1.0, "kd": 1.0}, 
        train_loader, test_loader
    )
    
    # Adam Baseline
    adam_acc, adam_time = run_experiment(
        "Adam (Baseline)", 
        optim.Adam, 
        {"lr": 1e-3}, 
        train_loader, test_loader
    )
    
    print(f"\nFinal Duel: Adaptive-PID {apid_acc:.2f}% vs Adam {adam_acc:.2f}%")

if __name__ == "__main__":
    main()
