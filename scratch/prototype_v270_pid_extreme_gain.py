import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os
import math

# --- PID Optimizer (V3 with Weight Decay Fix) ---
class PID(optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, derivative=0.1, kp=1.0, ki=1.0, kd=1.0, weight_decay=1e-4):
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
            wd = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                
                # Apply Weight Decay (Standard L2)
                if wd != 0:
                    grad = grad.add(p, alpha=wd)
                
                state = self.state[p]
                if len(state) == 0:
                    state['integral'] = torch.zeros_like(p)
                    state['prev_grad'] = torch.clone(grad).detach()
                    state['derivative'] = torch.zeros_like(p)
                integral, prev_grad, deriv_ema = state['integral'], state['prev_grad'], state['derivative']
                
                # I (Integral) component: EMA of gradients
                integral.mul_(momentum).add_(grad, alpha=1 - momentum)
                
                # D (Derivative) component: Change in gradient
                current_deriv = grad - prev_grad
                deriv_ema.mul_(derivative).add_(current_deriv, alpha=1 - derivative)
                
                # PID Update: update = Kp * P + Ki * I + Kd * D
                update = grad.mul(kp).add(integral, alpha=ki).add(deriv_ema, alpha=kd)
                
                p.add_(update, alpha=-lr)
                state['prev_grad'].copy_(grad)
        return loss

# --- Standard CNN for CIFAR-10 ---
class StandardCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_experiment(name, optimizer_class, optimizer_kwargs, train_loader, test_loader, epochs=12, device="cpu"):
    model = StandardCNN().to(device)
    num_params = count_parameters(model)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n>>> Running Experiment: {name}")
    
    wall_clock_start = time.time()
    func_eval_time = 0
    total_evaluations = 0
    
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            fe_start = time.time()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            fe_end = time.time()
            func_eval_time += (fe_end - fe_start)
            total_evaluations += 1
            
            # Robust Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            if i % 200 == 0 and epoch == 0:
                print(f"  Batch {i} | Loss: {loss.item():.4f}")
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data.to(device))
                correct += output.argmax(dim=1).eq(target.to(device)).sum().item()
        
        acc = 100. * correct / 10000
        if acc > best_acc: best_acc = acc
        print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
    
    wall_clock_time = time.time() - wall_clock_start
    internal_overhead_time = wall_clock_time - func_eval_time
    pei = best_acc / math.log10(num_params + 1)
    
    results = {
        "name": name,
        "final_objective": best_acc,
        "total_evaluations": total_evaluations,
        "wall_clock_time": wall_clock_time,
        "function_evaluation_time": func_eval_time,
        "internal_overhead_time": internal_overhead_time,
        "PEI": pei,
        "config": optimizer_kwargs
    }
    
    return results

def main():
    torch.manual_seed(42)
    
    # Device detection
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"Using DirectML device: {device}")
    except:
        device = torch.device("cpu")
        print("Using CPU device")
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_loader = DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.CIFAR10('./data', train=False, transform=transform), batch_size=128)
    
    ki_vals = [500, 1000, 2000]
    kd_vals = [1, 5]
    
    results_list = []
    
    print("="*50)
    print("V270: THE SONIC BARRIER - EXTREME GAIN SWEEP")
    print("="*50)
    
    for ki in ki_vals:
        for kd in kd_vals:
            name = f"PID(Ki={ki}, Kd={kd})"
            res = run_experiment(name, PID, {"lr": 1e-3, "kp": 1.0, "ki": ki, "kd": kd, "weight_decay": 1e-4}, train_loader, test_loader, epochs=12, device=device)
            results_list.append(res)
    
    # Final Adam Baseline for reference
    adam_res = run_experiment("Adam Baseline", optim.Adam, {"lr": 1e-3, "weight_decay": 1e-4}, train_loader, test_loader, epochs=12, device=device)
    results_list.append(adam_res)
    
    # Save results
    os.makedirs("results/raw", exist_ok=True)
    timestamp = int(time.time())
    with open(f"results/raw/cifar10_extreme_v270_{timestamp}.json", "w") as f:
        json.dump(results_list, f, indent=4)
        
    print(f"\nResults saved to results/raw/cifar10_extreme_v270_{timestamp}.json")

if __name__ == "__main__":
    main()
