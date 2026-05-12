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

# --- PID Optimizer (V4: Complex-Aware) ---
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
                if wd != 0: grad = grad.add(p, alpha=wd)
                
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

# --- Complex-Valued Components ---

class ModReLU(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.bias = nn.Parameter(torch.full((features,), -0.5))

    def forward(self, z):
        abs_z = torch.abs(z)
        scale = F.relu(abs_z + self.bias) / (abs_z + 1e-6)
        return z * scale

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.complex(
            torch.randn(out_features, in_features) / math.sqrt(in_features),
            torch.randn(out_features, in_features) / math.sqrt(in_features)
        ))
        if bias:
            self.bias = nn.Parameter(torch.complex(torch.zeros(out_features), torch.zeros(out_features)))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

# --- FFT Preprocessing ---

class FFT2D(nn.Module):
    def forward(self, x):
        # x: (B, 1, 28, 28)
        # Use ortho norm to keep magnitudes stable
        x_fft = torch.fft.fft2(x, norm="ortho")
        return torch.flatten(x_fft, start_dim=1) # (B, 784) complex

# --- Architectures ---

class ComplexFFTClassifier(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.fft = FFT2D()
        self.net = nn.Sequential(
            ComplexLinear(784, hidden_dim),
            ModReLU(hidden_dim),
            ComplexLinear(hidden_dim, 10)
        )
        self.bn = nn.BatchNorm1d(10) # Normalize logits before Softmax
        
    def forward(self, x):
        x = self.fft(x)
        x = self.net(x)
        # Final classification using magnitude
        logits = torch.abs(x)
        return self.bn(logits)

class RealFFTClassifier(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.fft = FFT2D()
        self.net = nn.Sequential(
            nn.Linear(784 * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 10)
        )
        self.bn = nn.BatchNorm1d(10)
        
    def forward(self, x):
        x = self.fft(x)
        x_combined = torch.cat([x.real, x.imag], dim=-1)
        logits = self.net(x_combined)
        return self.bn(logits)

# --- Utility ---

def count_parameters(model):
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel() * (2 if torch.is_complex(p) else 1)
    return total

def run_experiment(name, model, train_loader, test_loader, epochs=10, device="cpu"):
    num_params = count_parameters(model)
    # Reduced Ki for MNIST stability
    optimizer = PID(model.parameters(), lr=1e-3, kp=1.0, ki=10.0, kd=1.0, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n>>> Running Experiment: {name} | Parameters: {num_params}")
    
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
            
            optimizer.step()
            epoch_loss += loss.item()
            
            if i % 100 == 0 and epoch == 0:
                print(f"  Batch {i} | Loss: {loss.item():.4f}")
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data.to(device))
                correct += output.argmax(dim=1).eq(target.to(device)).sum().item()
        
        acc = 100. * correct / len(test_loader.dataset)
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
        "params": num_params
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
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=128)
    
    results_list = []
    
    print("="*60)
    print("V276: COMPLEX MNIST - SPECTRAL INTELLIGENCE")
    print("="*60)
    
    # Hidden dim 64 for complex -> approx 100k params
    # 784*64*2 + 64*10*2 = 100,352 + 1,280 = 101,632
    c_model = ComplexFFTClassifier(hidden_dim=64).to(device)
    r_model = RealFFTClassifier(hidden_dim=64).to(device)
    
    res_c = run_experiment("Complex FFT MLP", c_model, train_loader, test_loader, epochs=10, device=device)
    results_list.append(res_c)
    
    res_r = run_experiment("Real FFT MLP (Matched)", r_model, train_loader, test_loader, epochs=10, device=device)
    results_list.append(res_r)
    
    # Save results
    os.makedirs("results/raw", exist_ok=True)
    timestamp = int(time.time())
    with open(f"results/raw/complex_mnist_v276_{timestamp}.json", "w") as f:
        json.dump(results_list, f, indent=4)
        
    print(f"\nResults saved to results/raw/complex_mnist_v276_{timestamp}.json")
    
    # Final Analysis
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    for r in results_list:
        print(f"{r['name']:25} | Best Acc: {r['final_objective']:.2f}% | PEI: {r['PEI']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
