import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import os

# --- PID Optimizer Implementation ---
class PID(optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9, derivative=0.1, kp=1.0, ki=1.0, kd=1.0, weight_decay=0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
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
            kp = group['kp']
            ki = group['ki']
            kd = group['kd']
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
                    # Integral term (Momentum)
                    state['integral'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Previous gradient for derivative calculation
                    state['prev_grad'] = torch.clone(grad).detach()
                    # Derivative term (Change in gradient)
                    state['derivative'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                integral = state['integral']
                prev_grad = state['prev_grad']
                deriv_ema = state['derivative']
                
                state['step'] += 1

                # 1. Update Integral (I) - Similar to Momentum
                integral.mul_(momentum).add_(grad, alpha=1 - momentum)
                
                # 2. Update Derivative (D) - Change in gradient
                current_deriv = grad - prev_grad
                deriv_ema.mul_(derivative).add_(current_deriv, alpha=1 - derivative)
                
                # 3. Compute PID update
                # update = Kp*grad + Ki*integral + Kd*derivative
                update = grad.mul(kp).add(integral, alpha=ki).add(deriv_ema, alpha=kd)
                
                # Apply update
                p.add_(update, alpha=-lr)
                
                # Store current grad for next step
                state['prev_grad'].copy_(grad)

        return loss

# --- Benchmark Model (Standard MLP) ---
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

# --- Training Loop ---
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
        
    duration = time.time() - start_time
    return acc, duration, history

def main():
    torch.manual_seed(42)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=128, shuffle=False)
    
    # 1. Test PID
    # We use some initial "best guess" for Kp, Ki, Kd
    pid_acc, pid_time, pid_hist = run_experiment(
        "PID (Kp=1, Ki=150, Kd=1)", 
        PID, 
        {"lr": 1e-3, "momentum": 0.9, "derivative": 0.1, "kp": 1.0, "ki": 150.0, "kd": 1.0}, 
        train_loader, test_loader
    )
    
    # 2. Test Adam (as Baseline)
    adam_acc, adam_time, adam_hist = run_experiment(
        "Adam (Standard)", 
        optim.Adam, 
        {"lr": 1e-3}, 
        train_loader, test_loader
    )
    
    results = {
        "PID": {"acc": pid_acc, "time": pid_time, "history": pid_hist},
        "Adam": {"acc": adam_acc, "time": adam_time, "history": adam_hist}
    }
    
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v261_pid_benchmark.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nFinal Results: PID {pid_acc:.2f}% vs Adam {adam_acc:.2f}%")

if __name__ == "__main__":
    main()
