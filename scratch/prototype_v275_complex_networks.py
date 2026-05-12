import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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
                
                # Apply Weight Decay (Standard L2)
                if wd != 0:
                    grad = grad.add(p, alpha=wd)
                
                state = self.state[p]
                if len(state) == 0:
                    # Complex-aware initialization
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
                # Note: PyTorch handles complex add/mul correctly
                update = grad.mul(kp).add(integral, alpha=ki).add(deriv_ema, alpha=kd)
                
                p.add_(update, alpha=-lr)
                state['prev_grad'].copy_(grad)
        return loss

# --- Complex-Valued Components ---

class ModReLU(nn.Module):
    """
    ModReLU activation function: ModReLU(z, b) = ReLU(|z| + b) * (z / |z|)
    Preserves phase, modifies magnitude.
    """
    def __init__(self, features):
        super().__init__()
        self.bias = nn.Parameter(torch.full((features,), -0.5)) # Learnable threshold

    def forward(self, z):
        abs_z = torch.abs(z)
        scale = F.relu(abs_z + self.bias) / (abs_z + 1e-6)
        return z * scale

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weights initialized as complex
        self.weight = nn.Parameter(torch.complex(
            torch.randn(out_features, in_features) / math.sqrt(in_features),
            torch.randn(out_features, in_features) / math.sqrt(in_features)
        ))
        if bias:
            self.bias = nn.Parameter(torch.complex(
                torch.zeros(out_features),
                torch.zeros(out_features)
            ))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

# --- Architectures ---

class ComplexMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            ComplexLinear(input_dim, hidden_dim),
            ModReLU(hidden_dim),
            ComplexLinear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # Ensure input is complex
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        return self.net(x)

class RealMLP(nn.Module):
    """
    Baseline MLP with 2x neurons to match the parameter count of a Complex MLP.
    A Complex weight (a+bi) is 2 real parameters. 
    ComplexLinear(N, M) has 2NM weights.
    RealLinear(N, 2M) followed by RealLinear(2M, O) has roughly the same parameters if we count carefully.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # We use 2x hidden_dim to match degrees of freedom
        # Input dim for real is also 2x because we pass real/imag parts separately
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim * 2)
        )
        
    def forward(self, x):
        # x is complex, split into real and imag parts
        x_real = x.real
        x_imag = x.imag
        x_combined = torch.cat([x_real, x_imag], dim=-1)
        out = self.net(x_combined)
        # Convert back to complex for loss calculation consistency
        return torch.complex(out[..., :out.shape[-1]//2], out[..., out.shape[-1]//2:])

# --- Dataset: Wave Interference Challenge ---

def generate_wave_dataset(num_samples=10000, num_signals=4):
    """
    Predict the sum of complex signals.
    Input: N complex numbers (amplitudes and phases)
    Output: Sum of these complex numbers.
    """
    # Random amplitudes and phases
    amplitudes = torch.rand(num_samples, num_signals)
    phases = torch.rand(num_samples, num_signals) * 2 * math.pi
    
    # Inputs as complex numbers
    inputs = torch.complex(amplitudes * torch.cos(phases), amplitudes * torch.sin(phases))
    
    # Ground truth: sum of signals
    targets = torch.sum(inputs, dim=1, keepdim=True)
    
    return inputs, targets

# --- Utility ---

def count_parameters(model):
    # For complex parameters, numel() counts them as 1 complex number.
    # But each complex number has 2 real values.
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            if torch.is_complex(p):
                total += p.numel() * 2
            else:
                total += p.numel()
    return total

def complex_mse_loss(output, target):
    return torch.mean(torch.abs(output - target)**2)

def run_experiment(name, model, train_loader, test_loader, epochs=20, device="cpu"):
    num_params = count_parameters(model)
    optimizer = PID(model.parameters(), lr=1e-2, kp=1.0, ki=100.0, kd=1.0, weight_decay=1e-5)
    
    print(f"\n>>> Running Experiment: {name} | Parameters: {num_params}")
    
    wall_clock_start = time.time()
    func_eval_time = 0
    total_evaluations = 0
    
    history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            fe_start = time.time()
            optimizer.zero_grad()
            output = model(data)
            
            # Use custom complex-aware MSE loss
            loss = complex_mse_loss(output, target)
            loss.backward()
            fe_end = time.time()
            
            func_eval_time += (fe_end - fe_start)
            total_evaluations += 1
            
            optimizer.step()
            epoch_loss += loss.item()
            
            if i % 10 == 0 and epoch == 0:
                print(f"  Batch {i} | Loss: {loss.item():.6f}")
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data.to(device))
                val_loss += complex_mse_loss(output, target.to(device)).item()
        
        val_loss /= len(test_loader)
        history.append(val_loss)
        print(f"  Epoch {epoch+1:2d}/{epochs} | Train Loss: {epoch_loss/len(train_loader):.6f} | Val Loss: {val_loss:.6f}")
    
    wall_clock_time = time.time() - wall_clock_start
    internal_overhead_time = wall_clock_time - func_eval_time
    
    # PEI for regression (using -log10(loss) as a proxy for "intelligence")
    final_score = -math.log10(val_loss + 1e-12)
    pei = final_score / math.log10(num_params + 1)
    
    results = {
        "name": name,
        "final_objective": val_loss,
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
    
    # Data generation
    num_signals = 8
    x, y = generate_wave_dataset(num_samples=20000, num_signals=num_signals)
    
    # Split
    train_size = 16000
    train_x, val_x = x[:train_size], x[train_size:]
    train_y, val_y = y[:train_size], y[train_size:]
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=128)
    
    # Propose hidden dims to match parameters
    # Complex MLP: 8 -> 32 -> 1
    # Real MLP: 16 -> 64 -> 2 (approx)
    
    c_model = ComplexMLP(num_signals, 32, 1).to(device)
    r_model = RealMLP(num_signals, 32, 1).to(device)
    
    results_list = []
    
    print("="*60)
    print("V275: THE COMPLEX CHALLENGE - WAVE INTERFERENCE")
    print("="*60)
    
    res_c = run_experiment("Complex-Valued MLP", c_model, train_loader, val_loader, epochs=30, device=device)
    results_list.append(res_c)
    
    res_r = run_experiment("Real-Valued MLP (Matched)", r_model, train_loader, val_loader, epochs=30, device=device)
    results_list.append(res_r)
    
    # Save results
    os.makedirs("results/raw", exist_ok=True)
    timestamp = int(time.time())
    with open(f"results/raw/complex_v275_{timestamp}.json", "w") as f:
        json.dump(results_list, f, indent=4)
        
    print(f"\nResults saved to results/raw/complex_v275_{timestamp}.json")
    
    # Final Analysis
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    for r in results_list:
        print(f"{r['name']:25} | Val Loss: {r['final_objective']:.6e} | PEI: {r['PEI']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
