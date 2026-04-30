import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import math
import os
import json

# --- Custom Spectral Optimizer ---

class SmoothSpectralAdam(optim.Optimizer):
    """
    SWO: Smooth Walsh Optimizer (Bilinear Proxy)
    Compresses optimizer states (m, v) using bilinear interpolation.
    This acts as a low-pass filter on the gradient history.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 k_ratio=0.25, mode='bilinear'):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= k_ratio <= 1.0:
            raise ValueError(f"Invalid k_ratio: {k_ratio}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, k_ratio=k_ratio, mode=mode)
        super(SmoothSpectralAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # For 2D weights (Out, In), we compress both dimensions
                    if p.dim() >= 2:
                        h, w = p.shape[-2], p.shape[-1]
                        kh = max(2, int(h * group['k_ratio']))
                        kw = max(2, int(w * group['k_ratio']))
                        state['k_size'] = (kh, kw)
                        # Store states in low-res (1 channel, 1 depth)
                        state['exp_avg'] = torch.zeros((1, 1, kh, kw), device=p.device)
                        state['exp_avg_sq'] = torch.zeros((1, 1, kh, kw), device=p.device)
                    else:
                        # Don't compress 1D (bias) or very small tensors
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                beta1, beta2 = group['betas']
                
                if p.dim() >= 2:
                    h, w = p.shape[-2], p.shape[-1]
                    kh, kw = state['k_size']
                    
                    # 1. Compress Gradient (Spectral Projection / Downsample)
                    g_view = grad.view(1, 1, h, w)
                    g_small = F.interpolate(g_view, size=(kh, kw), mode=group['mode'], align_corners=True)
                    
                    # 2. Update Low-Res States
                    state['exp_avg'].mul_(beta1).add_(g_small, alpha=1 - beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(g_small, g_small, value=1 - beta2)
                    
                    # 3. Bias Correction
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    
                    # 4. Reconstruct (Smooth Upsample)
                    m_rec = F.interpolate(state['exp_avg'], size=(h, w), mode=group['mode'], align_corners=True).view(p.shape)
                    v_rec = F.interpolate(state['exp_avg_sq'], size=(h, w), mode=group['mode'], align_corners=True).view(p.shape)
                    
                    # Ensure v_rec is positive (bilinear can sometimes produce eps-negatives at edges)
                    v_rec.clamp_(min=0.0) 
                    
                    # 5. Apply Adam Step
                    denom = (v_rec.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    step_size = group['lr'] / bias_correction1
                    p.addcdiv_(m_rec, denom, value=-step_size)
                    
                else:
                    # Standard Adam logic for non-compressible parameters
                    state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    step_size = group['lr'] / bias_correction1
                    p.addcdiv_(state['exp_avg'], denom, value=-step_size)

        return loss

# --- Benchmark Suite ---

def run_experiment(opt_class, name, k_ratio=0.25):
    device = torch.device("cpu") # Small model, CPU is fine
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)

    # Standard MLP for testing
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)

    if opt_class == optim.Adam:
        optimizer = opt_class(model.parameters(), lr=0.002)
    else:
        optimizer = opt_class(model.parameters(), lr=0.002, k_ratio=k_ratio)

    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining {name}...")
    start_time = time.time()
    
    # Train 2 Epochs for fast feedback
    for epoch in range(2):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx == 0 or batch_idx % 100 == 0:
                print(f"  Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")

    wall_clock = time.time() - start_time

    # Final Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            correct += output.argmax(dim=1).eq(target).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    
    # RAM Footprint calculation for states
    total_params = sum(p.numel() for p in model.parameters() if p.dim() >= 2)
    bias_params = sum(p.numel() for p in model.parameters() if p.dim() < 2)
    
    if opt_class == optim.Adam:
        # 2 states per param, 4 bytes each
        ram_mb = (total_params + bias_params) * 2 * 4 / (1024**2)
    else:
        # Compressed states + bias
        compressed_params = total_params * (k_ratio**2)
        ram_mb = (compressed_params + bias_params) * 2 * 4 / (1024**2)

    return {
        'name': name,
        'accuracy': accuracy,
        'time': wall_clock,
        'ram_mb': ram_mb
    }

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    if not os.path.exists('results/summary'):
        os.makedirs('results/summary', exist_ok=True)

    results = []
    
    # 1. Baseline Adam
    results.append(run_experiment(optim.Adam, "Adam (Full)"))
    
    # 2. SWO with 25% resolution (1/16 memory)
    results.append(run_experiment(SmoothSpectralAdam, "SWO (K=0.25)", k_ratio=0.25))
    
    # 3. SWO with 12.5% resolution (1/64 memory!)
    results.append(run_experiment(SmoothSpectralAdam, "SWO (K=0.125)", k_ratio=0.125))

    # Summary
    print("\n" + "="*70)
    print(f"{'Optimizer':<18} | {'Accuracy':<10} | {'Time (s)':<10} | {'State RAM (MB)':<15}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<18} | {r['accuracy']:<10.2f} | {r['time']:<10.2f} | {r['ram_mb']:<15.3f}")
    print("="*70)
    
    # Save to JSON
    with open('results/summary/spectral_optimizer_v125.json', 'w') as f:
        json.dump(results, f, indent=4)
