import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import json
import os
import numpy as np

# --- CONFIGURACIÓN DE DISPOSITIVO (USER Rules) ---
device = torch.device("cpu")
try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using DirectML device: {device}", flush=True)
except ImportError:
    print("torch-directml not found, using CPU (v3.13 faster for small nets)", flush=True)

# --- FAST WALSH-HADAMARD TRANSFORM (Optimized) ---
def fwht(x):
    """
    Fast Walsh-Hadamard Transform implementation.
    Input x: [Batch, N] where N is a power of 2.
    """
    b, n = x.shape
    res = x.clone()
    h = 1
    while h < n:
        res = res.view(b, n // (2 * h), 2, h)
        a = res[:, :, 0, :]
        b_ = res[:, :, 1, :]
        res = torch.cat([a + b_, a - b_], dim=2)
        h *= 2
    return res.view(b, n) / (n**0.5)

# --- FAST WALSH-HADAMARD TRANSFORM (2D) ---
def fwht_2d(x):
    """
    Fast Walsh-Hadamard Transform 2D.
    Input x: [Batch, 1, H, W] where H, W are powers of 2.
    """
    b, c, h, w = x.shape
    # FWHT on rows
    x = fwht(x.view(b * c * h, w)).view(b, c, h, w)
    # FWHT on cols (transpose, fwht, transpose)
    x = x.permute(0, 1, 3, 2).contiguous()
    x = fwht(x.view(b * c * w, h)).view(b, c, w, h)
    return x.permute(0, 1, 3, 2).contiguous()

# --- MODELS ---

class MLPBaseline(nn.Module):
    """Standard Dense MLP for comparison."""
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SpectralMatrixFree(nn.Module):
    """
    Spectral Matrix-Free architecture V2.
    Uses 2D-FWHT and selects the 16x16 low-frequency core.
    """
    def __init__(self, core_size=16):
        super().__init__()
        self.core_size = core_size # 16x16 = 256 coefficients
        self.h1 = 64
        # We project from the spectral core to a small hidden space
        self.spectral_proj = nn.Parameter(torch.randn(core_size * core_size, self.h1) * 0.01)
        self.fc_out = nn.Linear(self.h1, 10)
        self.bias_proj = nn.Parameter(torch.zeros(self.h1))
        
    def forward(self, x):
        # 1. Reshape and Pad to 32x32
        x = x.view(-1, 1, 28, 28)
        x_padded = F.pad(x, (2, 2, 2, 2)) # 28+4=32
        
        # 2. Transform to 2D Walsh Domain (Matrix Free!)
        x_spec = fwht_2d(x_padded)
        
        # 3. Select 16x16 Low-Frequency Core (Top-Left corner in sequency order?)
        # For simplicity in this implementation, we take the first core_size x core_size
        x_core = x_spec[:, 0, :self.core_size, :self.core_size].reshape(x.size(0), -1)
        
        # 4. Small non-linear projection
        z = torch.relu(torch.matmul(x_core, self.spectral_proj) + self.bias_proj)
        return self.fc_out(z)

# --- NOISE INJECTION ---

def inject_label_noise(target, noise_prob):
    """Injects symmetric label noise."""
    if noise_prob == 0:
        return target
    
    mask = torch.rand(target.shape) < noise_prob
    # Random labels from 0-9
    random_labels = torch.randint(0, 10, target.shape).to(target.device)
    
    # Apply mask
    new_target = torch.where(mask, random_labels, target)
    return new_target

# --- TRAINING LOOP ---

def train_and_evaluate(model_name, model, noise_level, epochs=5):
    print(f"\n>>> Model: {model_name} | Noise: {noise_level*100:.0f}%", flush=True)
    
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}", flush=True)
    
    t0 = time.time()
    eval_times = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Inject Noise
            target_noisy = inject_label_noise(target, noise_level)
            
            optimizer.zero_grad()
            
            t_start_eval = time.time()
            output = model(data)
            eval_times.append(time.time() - t_start_eval)
            
            loss = criterion(output, target_noisy)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Fast Feedback Rule
            if epoch == 1 and batch_idx < 5:
                print(f"    [Fast Feedback] B{batch_idx} Loss: {loss.item():.4f}", flush=True)
        
        # Evaluate on CLEAN test set
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                correct += output.argmax(dim=1).eq(target).sum().item()
        
        acc = 100. * correct / 10000
        print(f"  Epoch {epoch} | Test Acc (Clean): {acc:.2f}% | Avg Loss: {train_loss/len(train_loader):.4f}", flush=True)

    wall_clock = time.time() - t0
    eval_time = sum(eval_times)
    
    # PEI Calculation
    pei = acc / np.log10(total_params + 1)
    
    metrics = {
        "model_name": model_name,
        "noise_level": noise_level,
        "final_objective": acc,
        "total_params": total_params,
        "wall_clock_time": wall_clock,
        "function_evaluation_time": eval_time,
        "internal_overhead_time": wall_clock - eval_time,
        "PEI": pei
    }
    return metrics

# --- MAIN EXPERIMENT ---

def main():
    noise_levels = [0.0, 0.1, 0.2, 0.4, 0.6]
    all_results = []
    
    # 1. Candidate First (Spectral Matrix Free)
    print("\n" + "="*60, flush=True)
    print("PHASE 1: SPECTRAL MATRIX-FREE (CANDIDATE)", flush=True)
    print("="*60, flush=True)
    for nl in noise_levels:
        model = SpectralMatrixFree(core_size=16).to(device)
        res = train_and_evaluate("Spectral-MF", model, nl, epochs=5)
        all_results.append(res)
        
    # 2. Baseline (MLP)
    print("\n" + "="*60, flush=True)
    print("PHASE 2: MLP BASELINE", flush=True)
    print("="*60, flush=True)
    for nl in noise_levels:
        model = MLPBaseline(hidden=128).to(device)
        res = train_and_evaluate("MLP-Baseline", model, nl, epochs=5)
        all_results.append(res)
        
    # Save Results
    os.makedirs("results/raw", exist_ok=True)
    with open("results/raw/v245_noise_robustness.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    # Final Summary Table
    print("\n" + "="*80, flush=True)
    print(f"{'Model':<15} | {'Noise':<6} | {'Acc':<8} | {'PEI':<8} | {'Params':<10}", flush=True)
    print("-" * 80, flush=True)
    for r in all_results:
        print(f"{r['model_name']:<15} | {r['noise_level']*100:4.0f}% | {r['final_objective']:8.2f}% | {r['PEI']:8.2f} | {r['total_params']:<10,}", flush=True)
    print("="*80, flush=True)

if __name__ == "__main__":
    main()
