import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
import json

# --- Fractal Encoding Logic ---
def get_fractal_vector(x):
    """
    Transforms (B, 1, 32, 32) into a hierarchical vector of 1365 dims.
    Levels: 1x1, 2x2, 4x4, 8x8, 16x16, 32x32.
    """
    scales = [1, 2, 4, 8, 16, 32]
    levels = [F.adaptive_avg_pool2d(x, (s, s)).view(x.size(0), -1) for s in scales]
    return torch.cat(levels, dim=1)

# --- Standard MLP Model ---
class StandardMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        return self.net(x)

def run_experiment(mode="raster", epochs=5, lr=0.001, device='cpu'):
    print(f"\n--- Running Experiment: {mode.upper()} ---", flush=True)
    
    BATCH_SIZE = 128
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000)
    
    input_dim = 1365 if mode == "fractal" else 1024
    model = StandardMLP(input_dim=input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    wall_clock_start = time.time()
    total_evals = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # Pad to 32x32
            data = F.pad(data, (2, 2, 2, 2))
            
            if mode == "fractal":
                data_in = get_fractal_vector(data)
            else:
                data_in = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            output = model(data_in)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_evals += data.size(0)
            
            if batch_idx % 200 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx:3d} | Loss: {loss.item():.4f}", flush=True)

        # Eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = F.pad(data, (2, 2, 2, 2))
                if mode == "fractal":
                    data_in = get_fractal_vector(data)
                else:
                    data_in = data.view(data.size(0), -1)
                pred = model(data_in).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = correct / 10000
        t_epoch = time.time() - t0
        print(f"  Epoch {epoch} | Acc: {acc:.4f} | Time: {t_epoch:.1f}s", flush=True)
        history.append({"epoch": epoch, "acc": acc, "loss": loss.item()})
        
    wall_clock_time = time.time() - wall_clock_start
    return {
        "mode": mode,
        "final_acc": acc,
        "total_evaluations": total_evals,
        "wall_clock_time": wall_clock_time,
        "history": history
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Info: {device}", flush=True)
    
    results = {}
    
    # 1. Run Baseline (Raster)
    results['raster'] = run_experiment(mode="raster", epochs=5, device=device)
    
    # 2. Run Fractal (Hierarchical)
    results['fractal'] = run_experiment(mode="fractal", epochs=5, device=device)
    
    # Save results
    os.makedirs('results/raw', exist_ok=True)
    with open('results/raw/v93b_fractal_vs_raster.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- Final Results Summary ---", flush=True)
    print(f"Raster Final Acc: {results['raster']['final_acc']:.4f}", flush=True)
    print(f"Fractal Final Acc: {results['fractal']['final_acc']:.4f}", flush=True)
    
    improvement = (results['fractal']['final_acc'] - results['raster']['final_acc']) * 100
    print(f"Improvement: {improvement:+.2f}%", flush=True)

if __name__ == "__main__":
    main()
