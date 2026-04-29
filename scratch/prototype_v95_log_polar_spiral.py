import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
import json
import math

# --- Log-Polar Spiral Sampler ---
class SpiralSampler:
    def __init__(self, num_samples=1024, size=32, device='cpu'):
        self.num_samples = num_samples
        self.size = size
        self.device = device
        
        # Pre-calculate sampling grid
        # We want a spiral: r = scale * exp(a * t), theta = b * t
        t = torch.linspace(0, 1, num_samples, device=device)
        
        # Log-radial distribution: more samples near the center? 
        # Or uniform along the spiral path? 
        # User said: "desde el centro hacia afuera... más grande y velocidad angular mayor"
        # Let's use a power law for radius to control density
        r = t ** 1.5  # Concentrates more samples near the center (t=0)
        theta = 20 * math.pi * t # 10 full rotations
        
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        
        # Grid sample expects coords in [-1, 1]
        self.grid = torch.stack([x, y], dim=1).view(1, 1, num_samples, 2)
        
        # For dynamic blurring (kernel grows with r), we'd need a more complex implementation.
        # For this prototype, we'll use a fixed sampling first, 
        # but we can simulate the "growing circle" by sampling multiple points 
        # around each spiral point or using a multi-scale pyramid.
        
    def sample(self, x):
        """
        x: (B, 1, 32, 32)
        Returns: (B, num_samples)
        """
        # grid_sample uses (x, y) coordinates
        B = x.size(0)
        # Expand grid to match batch size
        grid_expanded = self.grid.expand(B, -1, -1, -1)
        samples = F.grid_sample(x, grid_expanded, 
                                mode='bilinear', padding_mode='zeros', align_corners=True)
        return samples.view(B, -1)

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
    
    sampler = SpiralSampler(num_samples=1024, device=device)
    model = StandardMLP(input_dim=1024).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    wall_clock_start = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = F.pad(data, (2, 2, 2, 2))
            
            if mode == "spiral":
                data_in = sampler.sample(data)
            else:
                data_in = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            output = model(data_in)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 200 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx:3d} | Loss: {loss.item():.4f}", flush=True)

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = F.pad(data, (2, 2, 2, 2))
                if mode == "spiral":
                    data_in = sampler.sample(data)
                else:
                    data_in = data.view(data.size(0), -1)
                pred = model(data_in).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = correct / 10000
        print(f"  Epoch {epoch} | Acc: {acc:.4f} | Time: {time.time()-t0:.1f}s", flush=True)
        history.append({"epoch": epoch, "acc": acc, "loss": loss.item()})
        
    return {"mode": mode, "final_acc": acc, "history": history}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Info: {device}", flush=True)
    
    results = {}
    results['raster'] = run_experiment(mode="raster", epochs=5, device=device)
    results['spiral'] = run_experiment(mode="spiral", epochs=5, device=device)
    
    os.makedirs('results/raw', exist_ok=True)
    with open('results/raw/v95_log_polar_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- Final Results Summary ---", flush=True)
    print(f"Raster Final Acc: {results['raster']['final_acc']:.4f}", flush=True)
    print(f"Spiral Final Acc: {results['spiral']['final_acc']:.4f}", flush=True)

if __name__ == "__main__":
    main()
