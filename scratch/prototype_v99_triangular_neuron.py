import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import math
import os
import json
import numpy as np

# --- 1D Reordering Utilities ---

def get_spiral_indices(size=28):
    """
    Center-out spiral for 2D square grid.
    """
    indices = []
    r, c = size // 2, size // 2
    indices.append(r * size + c)
    
    step = 1
    while len(indices) < size * size:
        # Move Right
        for _ in range(step):
            c += 1
            if 0 <= r < size and 0 <= c < size:
                indices.append(r * size + c)
        # Move Down
        for _ in range(step):
            r += 1
            if 0 <= r < size and 0 <= c < size:
                indices.append(r * size + c)
        step += 1
        # Move Left
        for _ in range(step):
            c -= 1
            if 0 <= r < size and 0 <= c < size:
                indices.append(r * size + c)
        # Move Up
        for _ in range(step):
            r -= 1
            if 0 <= r < size and 0 <= c < size:
                indices.append(r * size + c)
        step += 1
        
    return torch.tensor(indices[:size*size])

# --- Core Component: Triangular Attention Neuron ---

class TriangularNeuronLayer(nn.Module):
    """
    V99: Each neuron has exactly 2 parameters: center and width.
    Mask(i) = relu(1 - |i - center| / width)
    """
    def __init__(self, in_features, out_features, device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        # Parameters: (out_features, 1)
        # We use a normalized space [0, 1] internally for stability, then scale to in_features
        self.raw_center = nn.Parameter(torch.rand(out_features, 1))
        self.raw_width = nn.Parameter(torch.rand(out_features, 1) * 0.2 + 0.05) # Start with 5-25% coverage
        
        # Grid of indices: (1, in_features)
        self.register_buffer("indices", torch.linspace(0, 1, in_features).view(1, -1))

    def get_masks(self):
        # center: (N, 1), width: (N, 1), indices: (1, L)
        # Enforce width > 0 using softplus or abs
        width = F.softplus(self.raw_width) + 1e-6
        # Triangular profile: relu(1 - |i - c| / w)
        dist = torch.abs(self.indices - self.raw_center)
        masks = F.relu(1.0 - dist / width)
        
        # Optional: Normalize masks so they act like an average rather than a weighted sum
        # This prevents activations from exploding with large widths
        mask_sums = masks.sum(dim=1, keepdim=True) + 1e-8
        return masks / mask_sums

    def forward(self, x):
        # x: (Batch, in_features)
        masks = self.get_masks() # (out_features, in_features)
        # Matrix multiply: (B, L) @ (L, N) -> (B, N)
        return x @ masks.t()

# --- The Model ---

class NeedleNet(nn.Module):
    def __init__(self, input_dim=784, hidden1=1024, hidden2=512, order='raster', device='cpu'):
        super().__init__()
        self.order_type = order
        
        if order == 'spiral':
            self.register_buffer("order_idx", get_spiral_indices(28))
        else:
            self.register_buffer("order_idx", torch.arange(784))
            
        self.layer1 = TriangularNeuronLayer(input_dim, hidden1, device=device)
        self.bn1 = nn.BatchNorm1d(hidden1)
        
        self.layer2 = TriangularNeuronLayer(hidden1, hidden2, device=device)
        self.bn2 = nn.BatchNorm1d(hidden2)
        
        self.classifier = nn.Linear(hidden2, 10)
        
    def forward(self, x):
        # Flatten and reorder
        x = x.view(x.size(0), -1)
        x = x[:, self.order_idx]
        
        # L1
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # L2
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Output
        return self.classifier(x)

# --- Training and Benchmarking ---

def run_trial(seed=42, order='spiral', epochs=10, lr=0.005, device='cpu'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1024)
    
    model = NeedleNet(order=order, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Metrics tracking
    metrics = {
        "wall_clock_time": 0,
        "eval_time": 0,
        "final_objective": 0,
        "total_evaluations": 0,
        "history": []
    }
    
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            t_eval_0 = time.time()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            metrics["eval_time"] += (time.time() - t_eval_0)
            metrics["total_evaluations"] += data.size(0)
            
            epoch_loss += loss.item()
            
            # Fast Feedback Rule: print first 5 batches
            if epoch == 1 and batch_idx < 5:
                print(f"  Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = correct / 10000
        print(f"Epoch {epoch:2d} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Time: {time.time()-t0:.1f}s")
        metrics["history"].append({"epoch": epoch, "acc": acc, "loss": epoch_loss/len(train_loader)})

    metrics["wall_clock_time"] = time.time() - t_start
    metrics["final_objective"] = acc
    metrics["internal_overhead_time"] = metrics["wall_clock_time"] - metrics["eval_time"]
    
    return metrics, model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V99: TRIANGULAR ATTENTION NEURON ---")
    print(f"Hardware: {device}")
    
    configs = [
        {"order": "raster", "name": "Raster Order"},
        {"order": "spiral", "name": "Spiral Order"}
    ]
    
    all_results = {}
    for config in configs:
        print(f"\nTesting: {config['name']}")
        metrics, model = run_trial(order=config['order'], device=device)
        all_results[config['order']] = metrics
        
        # Param count
        tri_params = sum(p.numel() for p in model.layer1.parameters()) + sum(p.numel() for p in model.layer2.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Triangular Params: {tri_params}")
        print(f"Total Params: {total_params:,}")
        print(f"Final Accuracy: {metrics['final_objective']:.4f}")

    # Save results
    os.makedirs('results/raw', exist_ok=True)
    with open('results/raw/v99_triangular_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print("\n--- Summary ---")
    for k, v in all_results.items():
        print(f"{k}: {v['final_objective']:.4f} in {v['wall_clock_time']:.1f}s")

if __name__ == "__main__":
    main()
