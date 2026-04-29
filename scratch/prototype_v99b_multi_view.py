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
    indices = []
    r, c = size // 2, size // 2
    indices.append(r * size + c)
    step = 1
    while len(indices) < size * size:
        for _ in range(step):
            c += 1
            if 0 <= r < size and 0 <= c < size: indices.append(r * size + c)
        for _ in range(step):
            r += 1
            if 0 <= r < size and 0 <= c < size: indices.append(r * size + c)
        step += 1
        for _ in range(step):
            c -= 1
            if 0 <= r < size and 0 <= c < size: indices.append(r * size + c)
        for _ in range(step):
            r -= 1
            if 0 <= r < size and 0 <= c < size: indices.append(r * size + c)
        step += 1
    return torch.tensor(indices[:size*size])

def get_column_indices(size=28):
    idx = torch.arange(size * size).view(size, size)
    return idx.t().contiguous().view(-1)

# --- Core Component: Triangular Attention Neuron ---

class TriangularNeuronLayer(nn.Module):
    def __init__(self, in_features, out_features, device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        # Parameters: (out_features, 1)
        self.raw_center = nn.Parameter(torch.rand(out_features, 1))
        self.raw_width = nn.Parameter(torch.rand(out_features, 1) * 0.1 + 0.02)
        
        # Grid of indices: (1, in_features)
        self.register_buffer("indices", torch.linspace(0, 1, in_features).view(1, -1))

    def get_masks(self):
        width = F.softplus(self.raw_width) + 1e-6
        dist = torch.abs(self.indices - self.raw_center)
        masks = F.relu(1.0 - dist / width)
        mask_sums = masks.sum(dim=1, keepdim=True) + 1e-8
        return masks / mask_sums

    def forward(self, x):
        masks = self.get_masks()
        return x @ masks.t()

# --- Multi-View Model ---

class MultiViewNeedleNet(nn.Module):
    def __init__(self, hidden1=1024, hidden2=512, device='cpu'):
        super().__init__()
        # Views: Rows, Cols, Spiral
        self.register_buffer("spiral_idx", get_spiral_indices(28))
        self.register_buffer("column_idx", get_column_indices(28))
        self.register_buffer("row_idx", torch.arange(784))
        
        # Input features = 784 * 3 = 2352
        self.layer1 = TriangularNeuronLayer(2352, hidden1, device=device)
        self.bn1 = nn.BatchNorm1d(hidden1)
        
        # Layer 2 stays triangular (looking at Layer 1 outputs)
        self.layer2 = TriangularNeuronLayer(hidden1, hidden2, device=device)
        self.bn2 = nn.BatchNorm1d(hidden2)
        
        self.classifier = nn.Linear(hidden2, 10)
        
    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)
        
        # Generate 3 views
        v_rows = x[:, self.row_idx]
        v_cols = x[:, self.column_idx]
        v_spir = x[:, self.spiral_idx]
        
        # Concatenate: (B, 2352)
        x_multi = torch.cat([v_rows, v_cols, v_spir], dim=1)
        
        # L1
        x = self.layer1(x_multi)
        x = self.bn1(x)
        x = F.relu(x)
        
        # L2
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        return self.classifier(x)

# --- Training and Benchmarking ---

def run_trial(seed=42, epochs=15, lr=0.005, device='cpu'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1024)
    
    model = MultiViewNeedleNet(device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    metrics = {"wall_clock_time": 0, "eval_time": 0, "final_objective": 0, "history": []}
    
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
            epoch_loss += loss.item()
            
            if epoch == 1 and batch_idx < 5:
                print(f"  Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = correct / 10000
        print(f"Epoch {epoch:2d} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Time: {time.time()-t0:.1f}s")
        metrics["history"].append({"epoch": epoch, "acc": acc})

    metrics["wall_clock_time"] = time.time() - t_start
    metrics["final_objective"] = acc
    
    # View Distribution Analysis
    with torch.no_grad():
        centers = model.layer1.raw_center.cpu().numpy()
        rows_count = np.sum(centers < 1/3)
        cols_count = np.sum((centers >= 1/3) & (centers < 2/3))
        spir_count = np.sum(centers >= 2/3)
        print(f"\nNeuron Distribution (Layer 1):")
        print(f"  Rows View:   {rows_count} neurons")
        print(f"  Cols View:   {cols_count} neurons")
        print(f"  Spiral View: {spir_count} neurons")
    
    return metrics, model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V99b: MULTI-VIEW TRIANGULAR ATTENTION ---")
    print(f"Hardware: {device}")
    
    metrics, model = run_trial(device=device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Params: {total_params:,}")
    print(f"Final Accuracy: {metrics['final_objective']:.4f}")

    os.makedirs('results/raw', exist_ok=True)
    with open('results/raw/v99b_multi_view_results.json', 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
