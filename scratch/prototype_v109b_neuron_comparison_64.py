import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import json
import os
import math

# --- Data Loading (from cached v107) ---

class FeatureDataset(Dataset):
    def __init__(self, cache_path):
        self.features = torch.load(cache_path, weights_only=False)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx]

# --- Custom Neuron Layers ---

class TriangularLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centers = nn.Parameter(torch.rand(out_features))
        self.widths = nn.Parameter(torch.rand(out_features) * 0.5 + 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('indices', torch.linspace(0, 1, in_features))

    def get_weights(self):
        diff = torch.abs(self.indices.unsqueeze(0) - self.centers.unsqueeze(1))
        weights = torch.clamp(1.0 - diff / (self.widths.unsqueeze(1) + 1e-6), min=0.0)
        return weights

    def forward(self, x):
        w = self.get_weights()
        return torch.matmul(x, w.t()) + self.bias

class SpectralLayer(nn.Module):
    def __init__(self, in_features, out_features, k=16, mode='dct'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.mode = mode
        self.spectral_core = nn.Parameter(torch.randn(out_features, k) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        if mode == 'dct':
            basis = torch.zeros(k, in_features)
            for freq in range(k):
                for i in range(in_features):
                    basis[freq, i] = math.cos(math.pi * freq * (2 * i + 1) / (2 * in_features))
            self.register_buffer('basis', basis)
        else:
            basis = torch.zeros(k, in_features)
            for freq in range(k):
                for i in range(in_features):
                    val = math.cos(math.pi * freq * (i / in_features))
                    basis[freq, i] = 1.0 if val >= 0 else -1.0
            self.register_buffer('basis', basis)

    def get_weights(self):
        return torch.matmul(self.spectral_core, self.basis)

    def forward(self, x):
        w = self.get_weights()
        return torch.matmul(x, w.t()) + self.bias

class ComparisonNet(nn.Module):
    def __init__(self, neuron_type, in_dim, out_dim=10, h=64, k=16): # CAMBIADO h=32 -> h=64
        super().__init__()
        if neuron_type == 'mlp':
            self.layer = nn.Sequential(
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Linear(h, out_dim)
            )
        elif neuron_type == 'triangular':
            self.layer = nn.Sequential(
                TriangularLayer(in_dim, h),
                nn.ReLU(),
                nn.Linear(h, out_dim)
            )
        elif neuron_type == 'dct':
            self.layer = nn.Sequential(
                SpectralLayer(in_dim, h, k=k, mode='dct'),
                nn.ReLU(),
                nn.Linear(h, out_dim)
            )
        elif neuron_type == 'walsh':
            self.layer = nn.Sequential(
                SpectralLayer(in_dim, h, k=k, mode='walsh'),
                nn.ReLU(),
                nn.Linear(h, out_dim)
            )
    
    def forward(self, x):
        return self.layer(x)

def run_experiment(neuron_name, representation_name, input_key, in_dim, train_loader, test_loader, epochs=5):
    print(f"Running (H=64): {neuron_name} + {representation_name}...")
    device = torch.device("cpu")
    model = ComparisonNet(neuron_name, in_dim, h=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    params = sum(p.numel() for p in model.parameters())
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            if isinstance(input_key, list):
                x = torch.cat([batch[k] for k in input_key], dim=1).to(device)
            else:
                x = batch[input_key].to(device)
            y = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(input_key, list):
                x = torch.cat([batch[k] for k in input_key], dim=1).to(device)
            else:
                x = batch[input_key].to(device)
            y = batch['label'].to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    acc = 100. * correct / total
    print(f"Result: {acc:.2f}% | Params: {params}")
    return {'acc': acc, 'params': params}

if __name__ == "__main__":
    train_cache = 'data/mnist_features_train.pt'
    test_cache = 'data/mnist_features_test.pt'
    
    if not os.path.exists(train_cache):
        print("Error: Run v107 first.")
        exit()
        
    train_data = FeatureDataset(train_cache)
    test_data = FeatureDataset(test_cache)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)
    
    neuron_types = ['mlp', 'triangular', 'dct', 'walsh']
    representations = [
        ('Intensity', 'intensity', 57),
        ('Islands', 'islands', 56),
        ('I + Is', ['intensity', 'islands'], 113),
        ('Pixels', 'pixels', 784)
    ]
    
    final_table = []
    for n_type in neuron_types:
        for r_name, r_key, r_dim in representations:
            res = run_experiment(n_type, r_name, r_key, r_dim, train_loader, test_loader)
            final_table.append({
                'Neuron': n_type, 'Rep': r_name, 'Acc': res['acc'], 'Params': res['params']
            })
            
    print("\n" + "="*50)
    print(f"{'Neuron':<12} | {'Rep':<12} | {'Params':<8} | {'Acc':<8}")
    print("-" * 50)
    for row in final_table:
        print(f"{row['Neuron']:<12} | {row['Rep']:<12} | {row['Params']:<8} | {row['Acc']:.2f}%")
