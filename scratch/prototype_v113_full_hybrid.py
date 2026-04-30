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
        self.centers = nn.Parameter(torch.rand(out_features))
        self.widths = nn.Parameter(torch.rand(out_features) * 0.5 + 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer('indices', torch.linspace(0, 1, in_features))

    def get_weights(self):
        # Enforce minimum width to maintain gradient flow (at least ~2 steps)
        # Using softplus or clamp to ensure widths stay above 0.02
        safe_widths = torch.clamp(self.widths, min=0.02)
        diff = torch.abs(self.indices.unsqueeze(0) - self.centers.unsqueeze(1))
        weights = torch.clamp(1.0 - diff / (safe_widths.unsqueeze(1)), min=0.0)
        return weights

    def forward(self, x):
        w = self.get_weights()
        return torch.matmul(x, w.t()) + self.bias

class SpectralLayer(nn.Module):
    def __init__(self, in_features, out_features, k=32): # k=32 for better resolution
        super().__init__()
        self.k = k
        self.spectral_core = nn.Parameter(torch.randn(out_features, k) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
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

# --- Full Morph-Spectral Hybrid Model ---

class FullMorphSpectralHybrid(nn.Module):
    def __init__(self, h_tri=96, h_walsh=96, k=32):
        super().__init__()
        self.tri_path = TriangularLayer(113, h_tri) # Islands (56) + Intensity (57)
        self.walsh_path = SpectralLayer(784, h_walsh, k=k) # Raster Pixels
        self.classifier = nn.Linear(h_tri + h_walsh, 10)
    
    def forward(self, batch):
        # 1. Morph Path (Islands + Intensity)
        x_morph = torch.cat([batch['islands'], batch['intensity']], dim=1)
        feat_tri = torch.relu(self.tri_path(x_morph))
        
        # 2. Spectral Path (Raster Pixels)
        x_pixels = batch['pixels']
        feat_walsh = torch.relu(self.walsh_path(x_pixels))
        
        # 3. Fusion
        combined = torch.cat([feat_tri, feat_walsh], dim=1)
        return self.classifier(combined)

# --- Training Logic ---

def train_full_hybrid(train_loader, test_loader, epochs=10):
    device = torch.device("cpu")
    # Parameters Estimate:
    # Tri: 96*3 = 288
    # Walsh: 96*32 + 96 = 3168
    # Linear: 192*10 + 10 = 1930
    # Total: ~5386 parameters
    model = FullMorphSpectralHybrid(h_tri=96, h_walsh=96, k=32).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Full Morph-Spectral Hybrid Parameters: {params}")
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for batch in train_loader:
            y = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
        
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {100.*correct/total:.2f}%")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            y = batch['label'].to(device)
            outputs = model(batch)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    final_acc = 100. * correct / total
    print(f"\nFinal Test Accuracy: {final_acc:.2f}%")
    print(f"Total Parameters: {params}")
    print(f"Time: {time.time() - start_time:.2f}s")
    
    return final_acc, params

if __name__ == "__main__":
    train_cache = 'data/mnist_features_train.pt'
    test_cache = 'data/mnist_features_test.pt'
    
    if not os.path.exists(train_cache):
        print("Error: Run v107 first.")
        exit()
        
    train_data = FeatureDataset(train_cache)
    test_data = FeatureDataset(test_cache)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)
    
    train_full_hybrid(train_loader, test_loader)
