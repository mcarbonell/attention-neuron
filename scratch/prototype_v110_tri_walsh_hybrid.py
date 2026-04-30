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

# --- Custom Neuron Layers (from v109) ---

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
    def __init__(self, in_features, out_features, k=16, mode='walsh'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.spectral_core = nn.Parameter(torch.randn(out_features, k) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Precompute Walsh-like basis (simplified sequency ordered)
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

# --- Hybrid Model ---

class TriWalshHybrid(nn.Module):
    def __init__(self, h_tri=32, h_walsh=32, k=16):
        super().__init__()
        self.tri_path = TriangularLayer(56, h_tri) # Procesará Islands (56D)
        self.walsh_path = SpectralLayer(784, h_walsh, k=k) # Procesará Pixels (784D)
        self.classifier = nn.Linear(h_tri + h_walsh, 10)
    
    def forward(self, batch):
        # Usamos batch como dict para mayor claridad
        x_islands = batch['islands']
        x_pixels = batch['pixels']
        
        feat_tri = torch.relu(self.tri_path(x_islands))
        feat_walsh = torch.relu(self.walsh_path(x_pixels))
        
        combined = torch.cat([feat_tri, feat_walsh], dim=1)
        return self.classifier(combined)

# --- Training Logic ---

def train_hybrid(train_loader, test_loader, epochs=10):
    device = torch.device("cpu")
    model = TriWalshHybrid(h_tri=32, h_walsh=32, k=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Hybrid Model Parameters: {params}")
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for i, batch in enumerate(train_loader):
            y = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
            if i < 5 and epoch == 0:
                print(f"Batch {i}: Loss {loss.item():.4f} | Acc {100.*correct/total:.2f}%")
        
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {100.*correct/total:.2f}%")

    # Test
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
    
    train_hybrid(train_loader, test_loader)
