import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import time
import json
import os
import numpy as np

# --- Feature Extraction Functions ---

def count_islands_1d(line):
    """Counts contiguous blocks of non-zero values in a 1D tensor."""
    binary = (line > 0.1).float() # Threshold to avoid noise
    if len(binary) == 0: return 0
    # A block starts if binary[i] is 1 and (i=0 or binary[i-1] is 0)
    starts = (binary[0] == 1).float()
    if len(binary) > 1:
        starts += torch.sum((binary[1:] == 1) & (binary[:-1] == 0)).float()
    return starts

def extract_features(x):
    """
    Input: (1, 28, 28) tensor
    Returns: 
        - global_intensity (1)
        - row_intensities (28)
        - col_intensities (28)
        - row_islands (28)
        - col_islands (28)
    """
    x = x.squeeze() # (28, 28)
    
    global_int = x.sum().unsqueeze(0)
    row_int = x.sum(dim=1)
    col_int = x.sum(dim=0)
    
    row_islands = torch.tensor([count_islands_1d(x[i, :]) for i in range(28)])
    col_islands = torch.tensor([count_islands_1d(x[:, j]) for j in range(28)])
    
    return global_int, row_int, col_int, row_islands, col_islands

class FeatureMNIST(Dataset):
    def __init__(self, train=True, cache_path=None):
        self.base_dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transforms.ToTensor())
        self.features = []
        
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}...")
            self.features = torch.load(cache_path, weights_only=False)
        else:
            print(f"Extracting features for {'train' if train else 'test'} set...")
            start_time = time.time()
            for i in range(len(self.base_dataset)):
                img, label = self.base_dataset[i]
                g, ri, ci, ris, cis = extract_features(img)
                
                self.features.append({
                    'pixels': img.view(-1),
                    'intensity': torch.cat([g, ri, ci]),
                    'islands': torch.cat([ris, cis]),
                    'label': label
                })
                if i % 10000 == 0:
                    print(f"Processed {i}/{len(self.base_dataset)}")
            
            print(f"Feature extraction completed in {time.time() - start_time:.2f}s")
            if cache_path:
                print(f"Saving features to {cache_path}...")
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(self.features, cache_path)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

# --- Model ---

class GenericMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(GenericMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )
    
    def forward(self, x):
        return self.net(x)

# --- Training Loop ---

def train_config(config_name, input_key, input_dim, train_loader, test_loader, epochs=5):
    device = torch.device("cpu")
    model = GenericMLP(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    stats = {
        'config': config_name,
        'final_objective': 0,
        'wall_clock_time': 0,
        'function_evaluation_time': 0,
        'total_evaluations': 0,
        'train_acc': [],
        'test_acc': 0
    }
    
    start_wall = time.time()
    eval_time = 0
    
    print(f"\n--- Training {config_name} ---")
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        
        for i, batch in enumerate(train_loader):
            if isinstance(input_key, list):
                x = torch.cat([batch[k] for k in input_key], dim=1).to(device)
            else:
                x = batch[input_key].to(device)
                
            y = batch['label'].to(device)
            
            t0 = time.time()
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            eval_time += (time.time() - t0)
            
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            stats['total_evaluations'] += 1
            
            if i < 5 and epoch == 0:
                print(f"Batch {i}: Loss {loss.item():.4f} | Acc {100.*correct/total:.2f}%")
        
        train_acc = 100. * correct / total
        stats['train_acc'].append(train_acc)
        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.2f}%")

    stats['wall_clock_time'] = time.time() - start_wall
    stats['function_evaluation_time'] = eval_time
    stats['internal_overhead_time'] = stats['wall_clock_time'] - eval_time
    
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
    
    stats['test_acc'] = 100. * correct / total
    stats['final_objective'] = stats['test_acc']
    print(f"Test Acc: {stats['test_acc']:.2f}%")
    
    return stats

# --- Main ---

if __name__ == "__main__":
    # Cache paths
    train_cache = 'data/mnist_features_train.pt'
    test_cache = 'data/mnist_features_test.pt'
    
    # Load data
    train_data = FeatureMNIST(train=True, cache_path=train_cache)
    test_data = FeatureMNIST(train=False, cache_path=test_cache)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)
    
    results = []
    
    # 1. Baseline: Píxeles (784)
    results.append(train_config("Baseline (Pixels)", 'pixels', 784, train_loader, test_loader))
    
    # 2. Solo Intensidad (57)
    results.append(train_config("Intensity Only", 'intensity', 57, train_loader, test_loader))
    
    # 3. Solo Islas (56)
    results.append(train_config("Islands Only", 'islands', 56, train_loader, test_loader))

    # 4. Morfológico (Intensity + Islands) (113)
    results.append(train_config("Intensity + Islands", ['intensity', 'islands'], 57+56, train_loader, test_loader))
    
    # 5. Fusión Total (Pixels + Intensity + Islands)
    results.append(train_config("Full Fusion", ['pixels', 'intensity', 'islands'], 784+57+56, train_loader, test_loader))

    # Save results
    os.makedirs('results/feature_fusion', exist_ok=True)
    with open('results/feature_fusion/fusion_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nSummary:")
    for r in results:
        print(f"- {r['config']}: {r['test_acc']:.2f}% (Time: {r['wall_clock_time']:.1f}s)")
