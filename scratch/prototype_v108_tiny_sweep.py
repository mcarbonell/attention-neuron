import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import json
import os

# --- Dataset and Model (Same as v107) ---

class FeatureDataset(Dataset):
    def __init__(self, cache_path):
        self.features = torch.load(cache_path, weights_only=False)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx]

class GenericMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GenericMLP, self).__init__()
        # Usamos una sola capa oculta para que sea "Tiny"
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )
    def forward(self, x):
        return self.net(x)

def train_tiny(input_key, input_dim, hidden_dim, train_loader, test_loader, epochs=5):
    device = torch.device("cpu")
    model = GenericMLP(input_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Parámetros totales
    total_params = sum(p.numel() for p in model.parameters())
    
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
    
    return 100. * correct / total, total_params

if __name__ == "__main__":
    train_cache = 'data/mnist_features_train.pt'
    test_cache = 'data/mnist_features_test.pt'
    
    if not os.path.exists(train_cache):
        print("Error: Run v107 first to generate cache.")
        exit()
        
    train_data = FeatureDataset(train_cache)
    test_data = FeatureDataset(test_cache)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)
    
    hidden_sizes = [2, 4, 8, 16, 32, 64]
    
    results = {
        'pixels': [],
        'morph': [] # Intensity + Islands
    }
    
    print(f"{'Hidden':<8} | {'Type':<10} | {'Params':<8} | {'Accuracy':<10}")
    print("-" * 45)
    
    for h in hidden_sizes:
        # Pixels (784)
        acc_p, p_p = train_tiny('pixels', 784, h, train_loader, test_loader)
        results['pixels'].append({'h': h, 'params': p_p, 'acc': acc_p})
        print(f"{h:<8} | {'Pixels':<10} | {p_p:<8} | {acc_p:.2f}%")
        
        # Morph (113)
        acc_m, p_m = train_tiny(['intensity', 'islands'], 113, h, train_loader, test_loader)
        results['morph'].append({'h': h, 'params': p_m, 'acc': acc_m})
        print(f"{h:<8} | {'Morph':<10} | {p_m:<8} | {acc_m:.2f}%")
        print("-" * 45)

    # Save
    os.makedirs('results/tiny_sweep', exist_ok=True)
    with open('results/tiny_sweep/sweep_results.json', 'w') as f:
        json.dump(results, f, indent=4)
