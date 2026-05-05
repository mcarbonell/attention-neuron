import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import os
import json
import numpy as np

# --- CONFIGURACIÓN DE DISPOSITIVO ---
device = torch.device('cpu')
try:
    import torch_directml
    device = torch_directml.device()
except ImportError:
    pass

# --- GENERADOR DE MATRIZ DCT ---
def get_dct_matrix(n, device_target):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0: matrix[i, j] = np.sqrt(1/n)
            else: matrix[i, j] = np.sqrt(2/n) * np.cos(np.pi * i * (2*j + 1) / (2*n))
    return torch.tensor(matrix, dtype=torch.float32).to(device_target)

class SpectralSignatureLayer(nn.Module):
    def __init__(self, in_features, out_features, k=8, mode='sum'):
        super().__init__()
        self.k = k
        self.mode = mode
        self.in_features = in_features
        self.out_features = out_features
        self.spectral_params = nn.Parameter(torch.randn(out_features, k) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features if mode == 'sum' else out_features * k))
        n_dct = 1 << (in_features - 1).bit_length() 
        self.register_buffer('dct_mat', get_dct_matrix(n_dct, 'cpu'))

    def forward(self, x):
        x_freq = torch.matmul(x, self.dct_mat[:self.k, :self.in_features].t())
        out_spectral = x_freq.unsqueeze(1) * self.spectral_params.unsqueeze(0)
        if self.mode == 'sum':
            return torch.sum(out_spectral, dim=2) + self.bias
        else:
            return out_spectral.reshape(x.size(0), -1) + self.bias

class SignatureNoReluModel(nn.Module):
    def __init__(self, h1=64, k=8):
        super().__init__()
        self.l1 = SpectralSignatureLayer(784, h1, k=k, mode='signature')
        self.bn1 = nn.BatchNorm1d(h1 * k)
        self.classifier = nn.Linear(h1 * k, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        # Sin ReLU en las firmas
        x = self.bn1(self.l1(x))
        return self.classifier(x)

def train_and_eval(model_type, epochs=5):
    print(f"\n>>> Entrenando Modelo: {model_type}")
    model = SignatureNoReluModel().to(device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                correct += output.argmax(dim=1).eq(target).sum().item()
        
        acc = 100. * correct / 10000
        print(f"  Epoch {epoch} | Test Acc: {acc:.2f}%")
    return acc

if __name__ == "__main__":
    train_and_eval("Signature_No_ReLU")
