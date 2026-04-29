import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import math

# --- Fast Walsh-Hadamard Transform (Vectorized) ---
def fwht(x):
    B, N = x.shape
    h = 1
    while h < N:
        x = x.view(B, N // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        h *= 2
    return x.view(B, N)

# --- Hilbert Curve Generation (32x32) ---
def generate_hilbert_indices(n):
    def d2xy(n, d):
        t = d
        x = y = 0
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            x, y = rot(s, x, y, rx, ry)
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        return x, y
    def rot(n, x, y, rx, ry):
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            return y, x
        return x, y
    size = 2**n
    indices = []
    for d in range(size * size):
        x, y = d2xy(size, d)
        indices.append(y * size + x)
    return indices

class HilbertHolographicSupreme(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pow2_size = 1024 # 32x32
        
        # Hilbert Mapping
        self.register_buffer('hilbert_indices', torch.tensor(generate_hilbert_indices(5)))
        
        # Pesos Espaciales (para agregadores clásicos y base de resonancia)
        self.weight_spatial = nn.Parameter(torch.randn(out_features, in_features) / (in_features**0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Pesos Holográficos (Dominio Walsh)
        self.w_real = nn.Parameter(torch.randn(out_features, self.pow2_size) / (self.pow2_size**0.5))
        self.w_imag = nn.Parameter(torch.randn(out_features, self.pow2_size) / (self.pow2_size**0.5))
        
        # Gating Dinámico
        self.gate_fc = nn.Linear(in_features, out_features * 4) # SUM, VAR, LSE, HILBERT-HOLO
        
    def forward(self, x):
        B = x.size(0)
        O = self.out_features
        N = self.pow2_size
        
        # Gates
        gates = torch.sigmoid(self.gate_fc(x)).view(B, O, 4)
        
        # 1. Proyecciones Espaciales Clásicas
        Z = x.unsqueeze(1) * self.weight_spatial.unsqueeze(0)
        a_sum = Z.sum(dim=2)
        a_var = Z.var(dim=2)
        a_lse = torch.logsumexp(Z, dim=2)
        
        # 2. Resonancia Holográfica FRACTAL (HILBERT)
        # Pad y Reordenamiento
        x_img = x.view(B, 28, 28)
        x_pad = F.pad(x_img, (2, 2, 2, 2)).view(B, 1024)
        x_hilbert = x_pad[:, self.hilbert_indices]
        
        # Proyectamos al dominio Hilbert pesadamente
        # Para mantener el 97.92%+ de la V90e, necesitamos que la resonancia sea rica
        Z_h = x_hilbert.unsqueeze(1) * self.weight_spatial.mean(dim=1).view(1, O, 1) 
        W = fwht(Z_h.view(B * O, N)).view(B, O, N)
        
        res_real = (W * self.w_real.unsqueeze(0)).sum(dim=2)
        res_imag = (W * self.w_imag.unsqueeze(0)).sum(dim=2)
        a_holo_hilbert = torch.sqrt(res_real**2 + res_imag**2 + 1e-8)
        
        # Mezcla MoE Supreme
        experts = torch.stack([a_sum, a_var, a_lse, a_holo_hilbert], dim=2)
        out = (experts * gates).sum(dim=2)
        
        return out + self.bias

class SupremeNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.analog_plate = HilbertHolographicSupreme(784, 64)
        self.bn = nn.BatchNorm1d(64)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.analog_plate(x)
        x = self.bn(x)
        x = self.classifier(x)
        return x

def train_and_evaluate():
    print("=== Experimento V96: Hilbert-Resonator Supreme ===\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware: {device}")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=1000)

    model = SupremeNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        acc = 100. * correct / 10000
        print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f} | Acc: {acc:.2f}% | Time: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    train_and_evaluate()
