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

# --- Hilbert Curve Generation ---
def generate_hilbert_indices(n):
    """
    Generates indices for a Hilbert curve of size 2^n x 2^n.
    """
    def d2xy(n, d):
        """Converts distance along the curve to (x, y) coordinates."""
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
        """Rotates and flips the quadrant appropriately."""
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
        indices.append(y * size + x) # Row-major to index
    return indices

class HilbertHolographicLayer(nn.Module):
    def __init__(self, in_features, out_features, use_dct=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_dct = use_dct
        self.pow2_size = 1024 # MNIST padded to 32x32
        
        # Hilbert Indices (32x32)
        self.register_buffer('hilbert_indices', torch.tensor(generate_hilbert_indices(5)))
        
        # Proyección espacial
        self.weight_spatial = nn.Parameter(torch.randn(out_features, in_features) / (in_features**0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Pesos complejos
        self.w_real = nn.Parameter(torch.randn(out_features, self.pow2_size) / (self.pow2_size**0.5))
        self.w_imag = nn.Parameter(torch.randn(out_features, self.pow2_size) / (self.pow2_size**0.5))
        
        # Gate MoE
        self.gate_fc = nn.Linear(in_features, out_features * 3) # Sum, LSE, Spectral
        
    def forward(self, x):
        B = x.size(0)
        O = self.out_features
        N = self.pow2_size
        
        # 1. Reordenamiento Hilbert (Foveación Fractal)
        # Pad MNIST 28x28 a 32x32
        x_img = x.view(B, 28, 28)
        x_pad = F.pad(x_img, (2, 2, 2, 2)).view(B, 1024)
        x_hilbert = x_pad[:, self.hilbert_indices]
        
        # Gates
        gates = torch.sigmoid(self.gate_fc(x)).view(B, O, 3)
        
        # Expertos Espaciales (sobre entrada original)
        Z = x.unsqueeze(1) * self.weight_spatial.unsqueeze(0)
        a_sum = Z.sum(dim=2)
        a_lse = torch.logsumexp(Z, dim=2)
        
        # EXPERTO ESPECTRAL HILBERT
        # Aplicamos la transformada sobre el orden Hilbert
        # Z_hilbert: Activaciones pesadas en orden Hilbert (B, O, N)
        # Para simplificar y mantener la esencia holográfica:
        Z_h = x_hilbert.unsqueeze(1) * self.weight_spatial.mean(dim=1).view(1, O, 1) # Proyección simple para el espectro
        
        if self.use_dct:
            # DCT-II Aproximada (Usando FFT real)
            W = torch.fft.rfft(Z_h, dim=2).abs() # Magnitud FFT como proxy de DCT para este prototipo
            # Pad W to N if needed or just use rfft result
            W = F.pad(W, (0, N - W.size(2)))
        else:
            W = fwht(Z_h.view(B * O, N)).view(B, O, N)
            
        res_real = (W * self.w_real.unsqueeze(0)).sum(dim=2)
        res_imag = (W * self.w_imag.unsqueeze(0)).sum(dim=2)
        a_spectral = torch.sqrt(res_real**2 + res_imag**2 + 1e-8)
        
        experts = torch.stack([a_sum, a_lse, a_spectral], dim=2)
        out = (experts * gates).sum(dim=2)
        
        return out + self.bias

class HilbertNetwork(nn.Module):
    def __init__(self, use_dct=False):
        super().__init__()
        self.analog_plate = HilbertHolographicLayer(784, 64, use_dct=use_dct)
        self.bn = nn.BatchNorm1d(64)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.analog_plate(x)
        x = self.bn(x)
        x = self.classifier(x)
        return x

def train_and_evaluate():
    modes = [("V94: Hilbert-Walsh", False), ("V95: Hilbert-DCT", True)]
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=1000)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for name, use_dct in modes:
        print(f"\n=== {name} ===")
        model = HilbertNetwork(use_dct=use_dct).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        for epoch in range(1, 4): # 3 épocas para feedback rápido
            model.train()
            t0 = time.time()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(data), target)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Time: {time.time()-t0:.1f}s")
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        print(f"Final Accuracy {name}: {100. * correct / 10000:.2f}%")

if __name__ == '__main__':
    train_and_evaluate()
