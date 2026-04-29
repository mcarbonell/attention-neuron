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

class HolographicResonatorLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pow2_size = 2**math.ceil(math.log2(in_features))
        
        # Pesos para la proyección base (espacial)
        self.weight_spatial = nn.Parameter(torch.randn(out_features, in_features) / (in_features**0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # MEMORIA HOLOGRÁFICA (Pesos complejos en dominio Walsh)
        # Representamos la "fase" y "amplitud" almacenada como vectores Real/Imag
        self.w_real = nn.Parameter(torch.randn(out_features, self.pow2_size) / (self.pow2_size**0.5))
        self.w_imag = nn.Parameter(torch.randn(out_features, self.pow2_size) / (self.pow2_size**0.5))
        
        # Gating dinámico
        self.gate_fc = nn.Linear(in_features, out_features * 5)
        
    def forward(self, x):
        B = x.size(0)
        O = self.out_features
        N = self.pow2_size
        
        # Gates dinámicos (B, O, 5)
        gates = torch.sigmoid(self.gate_fc(x)).view(B, O, 5)
        
        # Activaciones pre-agregación pesadas
        Z = x.unsqueeze(1) * self.weight_spatial.unsqueeze(0)
        
        # Expertos Clásicos
        a_sum = Z.sum(dim=2)
        a_var = Z.var(dim=2)
        a_l2 = torch.norm(Z, p=2, dim=2)
        a_lse = torch.logsumexp(Z, dim=2)
        
        # EXPERTO HOLOGRÁFICO (Resonancia por Interferencia)
        Z_pad = F.pad(Z, (0, N - self.in_features))
        # FWHT de la señal pesada
        Z_walsh = fwht(Z_pad.view(B * O, N)).view(B, O, N)
        
        # Interferencia Constructiva/Destructiva:
        # Re(Z*W) e Im(Z*W) donde Z es real.
        # Resonancia = sqrt( (Z*W_real)^2 + (Z*W_imag)^2 )
        res_real = (Z_walsh * self.w_real.unsqueeze(0)).sum(dim=2)
        res_imag = (Z_walsh * self.w_imag.unsqueeze(0)).sum(dim=2)
        
        # La activación es la MAGNITUD de la interferencia (Resonancia)
        a_holographic = torch.sqrt(res_real**2 + res_imag**2 + 1e-8)
        
        # Mezcla MoE
        experts = torch.stack([a_sum, a_var, a_l2, a_lse, a_holographic], dim=2)
        out = (experts * gates).sum(dim=2)
        
        return out + self.bias

class HolographicNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.analog_plate = HolographicResonatorLayer(784, 64)
        self.bn = nn.BatchNorm1d(64)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.analog_plate(x)
        x = self.bn(x)
        x = self.classifier(x)
        return x

class BaselineNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.classifier(x)
        return x

def train_and_evaluate():
    print("=== Experimento V90e: Holographic Resonator Plate (Interferencia Walsh) ===\n")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    model_analog = HolographicNetwork().to(device)
    model_base = BaselineNetwork().to(device)
    
    opt_analog = optim.Adam(model_analog.parameters(), lr=0.005)
    opt_base = optim.Adam(model_base.parameters(), lr=0.005)
    
    epochs = 5
    print(f"Entrenando por {epochs} épocas...")
    
    for epoch in range(epochs):
        model_analog.train()
        model_base.train()
        
        t0 = time.time()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            opt_analog.zero_grad()
            out_a = model_analog(data)
            loss_a = F.cross_entropy(out_a, target)
            loss_a.backward()
            opt_analog.step()
            
            opt_base.zero_grad()
            out_b = model_base(data)
            loss_b = F.cross_entropy(out_b, target)
            loss_b.backward()
            opt_base.step()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss Holo: {loss_a.item():.4f} | Loss Base: {loss_b.item():.4f} | Tiempo: {time.time()-t0:.2f}s")

    print("\nEvaluando en Test Set...")
    model_analog.eval()
    model_base.eval()
    
    correct_a = 0
    correct_b = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred_a = model_analog(data).argmax(dim=1, keepdim=True)
            correct_a += pred_a.eq(target.view_as(pred_a)).sum().item()
            
            pred_b = model_base(data).argmax(dim=1, keepdim=True)
            correct_b += pred_b.eq(target.view_as(pred_b)).sum().item()

    acc_a = 100. * correct_a / len(test_loader.dataset)
    acc_b = 100. * correct_b / len(test_loader.dataset)
    
    print(f"\n--- Resultados Finales V90e ---")
    print(f"Baseline (64 SUM + ReLU):   {acc_b:.2f}%")
    print(f"Holographic (64 neur):      {acc_a:.2f}%")

if __name__ == '__main__':
    train_and_evaluate()
