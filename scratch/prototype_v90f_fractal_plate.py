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

class FractalHolographicLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # --- Escala Global (28x28 -> 1024) ---
        self.pow2_global = 1024
        self.w_global_real = nn.Parameter(torch.randn(out_features, self.pow2_global) / (self.pow2_global**0.5))
        self.w_global_imag = nn.Parameter(torch.randn(out_features, self.pow2_global) / (self.pow2_global**0.5))
        
        # --- Escala Local (Parches de 7x7 -> 64) ---
        # Dividimos la imagen 28x28 en 16 parches de 7x7
        self.pow2_local = 64
        self.w_local_real = nn.Parameter(torch.randn(out_features, self.pow2_local) / (self.pow2_local**0.5))
        self.w_local_imag = nn.Parameter(torch.randn(out_features, self.pow2_local) / (self.pow2_local**0.5))
        
        # Proyección clásica y gating
        self.weight_spatial = nn.Parameter(torch.randn(out_features, in_features) / (in_features**0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_fc = nn.Linear(in_features, out_features * 4) # 4 expertos: Global, Local, Sum, LSE
        
    def forward(self, x):
        B = x.size(0)
        O = self.out_features
        
        # Gates (B, O, 4)
        gates = torch.sigmoid(self.gate_fc(x)).view(B, O, 4)
        
        # 1. Expertos Espaciales
        Z = x.unsqueeze(1) * self.weight_spatial.unsqueeze(0)
        a_sum = Z.sum(dim=2)
        a_lse = torch.logsumexp(Z, dim=2)
        
        # 2. Experto Holográfico GLOBAL
        Z_global = F.pad(Z, (0, self.pow2_global - self.in_features))
        W_global = fwht(Z_global.view(B * O, self.pow2_global)).view(B, O, self.pow2_global)
        res_g_real = (W_global * self.w_global_real.unsqueeze(0)).sum(dim=2)
        res_g_imag = (W_global * self.w_global_imag.unsqueeze(0)).sum(dim=2)
        a_global = torch.sqrt(res_g_real**2 + res_g_imag**2 + 1e-8)
        
        # 3. Experto Holográfico LOCAL (Fractal)
        # Re-organizamos x (B, 784) a (B, 1, 28, 28)
        x_img = x.view(B, 1, 28, 28)
        # Extraemos 16 parches de 7x7
        patches = F.unfold(x_img, kernel_size=7, stride=7) # (B, 49, 16)
        patches = patches.transpose(1, 2) # (B, 16, 49)
        # Pad parches a 64
        patches_pad = F.pad(patches, (0, self.pow2_local - 49)) # (B, 16, 64)
        # FWHT de parches
        W_local = fwht(patches_pad.view(B * 16, self.pow2_local)).view(B, 16, self.pow2_local)
        # Resonancia local (cada neurona promedia la resonancia de los 16 parches)
        # Z_local: (B, O, 16, 64) -> Interferencia de cada parche con el patrón local de la neurona
        # Para simplificar: proyectamos parches y luego resonamos
        # Pero para ser "holográfico" puro, cada neurona tiene un patrón local:
        res_l_real = (W_local.unsqueeze(1) * self.w_local_real.view(1, O, 1, 64)).sum(dim=3).mean(dim=2)
        res_l_imag = (W_local.unsqueeze(1) * self.w_local_imag.view(1, O, 1, 64)).sum(dim=3).mean(dim=2)
        a_local = torch.sqrt(res_l_real**2 + res_l_imag**2 + 1e-8)
        
        # Mezcla Fractal
        experts = torch.stack([a_sum, a_lse, a_global, a_local], dim=2)
        out = (experts * gates).sum(dim=2)
        
        return out + self.bias

class FractalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.analog_plate = FractalHolographicLayer(784, 64)
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
    print("=== Experimento V90f: Fractal Resonator Plate (Multiscale Holograms) ===\n")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    model_analog = FractalNetwork().to(device)
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
            
        print(f"Epoch {epoch+1}/{epochs} | Loss Fractal: {loss_a.item():.4f} | Loss Base: {loss_b.item():.4f} | Tiempo: {time.time()-t0:.2f}s")

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
    
    print(f"\n--- Resultados Finales V90f ---")
    print(f"Baseline (64 SUM + ReLU):   {acc_b:.2f}%")
    print(f"Fractal Plate (64 neur):    {acc_a:.2f}%")

if __name__ == '__main__':
    train_and_evaluate()
