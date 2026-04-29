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

class GatedAnalogLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pow2_size = 2**math.ceil(math.log2(in_features))
        
        # Pesos compartidos para la proyección base
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / (in_features**0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Gating: ¿Cuánto de cada experto dejamos pasar?
        # Usamos una proyección lineal de la entrada para determinar el gate
        self.gate_fc = nn.Linear(in_features, out_features * 5)
        
    def forward(self, x):
        # x: (B, in_features)
        B = x.size(0)
        
        # Cálculo de los Gates (B, out_features, 5)
        gates = torch.sigmoid(self.gate_fc(x)).view(B, self.out_features, 5)
        
        # Activaciones pre-agregación pesadas: (B, out_features, in_features)
        Z = x.unsqueeze(1) * self.weight.unsqueeze(0)
        
        # Agregadores (Expertos)
        # 0: SUM
        a_sum = Z.sum(dim=2)
        # 1: VAR
        a_var = Z.var(dim=2)
        # 2: L2
        a_l2 = torch.norm(Z, p=2, dim=2)
        # 3: LSE
        a_lse = torch.logsumexp(Z, dim=2)
        # 4: WALSH
        Z_pad = F.pad(Z, (0, self.pow2_size - self.in_features))
        _, O, N = Z_pad.shape
        Z_walsh = fwht(Z_pad.view(B * O, N)).view(B, O, N)
        a_walsh = Z_walsh.abs().mean(dim=2)
        
        # Stack: (B, out_features, 5)
        experts = torch.stack([a_sum, a_var, a_l2, a_lse, a_walsh], dim=2)
        
        # Aplicación del Gate dinámico
        # Cada neurona decide dinámicamente qué agregadores usar según la entrada actual
        out = (experts * gates).sum(dim=2)
        
        return out + self.bias

class GatedAnalogNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.analog_plate = GatedAnalogLayer(784, 64)
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
    print("=== Experimento V90c: Placa Analógica GATED (Dynamic Selection) ===\n")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    model_analog = GatedAnalogNetwork().to(device)
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
            
        print(f"Epoch {epoch+1}/{epochs} | Loss Gated: {loss_a.item():.4f} | Loss Base: {loss_b.item():.4f} | Tiempo: {time.time()-t0:.2f}s")

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
    
    print(f"\n--- Resultados Finales V90c ---")
    print(f"Baseline (64 SUM + ReLU):   {acc_b:.2f}%")
    print(f"Placa GATED (64 experts):   {acc_a:.2f}%")

if __name__ == '__main__':
    train_and_evaluate()
