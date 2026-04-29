import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

class AnalogLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 4 grupos matemáticos
        self.groups = 4
        self.group_size = out_features // self.groups
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / (in_features**0.5))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # x: (B, in_features)
        # Expandimos para hacer la ponderación pre-agregación: (B, out_features, in_features)
        Z = x.unsqueeze(1) * self.weight.unsqueeze(0)
        
        # Grupo 1: SUMA LINEAL CLÁSICA (Acumulador)
        out_sum = Z[:, 0:self.group_size, :].sum(dim=2)
        
        # Grupo 2: VARIANZA (Detector de Contraste/Dispersión)
        out_var = Z[:, self.group_size:2*self.group_size, :].var(dim=2)
        
        # Grupo 3: NORMA L2 (Detector de Energía/Picos)
        out_l2 = torch.norm(Z[:, 2*self.group_size:3*self.group_size, :], p=2, dim=2)
        
        # Grupo 4: LOG-SUM-EXP (Soft-Max / Enrutador suave)
        out_lse = torch.logsumexp(Z[:, 3*self.group_size:4*self.group_size, :], dim=2)
        
        # Concatenamos la respuesta de la placa analógica
        out = torch.cat([out_sum, out_var, out_l2, out_lse], dim=1)
        return out + self.bias

class AnalogNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Una sola capa oculta "Shallow" de 64 neuronas (16 de cada tipo)
        self.analog_plate = AnalogLayer(784, 64)
        self.bn = nn.BatchNorm1d(64) # Estabilizador de escalas entre funciones
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.analog_plate(x)
        x = self.bn(x) # No usamos ReLU porque VAR, L2 y LSE ya son fuertemente no-lineales
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
    print("=== Experimento V90: Placa Analógica Polimórfica vs Baseline (Shallow Networks) ===\n")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model_analog = AnalogNetwork()
    model_base = BaselineNetwork()
    
    opt_analog = optim.Adam(model_analog.parameters(), lr=0.005)
    opt_base = optim.Adam(model_base.parameters(), lr=0.005)
    
    epochs = 3
    print("Entrenando ambas redes (1 capa oculta de 64 neuronas) por 3 épocas...")
    
    for epoch in range(epochs):
        model_analog.train()
        model_base.train()
        
        t0 = time.time()
        for data, target in train_loader:
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
            
        print(f"Epoch {epoch+1} completada en {time.time()-t0:.2f}s | Loss Analog: {loss_a.item():.4f} | Loss Base: {loss_b.item():.4f}")

    print("\nEvaluando en Test Set...")
    model_analog.eval()
    model_base.eval()
    
    correct_a = 0
    correct_b = 0
    with torch.no_grad():
        for data, target in test_loader:
            pred_a = model_analog(data).argmax(dim=1, keepdim=True)
            correct_a += pred_a.eq(target.view_as(pred_a)).sum().item()
            
            pred_b = model_base(data).argmax(dim=1, keepdim=True)
            correct_b += pred_b.eq(target.view_as(pred_b)).sum().item()

    acc_a = 100. * correct_a / len(test_loader.dataset)
    acc_b = 100. * correct_b / len(test_loader.dataset)
    
    print(f"\n--- Resultados Finales V90 ---")
    print(f"Baseline (64 SUM + ReLU):        {acc_b:.2f}%")
    print(f"Placa Analógica (SUM/VAR/L2/LSE): {acc_a:.2f}%")
    
    if acc_a > acc_b:
        print("¡ÉXITO! La Placa Analógica supera a la red estándar con el mismo número de neuronas.")
    else:
        print("Interesante. La diversidad matemática cambia la dinámica de aprendizaje.")

if __name__ == '__main__':
    train_and_evaluate()
