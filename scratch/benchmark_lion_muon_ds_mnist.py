"""
scratch/benchmark_lion_muon_ds_mnist.py — Muon Duel: Candidate vs Clean
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from optimizer_lion_muon_ds import LionMuonDS
from optimizer_muon_clean import MuonClean

# Configuración
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
EPOCHS = 10

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def train_model(name, optimizer_class, **opt_kwargs):
    print(f"\n--- Entrenando con {name} ---")
    model = SimpleNet().to(device)
    optimizer = optimizer_class(model.parameters(), **opt_kwargs)
    criterion = nn.CrossEntropyLoss()
    
    history = {'loss': [], 'acc': []}
    t0 = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / len(train_loader.dataset)
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        
    duration = time.time() - t0
    return history, duration

# REGLA DE ORO: PRIMERO EL CANDIDATO
# Usamos el LR de Muon (0.01) para ambos
hist_muon_ds, dur_muon_ds = train_model("Lion-Muon-DS (Candidate)", LionMuonDS, lr=0.01, alpha=0.5)

# Luego el baseline original
hist_muon_clean, dur_muon_clean = train_model("Muon Original (Clean)", MuonClean, lr=0.01)

# Visualización
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(hist_muon_ds['loss'], label='Lion-Muon-DS', marker='*', color='purple')
plt.plot(hist_muon_clean['loss'], label='Muon Clean', marker='o', color='black')
plt.title('Pérdida (Loss) por Época')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(hist_muon_ds['acc'], label='Lion-Muon-DS', marker='*', color='purple')
plt.plot(hist_muon_clean['acc'], label='Muon Clean', marker='o', color='black')
plt.title('Precisión (Accuracy) por Época')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results/figures/muon_duel_comparison.png')
print(f"\nComparativa finalizada.")
print(f"Tiempos: Candidate {dur_muon_ds:.1f}s, Clean {dur_muon_clean:.1f}s")
plt.show()
