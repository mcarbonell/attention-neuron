"""
scratch/benchmark_ultimate_comparison.py — The Ultimate Optimizer Showdown
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import os

# Importar optimizadores personalizados
from optimizer_adam_ds import AdamDS
from optimizer_lion_ds import LionDS
from optimizer_sign_ds import SignDS
from optimizer_lion import Lion
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
            
            if epoch == 0 and batch_idx < 5:
                print(f"Batch {batch_idx} - Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / len(train_loader.dataset)
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        
    duration = time.time() - t0
    return history, duration

# REGLA DE ORO: PRIMERO EL CANDIDATO Y NOVEDADES
hist_sign_ds, dur_sign_ds = train_model("Sign-DS (2b/p)", SignDS, lr=1e-4, alpha=0.5)
hist_muon, dur_muon = train_model("Muon (4b/p)", MuonClean, lr=0.02)
hist_lion_ds, dur_lion_ds = train_model("Lion-DS (5b/p)", LionDS, lr=1e-4, alpha=0.5)
hist_lion, dur_lion = train_model("Lion Standard (4b/p)", Lion, lr=1e-4)

# Baselines clásicos
hist_adam, dur_adam = train_model("Adam (8b/p)", optim.Adam, lr=1e-3)
hist_rmsprop, dur_rmsprop = train_model("RMSprop (4b/p)", optim.RMSprop, lr=1e-3)
hist_sgd_mom, dur_sgd_mom = train_model("SGD+Momentum (4b/p)", optim.SGD, lr=0.01, momentum=0.9)
hist_sgd, dur_sgd = train_model("SGD Puro (0b/p)", optim.SGD, lr=0.01)

# Visualización
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(hist_sign_ds['loss'], label='Sign-DS (2b)', marker='v', color='purple')
plt.plot(hist_muon['loss'], label='Muon (4b)', marker='x', color='orange')
plt.plot(hist_lion_ds['loss'], label='Lion-DS (5b)', marker='^', color='red')
plt.plot(hist_lion['loss'], label='Lion (4b)', marker='1', color='brown')
plt.plot(hist_adam['loss'], label='Adam (8b)', marker='o', color='blue')
plt.plot(hist_rmsprop['loss'], label='RMSprop (4b)', marker='D', color='cyan')
plt.plot(hist_sgd_mom['loss'], label='SGD+Mom (4b)', marker='s', color='green')
plt.plot(hist_sgd['loss'], label='SGD Puro (0b)', marker='.', color='black')
plt.title('Pérdida (Loss) por Época')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(hist_sign_ds['acc'], label='Sign-DS', marker='v', color='purple')
plt.plot(hist_muon['acc'], label='Muon', marker='x', color='orange')
plt.plot(hist_lion_ds['acc'], label='Lion-DS', marker='^', color='red')
plt.plot(hist_lion['acc'], label='Lion', marker='1', color='brown')
plt.plot(hist_adam['acc'], label='Adam', marker='o', color='blue')
plt.plot(hist_rmsprop['acc'], label='RMSprop', marker='D', color='cyan')
plt.plot(hist_sgd_mom['acc'], label='SGD+Mom', marker='s', color='green')
plt.plot(hist_sgd['acc'], label='SGD Puro', marker='.', color='black')
plt.title('Precisión (Accuracy) por Época')
plt.legend()
plt.grid(True)

plt.tight_layout()
os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/ultimate_optimizer_comparison.png')
print(f"\nBenchmark finalizado.")

# Tabla de Tiempos y Memoria (Consola)
print("\n" + "="*50)
print(f"{'Optimizador':<20} | {'Memoria':<10} | {'Tiempo':<10} | {'Acc Final':<10}")
print("-" * 50)
print(f"{'Sign-DS':<20} | {'2b/p':<10} | {dur_sign_ds:<10.1f}s | {hist_sign_ds['acc'][-1]:<10.4f}")
print(f"{'Muon':<20} | {'4b/p':<10} | {dur_muon:<10.1f}s | {hist_muon['acc'][-1]:<10.4f}")
print(f"{'Lion-DS':<20} | {'5b/p':<10} | {dur_lion_ds:<10.1f}s | {hist_lion_ds['acc'][-1]:<10.4f}")
print(f"{'Lion':<20} | {'4b/p':<10} | {dur_lion:<10.1f}s | {hist_lion['acc'][-1]:<10.4f}")
print(f"{'Adam':<20} | {'8b/p':<10} | {dur_adam:<10.1f}s | {hist_adam['acc'][-1]:<10.4f}")
print(f"{'RMSprop':<20} | {'4b/p':<10} | {dur_rmsprop:<10.1f}s | {hist_rmsprop['acc'][-1]:<10.4f}")
print(f"{'SGD+Mom':<20} | {'4b/p':<10} | {dur_sgd_mom:<10.1f}s | {hist_sgd_mom['acc'][-1]:<10.4f}")
print(f"{'SGD Puro':<20} | {'0b/p':<10} | {dur_sgd:<10.1f}s | {hist_sgd['acc'][-1]:<10.4f}")
print("="*50)

plt.show()
