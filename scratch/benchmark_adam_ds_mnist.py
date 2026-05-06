"""
scratch/benchmark_adam_ds_mnist.py — Adam vs Adam-DS Head-to-Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from optimizer_adam_ds import AdamDS

# Configuración
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modelo Simple (MLP) para estresar el optimizador
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
    optimizer = optimizer_class(model.parameters(), lr=LR, **opt_kwargs)
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
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / len(train_loader.dataset)
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        print(f"Epoch {epoch} Finalizada - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")
        
    duration = time.time() - t0
    return history, duration

# Ejecutar comparativa
hist_adam, dur_adam = train_model("Adam Estándar", optim.Adam)
hist_adam_ds, dur_adam_ds = train_model("Adam-DS (Stability)", AdamDS, alpha=0.5)

# Visualización de Resultados
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(hist_adam['loss'], label='Adam', marker='o')
plt.plot(hist_adam_ds['loss'], label='Adam-DS', marker='s')
plt.title('Pérdida (Loss) por Época')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(hist_adam['acc'], label='Adam', marker='o')
plt.plot(hist_adam_ds['acc'], label='Adam-DS', marker='s')
plt.title('Precisión (Accuracy) por Época')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results/figures/adam_vs_adam_ds_mnist.png')
print(f"\nComparativa finalizada.")
print(f"Tiempo Adam: {dur_adam:.1f}s")
print(f"Tiempo Adam-DS: {dur_adam_ds:.1f}s")
print("Gráfica guardada en results/figures/adam_vs_adam_ds_mnist.png")
plt.show()
