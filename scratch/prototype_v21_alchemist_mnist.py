import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import json
import os

class AlchemistLayer(nn.Module):
    """
    V21: The Alchemist Layer (Double Substrate)
    Uses two fixed random matrices W_init_A and W_init_B.
    A learnable 'alpha' dial per neuron decides the mixture.
    """
    def __init__(self, in_features, out_features, rank=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # TWO Fixed base weights (Sustrato A y B)
        std = math.sqrt(2.0 / in_features)
        self.register_buffer('w_init_a', torch.randn(out_features, in_features) * std)
        self.register_buffer('w_init_b', torch.randn(out_features, in_features) * std)
        
        # Modulations (we share the same modulation parameters for efficiency, 
        # or we could have double too, but let's start with shared modulation 
        # acting on a mixed substrate)
        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.delta_in_a = nn.Parameter(torch.zeros(out_features, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_features))
        
        # The Alchemist's Dial: alpha logits per neuron
        # 0.0 means 50/50 mix
        self.alpha_logits = nn.Parameter(torch.zeros(out_features))
        
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        # 1. Mix the substrates
        alpha = torch.sigmoid(self.alpha_logits).view(-1, 1)
        w_mixed = alpha * self.w_init_a + (1.0 - alpha) * self.w_init_b
        
        # 2. Apply modulation to the mixed substrate
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        
        w_evolved = w_mixed * (1.0 + w_m) + w_a
        
        y = torch.matmul(x, w_evolved.t()) + self.bias
        return self.bn(y)

class AlchemistNetV21(nn.Module):
    def __init__(self, rank=64):
        super().__init__()
        # 784 -> 1024 -> 1024 -> 10
        self.layer1 = AlchemistLayer(784, 1024, rank=rank)
        self.layer2 = AlchemistLayer(1024, 1024, rank=rank)
        self.layer3 = AlchemistLayer(1024, 10, rank=rank)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V21 'THE ALCHEMIST' (Double Substrate) on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 30
    RANK = 64
    MAX_LR = 0.005
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    model = AlchemistNetV21(rank=RANK).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR / 10, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=len(train_loader)*EPOCHS, 
        pct_start=0.3, anneal_strategy='cos'
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0
    t_start = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = correct / 10000
        if acc > best_acc:
            best_acc = acc
            
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f}")

    # Final Analysis of the Dials
    print("\n--- Alchemist Analysis: Dial Distribution (A vs B) ---")
    for i, layer in enumerate([model.layer1, model.layer2, model.layer3]):
        alphas = torch.sigmoid(layer.alpha_logits).detach().cpu().numpy()
        a_pref = (alphas > 0.7).sum()
        b_pref = (alphas < 0.3).sum()
        mixed = len(alphas) - a_pref - b_pref
        print(f"Layer {i+1}: Prefers A: {a_pref} | Prefers B: {b_pref} | Mixed: {mixed}")

    t_total = time.time() - t_start
    print(f"\nALCHEMIST finished in {t_total:.1f}s")
    print(f"Final Best Test Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
