import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import json
import os

class RosettaLayer(nn.Module):
    """
    V23 Component: The Rosetta Layer (Multi-Substrate Library)
    Used for the massive first layer to save parameters.
    """
    def __init__(self, in_features, out_features, rank=32, num_substrates=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_substrates = num_substrates
        
        # K Fixed random substrates
        std = math.sqrt(2.0 / in_features)
        for k in range(num_substrates):
            self.register_buffer(f'w_init_{k}', torch.randn(out_features, in_features) * std)
        
        # Shared modulation parameters
        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.delta_in_a = nn.Parameter(torch.zeros(out_features, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_features))
        
        # Library Attention
        self.library_logits = nn.Parameter(torch.zeros(out_features, num_substrates))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        mix_weights = torch.softmax(self.library_logits, dim=1).unsqueeze(2)
        w_mixed = 0
        for k in range(self.num_substrates):
            w_init_k = getattr(self, f'w_init_{k}')
            w_mixed += mix_weights[:, k, :] * w_init_k
            
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        w_evolved = w_mixed * (1.0 + w_m) + w_a
        
        y = torch.matmul(x, w_evolved.t()) + self.bias
        return self.bn(y)

class HybridNetV23(nn.Module):
    def __init__(self, rank=32, num_substrates=4):
        super().__init__()
        # LAYER 1: ROSETTA (The "Frozen" Sensor - ~170K trainable params vs 6.3M)
        self.layer1 = RosettaLayer(3072, 2048, rank=rank, num_substrates=num_substrates)
        
        # LAYER 2: STANDARD (The "Plastic" Brain - ~2.1M params)
        self.layer2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        
        # LAYER 3: STANDARD (The "Plastic" Decision - ~10K params)
        self.layer3 = nn.Linear(1024, 10)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 3072)
        
        # Sensor phase
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        
        # Brain phase
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        
        # Decision phase
        x = self.layer3(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V23 'THE HYBRID' (Frozen Sensor + Plastic Brain) on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 50
    RANK = 32
    NUM_SUBSTRATES = 4
    MAX_LR = 0.003
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2)
    
    model = HybridNetV23(rank=RANK, num_substrates=NUM_SUBSTRATES).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
    
    # MLP Pure calculation for comparison
    # (3072*2048) + (2048*1024) + (1024*10) = 6.29M + 2.09M + 0.01M = ~8.4M
    print(f"MLP Pure Equivalent: ~8.4M parameters (V23 uses {params/8400000:.1%})")
    
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR / 10, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=len(train_loader)*EPOCHS, 
        pct_start=0.2, anneal_strategy='cos'
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

    t_total = time.time() - t_start
    print(f"\nHYBRID finished in {t_total:.1f}s")
    print(f"Final Best Test Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
