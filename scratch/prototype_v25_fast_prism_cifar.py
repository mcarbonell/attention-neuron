import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import json
import os

class FastPrismLayer(nn.Module):
    def __init__(self, in_channels, out_channels, rank=16, num_substrates=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_substrates = num_substrates
        
        # 4 Fixed random substrates
        std = math.sqrt(2.0 / (in_channels * 9))
        for k in range(num_substrates):
            self.register_buffer(f'w_init_{k}', torch.randn(out_channels, in_channels, 3, 3) * std)
            
        # Linear mix weights per output channel
        self.mix_logits = nn.Parameter(torch.zeros(out_channels, num_substrates))
        
        # Rank-r modulation
        self.delta_in_m = nn.Parameter(torch.randn(out_channels, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_channels) * 0.01)
        self.delta_in_a = nn.Parameter(torch.zeros(out_channels, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_channels))
        
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        mix = torch.softmax(self.mix_logits, dim=1).view(self.out_channels, self.num_substrates, 1, 1, 1)
        w_mixed = 0
        for k in range(self.num_substrates):
            w_mixed += mix[:, k] * getattr(self, f'w_init_{k}')
            
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m).view(self.out_channels, self.in_channels, 1, 1)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a).view(self.out_channels, self.in_channels, 1, 1)
        
        w_evolved = w_mixed * (1.0 + w_m) + w_a
        return self.bn(F.conv2d(x, w_evolved, padding=1) + self.bias.view(1, -1, 1, 1))

class FastPrismNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = FastPrismLayer(3, 64)
        self.c2 = FastPrismLayer(64, 128)
        self.c3 = FastPrismLayer(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.pool(x) # 16x16
        x = F.relu(self.c2(x))
        x = self.pool(x) # 8x8
        x = F.relu(self.c3(x))
        x = self.pool(x) # 4x4
        x = self.gap(x).view(-1, 256)
        return self.fc(x)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V25_FAST (Prism CNN) on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 20 # Quick iteration
    LR = 0.003
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_loader = DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.CIFAR10('./data', train=False, transform=transform), batch_size=1024)
    
    model = FastPrismNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = correct / 10000
        t_now = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch:2d}/{EPOCHS} | Acc: {acc:.4f} | Time: {t_now - t0:.1f}s | ETA: {(t_now-t_start)/epoch*(EPOCHS-epoch)/60:.1f}m")

if __name__ == "__main__":
    main()
