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

class AttentionConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, rank=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.rank = rank
        
        # Fixed base weights (Kaiming Normal)
        std = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.register_buffer('w_init', torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std)
        
        # Dual modulation on channels (rank-r)
        # We modulate the input and output channels of the convolution
        self.delta_in_m = nn.Parameter(torch.randn(out_channels, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_channels) * 0.01)
        
        self.delta_in_a = nn.Parameter(torch.zeros(out_channels, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_channels))
        
        # Bias and BatchNorm
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Channel modulation matrix (Out x In)
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m).view(self.out_channels, self.in_channels, 1, 1)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a).view(self.out_channels, self.in_channels, 1, 1)
        
        # Evolve the fixed kernels
        w_evolved = self.w_init * (1.0 + w_m) + w_a
        
        y = F.conv2d(x, w_evolved, padding=self.padding) + self.bias.view(1, -1, 1, 1)
        return self.bn(y)

class AttentionLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=32):
        super().__init__()
        std = math.sqrt(2.0 / in_features)
        self.register_buffer('w_init', torch.randn(out_features, in_features) * std)
        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.delta_in_a = nn.Parameter(torch.zeros(out_features, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        w_evolved = self.w_init * (1.0 + w_m) + w_a
        return self.bn(torch.matmul(x, w_evolved.t()) + self.bias)

class NavigatorNetV19(nn.Module):
    def __init__(self, rank=32):
        super().__init__()
        # Block 1: 32x32 -> 16x16
        self.conv1 = AttentionConv2dLayer(3, 64, rank=rank)
        self.conv2 = AttentionConv2dLayer(64, 64, rank=rank)
        self.pool1 = nn.MaxPool2d(2)
        
        # Block 2: 16x16 -> 8x8
        self.conv3 = AttentionConv2dLayer(64, 128, rank=rank)
        self.conv4 = AttentionConv2dLayer(128, 128, rank=rank)
        self.pool2 = nn.MaxPool2d(2)
        
        # Block 3: 8x8 -> 4x4
        self.conv5 = AttentionConv2dLayer(128, 256, rank=rank)
        self.conv6 = AttentionConv2dLayer(256, 256, rank=rank)
        self.pool3 = nn.MaxPool2d(2)
        
        # Global Average Pooling + Clasificador
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = AttentionLinearLayer(256, 10, rank=rank)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.dropout(x)
        
        x = self.gap(x).view(-1, 256)
        x = self.fc(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V19 'THE NAVIGATOR' (CIFAR-10) on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 50
    RANK = 32
    MAX_LR = 0.003
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
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
    
    os.makedirs("results", exist_ok=True)
    
    model = NavigatorNetV19(rank=RANK).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
    
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
            torch.save(model.state_dict(), "results/v19_navigator_best.pt")
            
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f}")
        
    t_total = time.time() - t_start
    print(f"\nNAVIGATOR finished in {t_total:.1f}s")
    print(f"Final Best Test Accuracy: {best_acc:.4f}")
    
    # Save results
    results = {
        "model": "V19_Navigator",
        "rank": RANK,
        "trainable_params": params,
        "best_acc": best_acc,
        "epochs": EPOCHS,
        "wall_clock_time": t_total,
        "dataset": "CIFAR-10"
    }
    
    with open("results/raw/v19_navigator_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
