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

class PrismConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, rank=16, num_substrates=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.num_substrates = num_substrates
        
        std = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        for k in range(num_substrates):
            self.register_buffer(f'w_init_{k}', torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std)
            
        self.mix_logits = nn.Parameter(torch.zeros(out_channels, num_substrates))
        
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
            
        m_chan = torch.matmul(self.delta_in_m, self.delta_out_m).view(self.out_channels, self.in_channels, 1, 1)
        a_chan = torch.matmul(self.delta_in_a, self.delta_out_a).view(self.out_channels, self.in_channels, 1, 1)
        
        w_evolved = w_mixed * (1.0 + m_chan) + a_chan
        return self.bn(F.conv2d(x, w_evolved, stride=self.stride, padding=self.padding) + self.bias.view(1, -1, 1, 1))

class PrismBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, rank=16):
        super().__init__()
        self.conv1 = PrismConvLayer(in_planes, planes, stride=stride, rank=rank)
        self.conv2 = PrismConvLayer(planes, planes, stride=1, rank=rank)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        return F.relu(out)

class PrismResNet(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2, 2], rank=16):
        super().__init__()
        self.in_planes = 64
        self.conv1 = PrismConvLayer(3, 64, kernel_size=3, stride=1, padding=1, rank=rank)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, rank=rank)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, rank=rank)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, rank=rank)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2, rank=rank)
        self.linear = nn.Linear(512, 10)

    def _make_layer(self, planes, num_blocks, stride, rank):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(PrismBasicBlock(self.in_planes, planes, s, rank))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return self.linear(out.view(out.size(0), -1))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V26 'THE PRISM-RESNET' (ResNet-18 + Multi-Substrate) on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 50
    LR = 0.003
    
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
    
    train_loader = DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transform_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(datasets.CIFAR10('./data', train=False, transform=transform_test), batch_size=1024, num_workers=2)
    
    model = PrismResNet().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR/10)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=len(train_loader)*EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0
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
            scheduler.step()
            
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = correct / 10000
        if acc > best_acc: best_acc = acc
        
        t_now = time.time()
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] Epoch {epoch:2d}/{EPOCHS} | Acc: {acc:.4f} | Best: {best_acc:.4f} | Time: {t_now-t0:.1f}s | ETA: {(t_now-t_start)/epoch*(EPOCHS-epoch)/60:.1f}m")

if __name__ == "__main__":
    main()
