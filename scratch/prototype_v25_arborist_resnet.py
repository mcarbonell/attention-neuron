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
import numpy as np

class ArboristConvLayer(nn.Module):
    """
    V25 Component: Dendritic Tree Tuning.
    Mixes 8 substrates using a binary tree of 7 learnable dials.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, rank=32, num_substrates=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.rank = rank
        self.num_substrates = num_substrates
        assert num_substrates == 8, "Arborist is hardcoded for a depth-3 tree (8 leaves)"
        
        # 8 Fixed random kernel substrates
        std = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        for k in range(num_substrates):
            self.register_buffer(f'w_init_{k}', torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std)
            
        # Dendritic Dials: 7 nodes for a binary tree of 8 leaves
        self.dials = nn.Parameter(torch.zeros(out_channels, 7))
        
        # Dual Channel Modulation
        self.delta_in_m = nn.Parameter(torch.randn(out_channels, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_channels) * 0.01)
        self.delta_in_a = nn.Parameter(torch.zeros(out_channels, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_channels))

    def forward(self, x):
        # 1. Dendritic Mixing (Tree structure)
        a = torch.sigmoid(self.dials).view(self.out_channels, 7, 1, 1, 1)
        
        # Level 0 (Leaves)
        s0, s1 = self.w_init_0, self.w_init_1
        s2, s3 = self.w_init_2, self.w_init_3
        s4, s5 = self.w_init_4, self.w_init_5
        s6, s7 = self.w_init_6, self.w_init_7
        
        # Level 1 (Branches)
        l1_0 = a[:, 0] * s0 + (1 - a[:, 0]) * s1
        l1_1 = a[:, 1] * s2 + (1 - a[:, 1]) * s3
        l1_2 = a[:, 2] * s4 + (1 - a[:, 2]) * s5
        l1_3 = a[:, 3] * s6 + (1 - a[:, 3]) * s7
        
        # Level 2 (Main Branches)
        l2_0 = a[:, 4] * l1_0 + (1 - a[:, 4]) * l1_1
        l2_1 = a[:, 5] * l1_2 + (1 - a[:, 5]) * l1_3
        
        # Level 3 (Trunk)
        w_mixed = a[:, 6] * l2_0 + (1 - a[:, 6]) * l2_1
        
        # 2. Apply modulation to the mixed trunk
        m_chan = torch.matmul(self.delta_in_m, self.delta_out_m).view(self.out_channels, self.in_channels, 1, 1)
        a_chan = torch.matmul(self.delta_in_a, self.delta_out_a).view(self.out_channels, self.in_channels, 1, 1)
        
        w_evolved = w_mixed * (1.0 + m_chan) + a_chan
        
        # 3. Convolution
        return F.conv2d(x, w_evolved, stride=self.stride, padding=self.padding)

class ArboristBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, rank=32):
        super().__init__()
        self.conv1 = ArboristConvLayer(in_planes, planes, stride=stride, rank=rank)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = ArboristConvLayer(planes, planes, stride=1, rank=rank)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            # 1x1 standard conv for dimensions matching (plastic)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ArboristResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, rank=32):
        super().__init__()
        self.in_planes = 64

        self.conv1 = ArboristConvLayer(3, 64, kernel_size=3, stride=1, padding=1, rank=rank)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, rank=rank)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, rank=rank)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, rank=rank)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, rank=rank)
        
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, rank):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, stride=s, rank=rank))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ArboristResNet18(rank=32):
    return ArboristResNet(ArboristBasicBlock, [2, 2, 2, 2], rank=rank)

# --- MIXUP UTILS ---
def mixup_data(x, y, alpha=1.0, device='cpu'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V25 'THE GREAT ARBORIST' (ResNet-18 + Dendritic Tree) on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 100 # ResNet needs time to mature
    RANK = 32
    MAX_LR = 0.005
    MIXUP_ALPHA = 1.0 # Standard mixup strength
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Extra robust
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
    
    model = ArboristResNet18(rank=RANK).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,} (Real ResNet-18 has ~11.1M)")
    
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR / 10, weight_decay=1e-3)
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
        t_epoch_start = time.time()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            # Apply Mixup
            mixed_data, target_a, target_b, lam = mixup_data(data, target, MIXUP_ALPHA, device)
            
            optimizer.zero_grad()
            output = model(mixed_data)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            
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
            torch.save(model.state_dict(), "results/v25_arborist_best.pt")
            
        t_now = time.time()
        elapsed = t_now - t_start
        epoch_time = t_now - t_epoch_start
        eta = (elapsed / epoch) * (EPOCHS - epoch)
        
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] Epoch {epoch:3d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f} | Time: {epoch_time:.1f}s | ETA: {eta/60:.1f}m")

    t_total = time.time() - t_start
    print(f"\nTHE GREAT ARBORIST finished in {t_total:.1f}s")
    print(f"Final Best Test Accuracy: {best_acc:.4f}")
    
    results = {
        "model": "V25_Arborist_ResNet18",
        "rank": RANK,
        "trainable_params": params,
        "best_acc": best_acc,
        "epochs": EPOCHS,
        "wall_clock_time": t_total,
        "dataset": "CIFAR-10",
        "mixup": True
    }
    
    with open("results/raw/v25_arborist_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
