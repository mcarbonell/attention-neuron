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

class BroadcasterConvLayer(nn.Module):
    """
    V32 Component: The Broadcaster (Fan-out Modulation).
    W is 100% frozen. We mix the OUTPUTS (activations) of K fixed random convolutions.
    This is mathematically equivalent to mixing the weights, but computationally much faster
    because we don't have to evolve a massive weight tensor every forward pass.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, rank=16, num_substrates=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_substrates = num_substrates
        
        # 1. The Fixed Sensors (K frozen random convolutions)
        # We use a single grouped convolution for extreme speed instead of a loop
        # Input: (B, C, H, W) -> Expand to (B, C*K, H, W) -> Group Conv -> (B, Out*K, H', W')
        std = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        
        # We create one giant fixed weight tensor containing all K substrates
        # Shape: (out_channels * num_substrates, in_channels, kernel_size, kernel_size)
        w_frozen = torch.randn(out_channels * num_substrates, in_channels, kernel_size, kernel_size) * std
        
        self.conv_frozen = nn.Conv2d(
            in_channels, 
            out_channels * num_substrates, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=False
        )
        self.conv_frozen.weight.data = w_frozen
        self.conv_frozen.weight.requires_grad = False # 100% Frozen!
        
        # 2. The Library Dial (Softmax over outputs)
        # One dial per output channel mixing its K versions
        self.library_logits = nn.Parameter(torch.zeros(out_channels, num_substrates))
        
        # 3. Post-Activation Fan-out Modulation (Rank-r approximation of a dense mixing)
        # Instead of modifying weights, we apply a channel-wise affine transform
        self.gain = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Fast Frozen Computation (All K substrates evaluated simultaneously)
        # Shape: (B, out_C * K, H', W')
        with torch.no_grad():
            raw_out = self.conv_frozen(x)
            
        # Reshape to separate channels and substrates: (B, out_C, K, H', W')
        _, _, H_out, W_out = raw_out.shape
        raw_out = raw_out.view(B, self.out_channels, self.num_substrates, H_out, W_out)
        
        # 2. Fan-out Mixing (The Alchemist Dial applied to activations)
        # mix shape: (1, out_C, K, 1, 1)
        mix = torch.softmax(self.library_logits, dim=1).view(1, self.out_channels, self.num_substrates, 1, 1)
        
        # Multiply and sum over substrates (K)
        # Shape: (B, out_C, H', W')
        mixed_out = torch.sum(raw_out * mix, dim=2)
        
        # 3. Fan-out Modulation (Gain & Bias)
        y = mixed_out * self.gain.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        
        return self.bn(y)

class BroadcasterBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, rank=16):
        super().__init__()
        self.conv1 = BroadcasterConvLayer(in_planes, planes, stride=stride, rank=rank)
        self.conv2 = BroadcasterConvLayer(planes, planes, stride=1, rank=rank)
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

class BroadcasterResNet(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2, 2], rank=16):
        super().__init__()
        self.in_planes = 64
        self.conv1 = BroadcasterConvLayer(3, 64, kernel_size=3, stride=1, padding=1, rank=rank)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, rank=rank)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, rank=rank)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, rank=rank)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2, rank=rank)
        self.linear = nn.Linear(512, 10)

    def _make_layer(self, planes, num_blocks, stride, rank):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(BroadcasterBasicBlock(self.in_planes, planes, s, rank))
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
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"Training V32 'THE BROADCASTER' (Fan-out Modulation ResNet) on: DirectML ({device})")
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training V32 'THE BROADCASTER' (Fan-out Modulation ResNet) on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 50
    RANK = 16
    NUM_SUBSTRATES = 4
    MAX_LR = 0.005 # Faster convergence expected due to direct activation scaling
    
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
    
    os.makedirs("results", exist_ok=True)
    
    model = BroadcasterResNet().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable Parameters: {params:,} (Frozen Base: {frozen_params:,})")
    
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR / 10, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, total_steps=len(train_loader)*EPOCHS, pct_start=0.2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Log the first 5 batches of epoch 1 to ensure it's alive
            if epoch == 1 and batch_idx < 5:
                print(f"[Batch {batch_idx+1}/5] Loss: {loss.item():.4f} | Time: {time.time()-t0:.1f}s")
                
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

    t_total = time.time() - t_start
    print(f"\nBROADCASTER finished in {t_total:.1f}s")
    print(f"Final Best Test Accuracy: {best_acc:.4f}")
    
    results = {
        "model": "V32_Broadcaster_ResNet",
        "trainable_params": params,
        "frozen_params": frozen_params,
        "best_acc": best_acc,
        "epochs": EPOCHS,
        "wall_clock_time": t_total,
        "dataset": "CIFAR-10"
    }
    
    with open("results/raw/v32_broadcaster_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
