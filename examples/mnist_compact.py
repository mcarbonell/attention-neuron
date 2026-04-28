import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys
import time

# Add the parent directory to sys.path to import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention_neuron import DCTLinear

class CompactDCTNet(nn.Module):
    """
    V63 Architecture: High-compression MLP for MNIST using DCT routing.
    Achieves >97% accuracy with only ~12k parameters.
    """
    def __init__(self):
        super().__init__()
        
        # Layer 1: 784 -> 512 (Core: 64x64)
        self.layer1 = DCTLinear(784, 512, k_in=64, k_out=64)
        self.bn1 = nn.BatchNorm1d(512)
        
        # Layer 2: 512 -> 512 (Core: 64x64)
        self.layer2 = DCTLinear(512, 512, k_in=64, k_out=64)
        self.bn2 = nn.BatchNorm1d(512)
        
        # Layer 3: 512 -> 10 (Core: 64x10)
        self.layer3 = DCTLinear(512, 10, k_in=64, k_out=10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        x = self.layer3(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Attention Neuron: MNIST Compact Demo (V63) ---")
    print(f"Device: {device}")
    
    BATCH_SIZE = 512
    EPOCHS = 10
    LR = 0.005
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)
    
    model = CompactDCTNet().to(device)
    
    # Parameter counting
    dense_params = (784*512 + 512) + (512*512 + 512) + (512*10 + 10)
    actual_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Learnable Parameters: {actual_params:,}")
    print(f"Equivalent Dense Parameters: {dense_params:,}")
    print(f"Global Network Compression: {dense_params/actual_params:.1f}x")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    t0 = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
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
        if acc > best_acc: best_acc = acc
        print(f"Epoch {epoch:2d} | Acc: {acc:.4f} | Best: {best_acc:.4f}")

    print(f"Final Best Accuracy: {best_acc:.4f} | Total Time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
