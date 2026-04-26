import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import os

# --- Fast Walsh-Hadamard Transform (Differentiable & Vectorized) ---

def fwht(x):
    """
    Computes the Fast Walsh-Hadamard Transform of a batch of vectors.
    Input x: (B, C, N) where N must be a power of 2.
    """
    B, C, N = x.shape
    h = 1
    while h < N:
        x = x.view(B, C, N // (2 * h), 2, h)
        a = x[:, :, :, 0, :]
        b = x[:, :, :, 1, :]
        x = torch.stack([a + b, a - b], dim=3)
        h *= 2
    return x.view(B, C, N)

def ifwht(x):
    N = x.shape[-1]
    return fwht(x) / N

# --- Walsh MLP Model ---

class WalshMNISTNet(nn.Module):
    """
    V36: The Walsh MLP for MNIST.
    Uses FWHT to project MNIST (padded to 32x32) into frequency space,
    then modulates the signal using our Attention Neuron mechanism.
    Very few parameters, very high efficiency.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.N = 1024 # 32x32 padded size
        
        # Attention Neurons in Walsh Space
        # We modulate the spectrum before transforming back or processing further.
        # This acts as a global 'frequency mask' learned by the network.
        self.delta_m = nn.Parameter(torch.randn(hidden_dim, self.N) * 0.01)
        self.delta_a = nn.Parameter(torch.zeros(hidden_dim, self.N))
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc_final = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        # 1. Padding from 28x28 to 32x32 (1024)
        x = F.pad(x, (2, 2, 2, 2))
        B = x.size(0)
        x_flat = x.view(B, 1, self.N)
        
        # 2. To Walsh Frequency Domain
        # Since it's an MLP, we treat the image as a single signal vector
        x_walsh = fwht(x_flat) # (B, 1, 1024)
        
        # 3. Walsh Modulation (Attention Mechanism)
        # We broadcast Walsh signal against hidden_dim filters
        # x_walsh: (B, 1, 1024)
        # delta_m: (hidden_dim, 1024)
        x_filtered = x_walsh * (1.0 + self.delta_m) + self.delta_a
        
        # 4. Return to Spatial Domain or stay in Walsh?
        # Let's return to spatial domain to extract localized features
        x_spatial = ifwht(x_filtered) # (B, hidden_dim, 1024)
        
        # 5. Global Energy per filter
        # We take the mean activation (or max) as the feature
        x_features = x_spatial.mean(dim=2) # (B, hidden_dim)
        
        # 6. Classify
        x = self.bn1(x_features)
        return self.fc_final(F.relu(x))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V36 'THE WALSH-MNIST MLP' on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 10
    LR = 0.01
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)
    
    model = WalshMNISTNet(hidden_dim=128).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
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
        print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch:2d}/{EPOCHS} | Acc: {acc:.4f} | Best: {best_acc:.4f} | Time: {t_now-t0:.1f}s")

if __name__ == "__main__":
    main()
