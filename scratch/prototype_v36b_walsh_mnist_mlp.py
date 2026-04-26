import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time

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

# --- Walsh MLP Model (V36b Fix) ---

class WalshMNISTNetB(nn.Module):
    """
    V36b: The Walsh MLP for MNIST (Spatial Fix).
    Fixes the catastrophic 'Global Average Pooling' issue of V36.
    We project to Walsh space, modulate, project back, and use a standard 
    linear classifier over the flattened spatial domain, proving Zero-Weight 
    attention works when spatial structure is preserved.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.N = 1024 # 32x32 padded size
        
        # Attention Neurons in Walsh Space
        self.delta_m = nn.Parameter(torch.randn(hidden_dim, self.N) * 0.01)
        self.delta_a = nn.Parameter(torch.zeros(hidden_dim, self.N))
        
        self.bn1 = nn.BatchNorm1d(hidden_dim * self.N) # Batch norm over the entire spatial map
        self.fc_final = nn.Linear(hidden_dim * self.N, 10) # Classify based on the filtered spatial maps

    def forward(self, x):
        # 1. Padding from 28x28 to 32x32 (1024)
        x = F.pad(x, (2, 2, 2, 2))
        B = x.size(0)
        x_flat = x.view(B, 1, self.N)
        
        # 2. To Walsh Frequency Domain
        x_walsh = fwht(x_flat) # (B, 1, 1024)
        
        # 3. Walsh Modulation (Attention Mechanism)
        x_filtered = x_walsh * (1.0 + self.delta_m) + self.delta_a # (B, hidden_dim, 1024)
        
        # 4. Return to Spatial Domain
        x_spatial = ifwht(x_filtered) # (B, hidden_dim, 1024)
        
        # 5. Flatten the filtered spatial maps instead of pooling!
        x_features = x_spatial.view(B, -1) # (B, hidden_dim * 1024)
        
        # 6. Classify
        x = self.bn1(x_features)
        return self.fc_final(F.relu(x))

def main():
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V36b 'THE WALSH-MNIST (SPATIAL FIX)' on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 10
    LR = 0.01
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)
    
    # 64 hidden dims to keep parameter count extremely efficient
    model = WalshMNISTNetB(hidden_dim=64).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=len(train_loader)*EPOCHS)
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
            scheduler.step()
            
            # Fast logging for Epoch 1
            if epoch == 1 and batch_idx % 100 == 0:
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
