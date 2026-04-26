import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# --- FWHT Core ---
def fwht(x):
    original_shape = x.shape
    if len(original_shape) == 1: x = x.unsqueeze(0)
    B, C, N = x.shape
    h = 1
    while h < N:
        x = x.view(B, C, N // (2 * h), 2, h)
        a = x[:, :, :, 0, :]
        b = x[:, :, :, 1, :]
        x = torch.stack([a + b, a - b], dim=3)
        h *= 2
    res = x.view(B, C, N)
    return res if len(original_shape) > 1 else res.squeeze(0)

def ifwht(x):
    N = x.shape[-1]
    return fwht(x) / N

# --- Banded Walsh Attention Model ---

class BandedWalshMNIST(nn.Module):
    """
    V39: The Banded Walsh Equalizer.
    Instead of 1024 parameters per neuron, we group frequencies into K bands.
    Each neuron learns only K multiplicative and K additive scalars.
    """
    def __init__(self, hidden_dim=64, num_bands=4):
        super().__init__()
        self.N = 1024 # 32x32
        self.num_bands = num_bands
        self.band_size = self.N // num_bands # 256 frequencies per band
        
        # Attention Equalizer: ONLY K parameters per neuron!
        # shape: (hidden_dim, num_bands)
        self.delta_m = nn.Parameter(torch.randn(hidden_dim, num_bands) * 0.01)
        self.delta_a = nn.Parameter(torch.zeros(hidden_dim, num_bands))
        
        self.bn1 = nn.BatchNorm1d(hidden_dim * self.N)
        self.fc_final = nn.Linear(hidden_dim * self.N, 10)

    def forward(self, x):
        # 1. Padding to 32x32
        x = F.pad(x, (2, 2, 2, 2))
        B = x.size(0)
        x_flat = x.view(B, 1, self.N)
        
        # 2. Transform
        x_walsh = fwht(x_flat) # (B, 1, 1024)
        
        # 3. Apply the Equalizer (Broadcasting bands to full frequency space)
        # Expand deltas from (hidden_dim, num_bands) to (hidden_dim, num_bands, band_size)
        # then reshape to (hidden_dim, 1024)
        m_expanded = self.delta_m.unsqueeze(-1).expand(-1, -1, self.band_size).contiguous().view(self.delta_m.shape[0], self.N)
        a_expanded = self.delta_a.unsqueeze(-1).expand(-1, -1, self.band_size).contiguous().view(self.delta_a.shape[0], self.N)
        
        # Apply the expanded equalizer masks
        x_filtered = x_walsh * (1.0 + m_expanded) + a_expanded
        
        # 4. Inverse Transform
        x_spatial = ifwht(x_filtered) # (B, hidden_dim, 1024)
        
        # 5. Flatten and Classify
        x_features = x_spatial.view(B, -1)
        x = self.bn1(x_features)
        return self.fc_final(F.relu(x))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V39 'BANDED WALSH EQUALIZER' (4 Channels/Neuron) on: {device}")
    
    BATCH_SIZE = 128
    EPOCHS = 10
    LR = 0.01
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)
    
    # 64 hidden dims, 4 bands -> The attention core has only 64 * 4 * 2 = 512 parameters!
    model = BandedWalshMNIST(hidden_dim=64, num_bands=4).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters (Total): {params:,}")
    print(f"Attention Core Parameters: {model.delta_m.numel() + model.delta_a.numel():,}")
    
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
            if epoch == 1 and batch_idx < 5:
                print(f"  > Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
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
