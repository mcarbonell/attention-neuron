import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# --- FWHT Core ---
def fwht(x):
    B, N = x.shape
    h = 1
    while h < N:
        x = x.view(B, N // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        h *= 2
    return x.view(B, N)

def ifwht(x):
    return fwht(x) / x.shape[-1]

# --- The "Micro-Walsh" Net (Slightly larger than Nano) ---

class MicroWalshNet(nn.Module):
    """
    V40b: The Micro-Walsh Net.
    We double the parameter count from V40 by:
    1. Increasing the number of frequency bands from 128 to 256.
    2. Keeping the 3 layers of FWHT modulation.
    3. Keeping the heavy spatial pooling (4x4 final resolution).
    Total parameters should be ~1700. Let's see if this pushes us >95%.
    """
    def __init__(self, num_bands=256):
        super().__init__()
        self.N = 1024 # Padded 32x32
        self.num_bands = num_bands
        self.band_size = self.N // num_bands # 4 frequencies per band
        
        # Layer 1 (512 params)
        self.m1 = nn.Parameter(torch.randn(num_bands) * 0.01)
        self.a1 = nn.Parameter(torch.zeros(num_bands))
        
        # Layer 2 (512 params)
        self.m2 = nn.Parameter(torch.randn(num_bands) * 0.01)
        self.a2 = nn.Parameter(torch.zeros(num_bands))
        
        # Layer 3 (512 params)
        self.m3 = nn.Parameter(torch.randn(num_bands) * 0.01)
        self.a3 = nn.Parameter(torch.zeros(num_bands))
        
        # Classifier (16 inputs -> 10 classes = 170 params)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        # 1. Pad to 32x32
        x = F.pad(x, (2, 2, 2, 2))
        B = x.size(0)
        h = x.view(B, self.N)
        
        # Helper for Walsh-Dense Layer
        def walsh_layer(h_in, m, a):
            w = fwht(h_in)
            # Broadcast bands to 1024
            m_exp = m.unsqueeze(1).expand(-1, self.band_size).reshape(self.N)
            a_exp = a.unsqueeze(1).expand(-1, self.band_size).reshape(self.N)
            # Modulate in frequency domain
            w = w * (1.0 + m_exp) + a_exp
            # Return to spatial for non-linearity
            h_out = ifwht(w)
            return F.relu(h_out) 
            
        # 2. Forward through 3 Walsh layers
        h = walsh_layer(h, self.m1, self.a1)
        h = walsh_layer(h, self.m2, self.a2)
        h = walsh_layer(h, self.m3, self.a3)
        
        # 3. Heavy spatial pooling (32x32 -> 4x4)
        spatial = h.view(B, 1, 32, 32)
        pooled = F.avg_pool2d(spatial, 8) # (B, 1, 4, 4)
        
        # 4. Classify
        out = self.fc(pooled.view(B, 16))
        return out

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training V40b 'MICRO-WALSH NET' on: {device}")
    
    BATCH_SIZE = 256
    EPOCHS = 15
    LR = 0.01 # Slightly lower LR since we have more parameters
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)
    
    model = MicroWalshNet(num_bands=256).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("========================================")
    print(f"🔥 TRAINABLE PARAMETERS: {params} 🔥")
    print("========================================")
    
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
        
    print(f"\n🚀 TOTAL TRAINING TIME: {time.time() - t_start:.1f}s 🚀")

if __name__ == "__main__":
    main()