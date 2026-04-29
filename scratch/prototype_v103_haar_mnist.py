import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import math

def get_haar_filters(device, dtype):
    # LL, HL (Vertical edge), LH (Horizontal edge), HH (Diagonal)
    kernel = torch.tensor([
        [[[1, 1], [1, 1]]],
        [[[1, -1], [1, -1]]],
        [[[1, 1], [-1, -1]]],
        [[[1, -1], [-1, 1]]]
    ], dtype=dtype, device=device) / 2.0
    return kernel

class Haar2D(nn.Module):
    def __init__(self, levels=5):
        super().__init__()
        self.levels = levels

    def forward(self, x):
        # x shape: [B, 1, 32, 32]
        res = []
        current_ll = x
        filters = get_haar_filters(x.device, x.dtype)
        
        for i in range(self.levels):
            # conv2d with stride 2
            out = F.conv2d(current_ll, filters, stride=2)
            current_ll = out[:, 0:1] # LL for next level
            hl = out[:, 1:2]
            lh = out[:, 2:3]
            hh = out[:, 3:4]
            res.append((hl, lh, hh))
            
        return current_ll, res # DC and list of (HL, LH, HH) from coarsest to finest? 
        # Actually, let's reverse to have res[0] = Level 1 (16x16)

class HaarFeatureExtractor(nn.Module):
    def __init__(self, levels=5):
        super().__init__()
        self.haar = Haar2D(levels=levels)
        # We will pool each level to a fixed small grid
        # Level 1 (from 32x32 -> 16x16): Pool to 4x4
        # Level 2 (from 16x16 -> 8x8): Pool to 2x2
        # Level 3 (from 8x8 -> 4x4): Pool to 2x2
        # Level 4 (from 4x4 -> 2x2): Pool to 1x1
        # Level 5 (from 2x2 -> 1x1): Pool to 1x1
        self.pool_configs = [4, 2, 2, 1, 1]

    def forward(self, x):
        # Pad 28x28 to 32x32
        x = F.pad(x, (2, 2, 2, 2))
        dc, details = self.haar(x)
        
        features = [dc.view(x.shape[0], -1)] # DC is already 1x1
        
        for i, (hl, lh, hh) in enumerate(details):
            p = self.pool_configs[i]
            # Average absolute energy in the pool
            # hl, lh, hh are [B, 1, H_i, W_i]
            f_hl = F.adaptive_avg_pool2d(hl.abs(), (p, p)).view(x.shape[0], -1)
            f_lh = F.adaptive_avg_pool2d(lh.abs(), (p, p)).view(x.shape[0], -1)
            f_hh = F.adaptive_avg_pool2d(hh.abs(), (p, p)).view(x.shape[0], -1)
            features.extend([f_hl, f_lh, f_hh])
            
        return torch.cat(features, dim=1)

class HaarNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.extractor = HaarFeatureExtractor(levels=5)
        
        # Calculate input features
        # DC: 1
        # L1: 3 * (4*4) = 48
        # L2: 3 * (2*2) = 12
        # L3: 3 * (2*2) = 12
        # L4: 3 * (1*1) = 3
        # L5: 3 * (1*1) = 3
        # Total = 1 + 48 + 12 + 12 + 3 + 3 = 79
        self.input_dim = 79
        
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        feat = self.extractor(x)
        return self.classifier(feat)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        import torch_directml
        if torch_directml.is_available() and device.type == 'cpu':
            device = torch_directml.device()
    except ImportError:
        pass
        
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = HaarNet(hidden_dim=40).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parámetros entrenables: {total_params}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
        start_time = time.time()
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        acc = 100. * correct / len(test_loader.dataset)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}% | Time: {elapsed:.1f}s")

if __name__ == '__main__':
    train()
