import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import os
import json

# DCT basis for 32x32
def get_dct_matrix(N, device='cpu'):
    dct_mat = torch.zeros(N, N, device=device)
    for i in range(N):
        for j in range(N):
            if i == 0:
                dct_mat[i, j] = math.sqrt(1/N)
            else:
                dct_mat[i, j] = math.sqrt(2/N) * math.cos((math.pi * i * (2*j + 1)) / (2*N))
    return dct_mat

class DCTCifarNet(nn.Module):
    """
    V61: DCT Attention for CIFAR-10.
    Handles 3 channels and 32x32 images.
    """
    def __init__(self, hidden_dim=512, k_size=8, device='cpu'):
        super().__init__()
        self.N = 32
        self.K = k_size
        self.hidden_dim = hidden_dim
        
        self.register_buffer('D', get_dct_matrix(self.N, device=device))
        
        # We learn DCT coefficients for each channel (R, G, B)
        # Total weight params: hidden_dim * 3 * K * K
        self.dct_weights = nn.Parameter(torch.randn(hidden_dim, 3, self.K, self.K) * 0.01)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc_final = nn.Linear(256, 10)

    def forward(self, x):
        # x: (B, 3, 32, 32)
        B = x.size(0)
        
        # 1. Transform each channel to DCT Domain
        # We can do this efficiently with matmul
        # x: (B, 3, 32, 32)
        x_dct = torch.matmul(self.D, x)           # (B, 3, 32, 32)
        x_dct = torch.matmul(x_dct, self.D.t())   # (B, 3, 32, 32)
        
        # 2. Extract low-frequency KxK quadrant
        x_low = x_dct[:, :, :self.K, :self.K]    # (B, 3, K, K)
        
        # 3. Activation (sum over channels and frequencies)
        # b: batch, c: channel, i: k_row, j: k_col, h: hidden_dim
        x_features = torch.einsum('bcij,hcij->bh', x_low, self.dct_weights)
        x_features = x_features + self.bias
        
        # 4. Readout
        x = self.bn1(x_features)
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        return self.fc_final(x)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V61: DCT ATTENTION (CIFAR-10) ---")
    
    HIDDEN_DIM = 1024 # More capacity for CIFAR
    K_SIZE = 8
    BATCH_SIZE = 128
    EPOCHS = 30
    LR = 0.001
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_loader = DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transform_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.CIFAR10('./data', train=False, transform=transform_test), batch_size=1000)

    model = DCTCifarNet(hidden_dim=HIDDEN_DIM, k_size=K_SIZE, device=device).to(device)
    
    weight_params = model.dct_weights.numel()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Weight Params (Layer 1): {weight_params:,} (Compression vs Dense: {(1024*3072)/weight_params:.1f}x)")
    print(f"Total Parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
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
        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Acc: {acc:.4f} | Best: {best_acc:.4f}")

    print(f"Final Best Accuracy: {best_acc:.4f} | Total Time: {time.time()-t_start:.1f}s")
    
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/v61_dct_cifar10.pth")

if __name__ == "__main__":
    train()
