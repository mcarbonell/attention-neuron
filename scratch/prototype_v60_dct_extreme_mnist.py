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

# Reuse DCT matrix from v59 logic
def get_dct_matrix(N, device='cpu'):
    dct_mat = torch.zeros(N, N, device=device)
    for i in range(N):
        for j in range(N):
            if i == 0:
                dct_mat[i, j] = math.sqrt(1/N)
            else:
                dct_mat[i, j] = math.sqrt(2/N) * math.cos((math.pi * i * (2*j + 1)) / (2*N))
    return dct_mat

class DCTAttentionNet(nn.Module):
    def __init__(self, hidden_dim=512, k_size=4, device='cpu'):
        super().__init__()
        self.N = 28
        self.K = k_size
        self.hidden_dim = hidden_dim
        
        self.register_buffer('D', get_dct_matrix(self.N, device=device))
        
        # Only 16 parameters per neuron! (K=4)
        self.dct_weights = nn.Parameter(torch.randn(hidden_dim, self.K, self.K) * 0.01)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc_final = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, self.N, self.N)
        
        # DCT Transform
        x_dct = torch.matmul(self.D, x)
        x_dct = torch.matmul(x_dct, self.D.t())
        
        # Extreme low-frequency crop (4x4)
        x_low = x_dct[:, :self.K, :self.K]
        
        # Activation via Einstein summation
        x_features = torch.einsum('bij,hij->bh', x_low, self.dct_weights)
        x_features = x_features + self.bias
        
        x = self.bn1(x_features)
        x = F.relu(x)
        return self.fc_final(x)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V60: EXTREME DCT ATTENTION (K=4) ---")
    
    HIDDEN_DIM = 512
    K_SIZE = 4
    BATCH_SIZE = 128
    EPOCHS = 15
    LR = 0.005
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)

    model = DCTAttentionNet(hidden_dim=HIDDEN_DIM, k_size=K_SIZE, device=device).to(device)
    
    weight_params = model.dct_weights.numel()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    compression = (HIDDEN_DIM * 784) / weight_params
    
    print(f"Compression: {compression:.1f}x | Weight Params: {weight_params:,} | Total: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
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

    print(f"Final Best Accuracy: {best_acc:.4f} | Time: {time.time()-t0:.1f}s")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/v60_dct_extreme_k4.pth")

if __name__ == "__main__":
    train()
