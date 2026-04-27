import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import os

# 1D DCT Basis Generator
def get_dct_matrix_1d(N, device='cpu'):
    """Generates a 1D DCT-II basis matrix of size (N, N)."""
    dct_mat = torch.zeros(N, N, device=device)
    for i in range(N):
        for j in range(N):
            if i == 0:
                dct_mat[i, j] = math.sqrt(1/N)
            else:
                dct_mat[i, j] = math.sqrt(2/N) * math.cos((math.pi * i * (2*j + 1)) / (2*N))
    return dct_mat

class DCTLinear(nn.Module):
    """
    A Fully Connected layer where the weight matrix is synthesized from
    a much smaller learnable DCT core (K_out x K_in).
    """
    def __init__(self, in_features, out_features, k_in, k_out, device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k_in = k_in
        self.k_out = k_out
        
        # Precompute DCT bases for input and output dimensions
        self.register_buffer('D_in', get_dct_matrix_1d(in_features, device=device))
        self.register_buffer('D_out', get_dct_matrix_1d(out_features, device=device))
        
        # The only learnable weights: a tiny K_out x K_in matrix!
        self.dct_coeffs = nn.Parameter(torch.randn(k_out, k_in) * (1.0 / math.sqrt(k_in)))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Synthesize the full weight matrix W: (out_features, in_features)
        # W = D_out^T * C_padded * D_in
        
        C_padded = torch.zeros(self.out_features, self.in_features, device=x.device)
        C_padded[:self.k_out, :self.k_in] = self.dct_coeffs
        
        # D_out^T @ C_padded @ D_in
        W = torch.matmul(self.D_out.t(), torch.matmul(C_padded, self.D_in))
        
        # Standard linear projection with the synthesized dense matrix
        return F.linear(x, W, self.bias)

class AllDCT_MLP(nn.Module):
    """
    V63: An MLP where EVERY layer is compressed via DCT.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        
        # Layer 1: Image 28x28 (784) -> 512
        # Instead of 784x512 = 401,408 params
        # We use a core of 64 x 64 = 4,096 params (100x compression!)
        self.layer1 = DCTLinear(in_features=784, out_features=512, k_in=64, k_out=64, device=device)
        self.bn1 = nn.BatchNorm1d(512)
        
        # Layer 2: 512 -> 512
        # Instead of 512x512 = 262,144 params
        # We use a core of 64 x 64 = 4,096 params (64x compression!)
        self.layer2 = DCTLinear(in_features=512, out_features=512, k_in=64, k_out=64, device=device)
        self.bn2 = nn.BatchNorm1d(512)
        
        # Layer 3: 512 -> 10
        # Instead of 512x10 = 5,120 params
        # We use a core of 64 x 10 = 640 params
        self.layer3 = DCTLinear(in_features=512, out_features=10, k_in=64, k_out=10, device=device)

    def forward(self, x):
        # Flatten image to 1D sequence of length 784
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        x = self.layer3(x)
        
        return x

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V63: ALL-DCT MLP (MNIST) ---")
    
    BATCH_SIZE = 128
    EPOCHS = 15
    LR = 0.005
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)

    model = AllDCT_MLP(device=device).to(device)
    
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

    print(f"Final Best Accuracy: {best_acc:.4f} | Time: {time.time()-t0:.1f}s")
    
if __name__ == "__main__":
    train()
