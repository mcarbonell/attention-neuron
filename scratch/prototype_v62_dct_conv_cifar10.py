import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time
import os

# DCT basis for convolution kernels (8x8)
def get_dct_matrix(N, device='cpu'):
    dct_mat = torch.zeros(N, N, device=device)
    for i in range(N):
        for j in range(N):
            if i == 0:
                dct_mat[i, j] = math.sqrt(1/N)
            else:
                dct_mat[i, j] = math.sqrt(2/N) * math.cos((math.pi * i * (2*j + 1)) / (2*N))
    return dct_mat

class DCTConvLayer(nn.Module):
    """
    Learns convolution kernels in the DCT domain.
    Instead of out_channels * in_channels * 8 * 8 weights,
    it learns out_channels * in_channels * K * K coefficients.
    """
    def __init__(self, in_channels, out_channels, kernel_size=8, k_size=4, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.k_size = k_size
        self.stride = stride
        self.padding = padding
        
        self.register_buffer('D', get_dct_matrix(kernel_size))
        
        # Learnable DCT coefficients for the kernels
        # Each kernel is 8x8, but only KxK coefficients are learned
        self.dct_coeffs = nn.Parameter(torch.randn(out_channels, in_channels, k_size, k_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # 1. Synthesize kernels: W = D^T * C_padded * D
        # C_padded is (out, in, kernel_size, kernel_size)
        C_padded = torch.zeros(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, device=x.device)
        C_padded[:, :, :self.k_size, :self.k_size] = self.dct_coeffs
        
        # We can do this efficiently for all kernels at once
        # Using matrix multiplication logic: W = D^T @ C @ D
        # Transpose D for the inverse transform
        DT = self.D.t()
        # Matmul on last two dims
        kernels = torch.matmul(DT, torch.matmul(C_padded, self.D))
        
        # 2. Standard convolution with synthesized kernels
        return F.conv2d(x, kernels, self.bias, stride=self.stride, padding=self.padding)

class DCTConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First layer: DCT Conv (8x8 kernels, only 4x4 coeffs = 4x compression)
        self.conv1 = DCTConvLayer(3, 32, kernel_size=8, k_size=4, padding=4)
        self.pool1 = nn.MaxPool2d(2)
        
        # Second layer: Standard Conv or another DCT Conv? Let's use DCT Conv again
        self.conv2 = DCTConvLayer(32, 64, kernel_size=5, k_size=3, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc_final = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc_final(x)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- V62: DCT CONVOLUTIONAL NET (CIFAR-10) ---")
    
    BATCH_SIZE = 128
    EPOCHS = 20
    LR = 0.001
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_loader = DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets.CIFAR10('./data', train=False, transform=transform), batch_size=1000)

    model = DCTConvNet().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
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

if __name__ == "__main__":
    train()
