import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time

class ResidualAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=2, mask_prob=0.5):
        super().__init__()
        std = math.sqrt(2.0 / in_features)
        self.register_buffer('w_init', torch.randn(out_features, in_features) * std)
        
        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        
        self.delta_in_a = nn.Parameter(torch.zeros(out_features, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_features))
        self.theta_bias = nn.Parameter(torch.zeros(out_features))
        self.mask_prob = mask_prob

    def forward(self, x):
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        
        if self.training and self.mask_prob < 1.0:
            mask = torch.bernoulli(torch.full(self.w_init.shape, self.mask_prob, device=self.w_init.device))
            w_evolved = torch.where(mask > 0, self.w_init + self.w_init * w_m + w_a, self.w_init)
        else:
            m_eff = self.mask_prob * w_m
            a_eff = self.mask_prob * w_a
            w_evolved = self.w_init + self.w_init * m_eff + a_eff
            
        return torch.matmul(x, w_evolved.t()) + torch.sin(self.theta_bias)

class HybridAttentionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, rank=2, mask_prob=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Sustrato aleatorio congelado para el kernel convolucional
        std = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.register_buffer('w_init', torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std)
        
        # Modulación de Canal (Factorizada)
        self.delta_in_m = nn.Parameter(torch.randn(out_channels, rank) * 0.01)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_channels) * 0.01)
        
        # Modulación Espacial (Directa, al ser muy pequeña, ej: 3x3 = 9 params)
        self.m_spatial = nn.Parameter(torch.randn(1, 1, kernel_size, kernel_size) * 0.01)
        
        # Corrección Aditiva de Canal (Factorizada)
        self.delta_in_a = nn.Parameter(torch.zeros(out_channels, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_channels))
        
        self.theta_bias = nn.Parameter(torch.zeros(out_channels))
        self.mask_prob = mask_prob

    def forward(self, x):
        # M_chan shape: (out_channels, in_channels, 1, 1)
        m_chan = torch.matmul(self.delta_in_m, self.delta_out_m).view(self.out_channels, self.in_channels, 1, 1)
        # A_chan shape: (out_channels, in_channels, 1, 1)
        a_chan = torch.matmul(self.delta_in_a, self.delta_out_a).view(self.out_channels, self.in_channels, 1, 1)
        
        # Modulación Híbrida: Aproximación residual combinando canal y espacio
        # (1 + m_chan) * (1 + m_spatial) - 1 ≈ m_chan + m_spatial
        w_m = m_chan + self.m_spatial 
        
        if self.training and self.mask_prob < 1.0:
            mask = torch.bernoulli(torch.full(self.w_init.shape, self.mask_prob, device=self.w_init.device))
            w_evolved = torch.where(mask > 0, self.w_init + self.w_init * w_m + a_chan, self.w_init)
        else:
            m_eff = self.mask_prob * w_m
            a_eff = self.mask_prob * a_chan
            w_evolved = self.w_init + self.w_init * m_eff + a_eff
            
        return F.conv2d(x, w_evolved, padding=self.padding) + torch.sin(self.theta_bias).view(1, -1, 1, 1)

class HybridAttentionCNN(nn.Module):
    def __init__(self, mask_prob=0.5):
        super().__init__()
        # Extractor de características convolucional puro
        self.conv1 = HybridAttentionConv2d(1, 16, kernel_size=3, padding=1, rank=2, mask_prob=mask_prob)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = HybridAttentionConv2d(16, 32, kernel_size=3, padding=1, rank=2, mask_prob=mask_prob)
        self.pool2 = nn.MaxPool2d(2)
        
        # Clasificador lineal
        self.fc = ResidualAttentionLayer(32 * 7 * 7, 10, rank=2, mask_prob=mask_prob)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking V12 (Hybrid Attention CNN) with ADAM on: {device}")

    BATCH_SIZE = 256
    EPOCHS = 10
    LR = 0.001

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    model = HybridAttentionCNN(mask_prob=0.5).to(device)
    
    # Contar parámetros entrenables para comparar eficiencia
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

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

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        acc = correct / 10000
        print(f"Epoch {epoch} | Test Acc: {acc:.4f} | Time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
