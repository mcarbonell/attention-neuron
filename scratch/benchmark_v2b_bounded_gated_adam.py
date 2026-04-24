import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time

class BoundedGatedAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=2, mask_prob=0.5, alpha=1.0):
        super().__init__()
        std = math.sqrt(2.0 / in_features)
        self.register_buffer('w_init', torch.randn(out_features, in_features) * std)
        
        self.alpha = alpha
        # Inicializamos S pequeño para que tanh(S) empiece cerca de 0, y M cerca de 1.
        self.delta_in_s = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.delta_out_s = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        
        self.delta_in_a = nn.Parameter(torch.zeros(out_features, rank))
        self.delta_out_a = nn.Parameter(torch.zeros(rank, in_features))
        self.theta_bias = nn.Parameter(torch.zeros(out_features))
        self.mask_prob = mask_prob

    def forward(self, x):
        # S es la pre-activación del gating
        s = torch.matmul(self.delta_in_s, self.delta_out_s)
        # M está acotado en el rango [1 - alpha, 1 + alpha]
        w_m = 1.0 + self.alpha * torch.tanh(s)
        
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        
        if self.training and self.mask_prob < 1.0:
            mask = torch.bernoulli(torch.full(self.w_init.shape, self.mask_prob, device=self.w_init.device))
            w_evolved = torch.where(mask > 0, self.w_init * w_m + w_a, self.w_init)
        else:
            m_eff = 1.0 + self.mask_prob * (w_m - 1.0)
            a_eff = self.mask_prob * w_a
            w_evolved = self.w_init * m_eff + a_eff
            
        return torch.matmul(x, w_evolved.t()) + torch.sin(self.theta_bias)

class BoundedGatedAttentionMLP(nn.Module):
    def __init__(self, mask_prob=0.5):
        super().__init__()
        self.layer1 = BoundedGatedAttentionLayer(784, 512, rank=2, mask_prob=mask_prob)
        self.layer2 = BoundedGatedAttentionLayer(512, 10, rank=2, mask_prob=mask_prob)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking V2b (Bounded Gated Attention - Tanh) with ADAM on: {device}")

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

    model = BoundedGatedAttentionMLP(mask_prob=0.5).to(device)
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
