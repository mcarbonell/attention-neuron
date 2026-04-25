import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time

class DualNeuronLayerRank2(nn.Module):
    def __init__(self, in_features, out_features, rank=2):
        super().__init__()
        std = math.sqrt(2.0 / in_features)
        self.register_buffer('w_init', torch.randn(out_features, in_features) * std)

        self.delta_in_m = nn.Parameter(torch.randn(out_features, rank) * 0.1 + 1.0)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_features) * 0.1 + 1.0)
        self.delta_in_a = nn.Parameter(torch.randn(out_features, rank) * 0.1)
        self.delta_out_a = nn.Parameter(torch.randn(rank, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        w_evolved = self.w_init * w_m + w_a
        return torch.matmul(x, w_evolved.t()) + self.bias

class AN_MLP(nn.Module):
    def __init__(self, hidden1, hidden2, rank=2):
        super().__init__()
        self.l1 = DualNeuronLayerRank2(784, hidden1, rank)
        self.l2 = DualNeuronLayerRank2(hidden1, hidden2, rank)
        self.l3 = DualNeuronLayerRank2(hidden2, 10, rank)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_an(hidden1, hidden2, rank=2, epochs=10, seed=42):
    torch.manual_seed(seed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1024)

    model = AN_MLP(hidden1, hidden2, rank)
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            opt.zero_grad()
            crit(model(data), target).backward()
            opt.step()

    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            correct += model(data).argmax(1).eq(target).sum().item()
    return correct / 10000, count_params(model)

def main():
    print("Attention Neuron Sweep (matching MLP params)")
    print("=" * 60)

    configs = [
        (384, 128, 2, "3-layer ~16K"),
        (512, 256, 2, "3-layer ~32K"),
        (256, 128, 4, "3-layer rank4 ~16K"),
        (512, 256, 4, "3-layer rank4 ~32K"),
        (512, 128, 2, "3-layer wider ~20K"),
    ]

    results = []
    for h1, h2, rank, label in configs:
        t0 = time.time()
        acc, params = run_an(h1, h2, rank, epochs=10)
        t = time.time() - t0
        print(f"{label:<18} h=({h1},{h2}) r={rank} params={params:<6} acc={acc:.4f} ({t:.0f}s)")
        results.append((label, params, acc))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for label, params, acc in results:
        print(f"{label:<18} params={params:<6} acc={acc:.4f}")

if __name__ == "__main__":
    main()