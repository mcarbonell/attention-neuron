import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time

class NarrowMLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.l1 = nn.Linear(784, h)
        self.l2 = nn.Linear(h, 10)
    def forward(self, x):
        return self.l2(torch.relu(self.l1(x.view(-1, 784))))

class DualNeuronLayer(nn.Module):
    def __init__(self, in_f, out_f, rank=2):
        super().__init__()
        std = math.sqrt(2.0 / in_f)
        self.register_buffer('w_init', torch.randn(out_f, in_f) * std)
        self.delta_in_m = nn.Parameter(torch.randn(out_f, rank) * 0.1 + 1.0)
        self.delta_out_m = nn.Parameter(torch.randn(rank, in_f) * 0.1 + 1.0)
        self.delta_in_a = nn.Parameter(torch.randn(out_f, rank) * 0.1)
        self.delta_out_a = nn.Parameter(torch.randn(rank, in_f) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_f))
    def forward(self, x):
        w_m = torch.matmul(self.delta_in_m, self.delta_out_m)
        w_a = torch.matmul(self.delta_in_a, self.delta_out_a)
        return torch.matmul(x, (self.w_init * w_m + w_a).t()) + self.bias

class AN_MLP(nn.Module):
    def __init__(self, h1, h2, rank=2):
        super().__init__()
        self.l1 = DualNeuronLayer(784, h1, rank)
        self.l2 = DualNeuronLayer(h1, h2, rank)
        self.l3 = DualNeuronLayer(h2, 10, rank)
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)

def params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_eval(model, epochs=10, seed=42):
    torch.manual_seed(seed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1024)
    opt = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(data), target).backward()
            opt.step()
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            correct += model(data).argmax(1).eq(target).sum().item()
    return correct / 10000

def main():
    print("Sweep: MLP (2-layer) vs AN (3-layer) at similar parameter counts")
    print("=" * 70)

    sweep_points = [
        (6, None, 128, 2, "tiny"),
        (10, None, 192, 2, "small"),
        (16, None, 256, 2, "medium"),
        (24, None, 384, 2, "large"),
        (40, None, 512, 2, "xl"),
    ]

    print(f"{'Config':<10} {'MLP params':<12} {'AN params':<12} {'MLP acc':<10} {'AN acc':<10} {'Diff':<8}")
    print("-" * 70)

    results = []
    for mlp_h, _, an_h1, an_rank, label in sweep_points:
        mlp = NarrowMLP(mlp_h)
        mlp_p = params(mlp)
        mlp_acc = train_eval(mlp)

        an = AN_MLP(an_h1, an_h1//2, an_rank)
        an_p = params(an)
        an_acc = train_eval(an)

        diff = an_acc - mlp_acc
        print(f"{label:<10} {mlp_p:<12} {an_p:<12} {mlp_acc:.4f}     {an_acc:.4f}     {diff:+.4f}")
        results.append((label, mlp_p, an_p, mlp_acc, an_acc, diff))

    print("\n" + "=" * 70)
    print("Key findings:")
    print("-" * 70)

    wins_an = sum(1 for r in results if r[5] > 0)
    wins_mlp = len(results) - wins_an

    print(f"AN wins: {wins_an}/{len(results)}")
    print(f"MLP wins: {wins_mlp}/{len(results)}")

    if wins_an > wins_mlp:
        print("\nAttention Neuron shows advantage in parameter efficiency.")
    elif wins_mlp > wins_an:
        print("\nMLP shows equal or better performance at matched parameters.")
    else:
        print("\nResults are mixed - more data needed.")

if __name__ == "__main__":
    main()