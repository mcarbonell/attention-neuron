import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

class NarrowMLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.l1 = nn.Linear(784, h)
        self.l2 = nn.Linear(h, 10)
    def forward(self, x):
        return self.l2(torch.relu(self.l1(x.view(-1, 784))))

def run(hidden, epochs=10, seed=42):
    torch.manual_seed(seed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=512, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1024)

    model = NarrowMLP(hidden)
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
    return correct / 10000

def main():
    print("MLP Sweep Benchmark (single seed, 10 epochs)")
    print("=" * 60)

    configs = [
        (6, "v2 target"),
        (10, "v6b target"),
        (16, "~16K"),
        (32, "~32K"),
    ]

    results = []
    for hidden, label in configs:
        t0 = time.time()
        acc = run(hidden, epochs=10)
        t = time.time() - t0
        params = hidden * 784 + hidden + hidden * 10 + 10
        print(f"{label:<12} hidden={hidden:<3} params={params:<6} acc={acc:.4f} ({t:.0f}s)")
        results.append((label, hidden, params, acc))

    print("\n" + "=" * 60)
    print("Comparison: MLP vs Attention Neuron")
    print("=" * 60)
    print(f"{'Config':<12} {'Params':<8} {'MLP':<8} {'AN':<8} {'Diff':<8}")
    print("-" * 60)

    an = {"v2 target": (1566, 0.7618), "v6b target": (7794, 0.9453)}

    for label, hidden, params, mlp_acc in results:
        if label in an:
            an_params, an_acc = an[label]
            diff = an_acc - mlp_acc
            print(f"{label:<12} {params:<8} {mlp_acc:.4f}   {an_acc:.4f}   {diff:+.4f}")

if __name__ == "__main__":
    main()