import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

class NarrowMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(784, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        return self.layer2(torch.relu(self.layer1(x)))

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_eval(hidden_size, seed=42, epochs=10):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=256, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1024, shuffle=False)

    model = NarrowMLP(hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            criterion(model(data), target).backward()
            optimizer.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            correct += output.argmax(1).eq(target).sum().item()
    return correct / 10000

def main():
    print("MLP Sweep (2 seeds, 10 epochs)")
    print("-" * 50)

    configs = [
        (8, 1566, "v2 target"),
        (13, 7794, "v6b target"),
        (20, 12000, "~12K"),
        (32, 20000, "~20K"),
    ]

    results = []
    for hidden, target, label in configs:
        accs = []
        for seed in [42, 43]:
            acc = train_eval(hidden, seed)
            print(f"{label} seed{seed}: {acc:.4f}")
            accs.append(acc)
        results.append((label, hidden, target, sum(accs)/2))

    print("\nSUMMARY")
    print(f"{'Config':<15} {'Hidden':<8} {'Params':<10} {'Mean Acc':<10}")
    print("-" * 50)
    for label, hidden, target, acc in results:
        params = hidden * 784 + hidden + hidden * 10 + 10
        print(f"{label:<15} {hidden:<8} {params:<10} {acc:.4f}")

if __name__ == "__main__":
    main()