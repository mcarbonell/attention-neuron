import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import time

class NarrowMLP(nn.Module):
    def __init__(self, hidden_size=10):
        super().__init__()
        self.layer1 = nn.Linear(784, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Baseline MLP angosto (Adam) en: {device}")

    HIDDEN = 10
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

    model = NarrowMLP(hidden_size=HIDDEN).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {total_params}")
    print(f"Hidden units: {HIDDEN} | Architecture: 784 -> {HIDDEN} -> 10")

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
        print(f"Epoch {epoch:2d} | Test Acc: {acc:.4f} | Time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()