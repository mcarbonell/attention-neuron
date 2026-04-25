import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math

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

def p(model):
    return sum(x.numel() for x in model.parameters() if x.requires_grad)

def eval_model(model, epochs=10):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=512, shuffle=True)
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

print("Quick 2-point comparison")
print("=" * 50)

tests = [
    ("small", 10, 192, 2),
    ("medium", 16, 256, 2),
]

for label, mlp_h, an_h, an_r in tests:
    mlp = NarrowMLP(mlp_h)
    an = AN_MLP(an_h, an_h//2, an_r)
    mlp_acc = eval_model(mlp)
    an_acc = eval_model(an)
    print(f"{label}: MLP hidden={mlp_h} ({p(mlp)} params) acc={mlp_acc:.4f}")
    print(f"       AN h1={an_h} ({p(an)} params) acc={an_acc:.4f}  diff={an_acc-mlp_acc:+.4f}")
    print()