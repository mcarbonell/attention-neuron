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

class AN_MLP_2L(nn.Module):
    def __init__(self, h, rank=2):
        super().__init__()
        self.l1 = DualNeuronLayer(784, h, rank)
        self.l2 = DualNeuronLayer(h, 10, rank)
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.l1(x))
        return self.l2(x)

def p(model):
    return sum(x.numel() for x in model.parameters() if x.requires_grad)

def eval_model(model, epochs=10, seed=42):
    torch.manual_seed(seed)
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

print("Matched Parameter Count Comparison")
print("=" * 60)

configs = [
    (10, 79, 2, "7.9K"),
    (10, 109, 4, "7.8K (rank4)"),
    (14, 94, 2, "11K"),
    (14, 128, 4, "11K (rank4)"),
]

print(f"{'Target':<10} {'MLP h':<8} {'MLP p':<8} {'AN h':<6} {'AN r':<6} {'AN p':<8} {'MLP acc':<10} {'AN acc':<10} {'Diff':<8}")
print("-" * 60)

for mlp_h, an_h, an_rank, label in configs:
    mlp = NarrowMLP(mlp_h)
    an = AN_MLP_2L(an_h, an_rank)
    mlp_p = p(mlp)
    an_p = p(an)
    mlp_acc = eval_model(mlp)
    an_acc = eval_model(an)
    diff = an_acc - mlp_acc
    print(f"{label:<10} {mlp_h:<8} {mlp_p:<8} {an_h:<6} {an_rank:<6} {an_p:<8} {mlp_acc:.4f}     {an_acc:.4f}     {diff:+.4f}")

print("\n" + "=" * 60)
print("Note: AN uses rank-2 factorization, same architecture depth as MLP")
print("MLP has 2 layers, AN has 2 layers (both 784->h->10)")