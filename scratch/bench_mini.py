import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math

class MLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.l1 = nn.Linear(784, h)
        self.l2 = nn.Linear(h, 10)
    def forward(self, x):
        return self.l2(torch.relu(self.l1(x.view(-1, 784))))

class ANLayer(nn.Module):
    def __init__(self, i, o, r):
        super().__init__()
        self.w = torch.randn(o, i) * math.sqrt(2/i)
        self.register_buffer('wi', self.w)
        self.din_m = nn.Parameter(torch.randn(o, r) + 1)
        self.dout_m = nn.Parameter(torch.randn(r, i) + 1)
        self.din_a = nn.Parameter(torch.randn(o, r) * 0.1)
        self.dout_a = nn.Parameter(torch.randn(r, i) * 0.1)
        self.bias = nn.Parameter(torch.zeros(o))
    def forward(self, x):
        wm = self.din_m @ self.dout_m
        wa = self.din_a @ self.dout_a
        w = self.wi * wm + wa
        return x @ w.t() + self.bias

class AN(nn.Module):
    def __init__(self, h, r):
        super().__init__()
        self.l1 = ANLayer(784, h, r)
        self.l2 = ANLayer(h, 10, r)
    def forward(self, x):
        x = x.view(-1, 784)
        return self.l2(torch.relu(self.l1(x)))

def p(m): return sum(x.numel() for x in m.parameters() if x.requires_grad)

def test(model, epochs=10):
    t = transforms.ToTensor()
    tl = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=t), batch_size=512, shuffle=True)
    vl = DataLoader(datasets.MNIST('./data', train=False, transform=t), batch_size=1024)
    opt = optim.Adam(model.parameters(), lr=0.001)
    for e in range(epochs):
        model.train()
        for d, t in tl:
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(d), t).backward()
            opt.step()
    model.eval()
    c = 0
    with torch.no_grad():
        for d, t in vl:
            c += model(d).argmax(1).eq(t).sum().item()
    return c / 10000

print("Matched Param Comparison")
print(f"{'Config':<12} {'MLP-p':<8} {'AN-p':<8} {'MLP-a':<8} {'AN-a':<8} {'Diff':<8}")
print("-" * 52)

tests = [
    (10, 79, 2),
    (14, 94, 2),
    (10, 109, 4),
    (14, 128, 4),
]

for mlp_h, an_h, an_r in tests:
    mlp = MLP(mlp_h)
    an = AN(an_h, an_r)
    ma = test(mlp)
    aa = test(an)
    print(f"{an_h}({an_r})       {p(mlp):<8} {p(an):<8} {ma:.4f}     {aa:.4f}     {aa-ma:+.4f}")