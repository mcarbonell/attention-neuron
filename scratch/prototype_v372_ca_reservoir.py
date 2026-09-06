import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json

# --- CONFIGURACIÓN DE DISPOSITIVO ---
device = torch.device('cpu')
try:
    import torch_directml
    device = torch_directml.device()
except ImportError:
    pass

# --- CARGA MNIST DESDE IDX CRUDOS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MNIST_DIR = os.path.join(SCRIPT_DIR, "data", "MNIST", "raw")

def _read_idx_images(path):
    with open(path, 'rb') as f:
        f.read(16)
        buf = f.read()
    return torch.frombuffer(buf, dtype=torch.uint8).float().view(-1, 28, 28)

def _read_idx_labels(path):
    with open(path, 'rb') as f:
        f.read(8)
        buf = f.read()
    return torch.frombuffer(buf, dtype=torch.uint8).long().view(-1)

def load_mnist(n_train=3000, n_test=1000):
    xt = _read_idx_images(os.path.join(MNIST_DIR, "train-images-idx3-ubyte"))[:n_train]
    yt = _read_idx_labels(os.path.join(MNIST_DIR, "train-labels-idx1-ubyte"))[:n_train]
    xe = _read_idx_images(os.path.join(MNIST_DIR, "t10k-images-idx3-ubyte"))[:n_test]
    ye = _read_idx_labels(os.path.join(MNIST_DIR, "t10k-labels-idx1-ubyte"))[:n_test]
    return (xt, yt), (xe, ye)

# --- REGLAS DE RESERVORIO (CONGELADAS, NO ENTRENADAS) ---
class RandomCARule:
    """Regla CA aleatoria fija: célula = sigmoid(MLP(3x3)) con pesos random congelados."""
    def __init__(self, hidden=12, sigma=1.2, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(9, hidden, generator=g) * sigma
        self.b = torch.randn(hidden, generator=g) * 0.1
        self.Wout = torch.randn(hidden, 1, generator=g) * 0.5
        self.bo = torch.zeros(1)
    def step(self, grid):
        B, H, W = grid.shape
        patches = F.unfold(grid.unsqueeze(1), 3, padding=1).view(B, 9, H, W)
        flat = patches.permute(0, 2, 3, 1)                       # (B,H,W,9)
        h = torch.tanh(flat @ self.W + self.b)                  # (B,H,W,hidden)
        return torch.sigmoid(h @ self.Wout + self.bo).squeeze(-1)

class LifeRule:
    """Life suave (outer-totalística), determinista, sin parámetros."""
    @staticmethod
    def _band(x, lo, hi, k=8.0):
        return torch.sigmoid(k * (x - lo)) * torch.sigmoid(k * (hi - x))
    def step(self, grid):
        nb = F.conv2d(grid.unsqueeze(1), torch.ones(1, 1, 3, 3).to(grid.device),
                      padding=1).squeeze(1) - grid
        birth = self._band(nb, 2.5, 3.5)
        surv = self._band(nb, 1.5, 3.5)
        return (1.0 - grid) * birth + grid * surv

class PolyBlendRule:
    """Blend polimórfico congelado de varias reglas (alpha fijo)."""
    def __init__(self, rules, alpha=None):
        self.rules = rules
        a = torch.tensor(alpha or [1.0 / len(rules)] * len(rules))
        self.alpha = a.view(-1, 1, 1, 1)
    def step(self, grid):
        nb = F.conv2d(grid.unsqueeze(1), torch.ones(1, 1, 3, 3).to(grid.device),
                      padding=1).squeeze(1) - grid
        outs = [r(grid, nb) for r in self.rules]
        return (self.alpha * torch.stack(outs, 0)).sum(0)

def life_fn(c, s):
    birth = LifeRule._band(s, 2.5, 3.5)
    surv = LifeRule._band(s, 1.5, 3.5)
    return (1.0 - c) * birth + c * surv

def seeds_fn(c, s):
    return (1.0 - c) * LifeRule._band(s, 1.5, 2.5)

def highlife_fn(c, s):
    b = LifeRule._band(s, 2.5, 3.5) + LifeRule._band(s, 5.5, 6.5)
    sv = LifeRule._band(s, 1.5, 3.5)
    return (1.0 - c) * torch.clamp(b, 0, 1) + c * sv

# --- RESERVORIO ---
def run_reservoir_last(grid0, rule, T=20):
    """Corre T pasos y devuelve SOLO la grilla final (para readout convolucional)."""
    g = grid0.clone()
    for _ in range(T):
        g = rule.step(g)
    return g

def run_reservoir(grid0, rule, T=20, K_read=10):
    """Corre T pasos (dinámica congelada). Devuelve features del readout:
    estadísticas de los últimos K_read pasos."""
    g = grid0.clone()
    feats = []
    for t in range(T):
        g = rule.step(g)
        if t >= T - K_read:
            b = g.reshape(g.size(0), -1)
            feats.append(torch.cat([b.mean(1, keepdim=True),
                                    b.std(1, keepdim=True),
                                    b.max(1, keepdim=True)[0]], dim=1))
    return torch.cat(feats, dim=1)  # (B, 3*K_read)

# --- READOUT LINEAL (lo único que se entrena) ---
def train_readout(feat_tr, y_tr, feat_te, y_te, epochs=40, lr=0.05):
    in_f = feat_tr.size(1)
    head = nn.Linear(in_f, 10).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(head(feat_tr), y_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (head(feat_te).argmax(1) == y_te).float().mean().item()
    return acc

def train_conv_readout(grid_tr, y_tr, grid_te, y_te, epochs=15, lr=0.01):
    """Readout convolucional entrenado (el CA queda CONGELADO)."""
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(4), nn.Flatten(),
        nn.Linear(8 * 4 * 4, 10)
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(grid_tr.unsqueeze(1)), y_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (model(grid_te.unsqueeze(1)).argmax(1) == y_te).float().mean().item()
    return acc

# --- ENGINE ---
def to_binary(images):
    return (images.view(-1, 28, 28) / 255.0 > 0.4).float()

def main():
    (xt, yt), (xe, ye) = load_mnist()
    xt, yt, xe, ye = xt.to(device), yt.to(device), xe.to(device), ye.to(device)
    xt_b, xe_b = to_binary(xt), to_binary(xe)

    T, K = 20, 10
    configs = {
        "identity (raw image)": None,
        "random_CA (seed0)": RandomCARule(seed=0),
        "random_CA (seed1)": RandomCARule(seed=1),
        "Life (smooth)": LifeRule(),
        "polyblend (Life+Seeds+HighLife)": PolyBlendRule(
            [life_fn, seeds_fn, highlife_fn], alpha=[0.5, 0.2, 0.3]),
    }

    results = {}
    for name, rule in configs.items():
        if rule is None:
            def feat(x):
                b = x.reshape(x.size(0), -1)
                return torch.cat([b.mean(1, keepdim=True), b.std(1, keepdim=True),
                                  b.max(1, keepdim=True)[0]], dim=1).to(device)
            def final_grid(x):
                return x.to(device)
            ftr, fte = feat(xt_b), feat(xe_b)
            gtr, gte = final_grid(xt_b), final_grid(xe_b)
        else:
            ftr = run_reservoir(xt_b, rule, T=T, K_read=K).to(device)
            fte = run_reservoir(xe_b, rule, T=T, K_read=K).to(device)
            gtr = run_reservoir_last(xt_b, rule, T=T).to(device)
            gte = run_reservoir_last(xe_b, rule, T=T).to(device)
        acc_lin = train_readout(ftr, yt, fte, ye, epochs=40, lr=0.05)
        acc_conv = train_conv_readout(gtr, yt, gte, ye, epochs=15, lr=0.01)
        results[name] = {"linear_readout": round(acc_lin, 4),
                         "conv_readout": round(acc_conv, 4)}
        print(f"  [v372] {name:32s} -> lin {acc_lin:.3f} | conv {acc_conv:.3f}")

    findings = {
        "id": "v372_ca_reservoir",
        "description": "CA como reservorio: regla CONGELADA itera T pasos; solo se entrena un readout lineal.",
        "T_steps": T, "K_readout": K,
        "n_train": 3000, "n_test": 1000,
        "results": results,
        "note": "El readout (lo unico entrenado) es lineal sobre estadisticas de los ultimos K pasos.",
    }
    os.makedirs(os.path.join(SCRIPT_DIR, "..", "docs"), exist_ok=True)
    with open(os.path.join(SCRIPT_DIR, "..", "docs", "v372_findings.json"), "w") as f:
        json.dump(findings, f, indent=2)
    print("[v372] findings ->", json.dumps(findings, indent=2))

if __name__ == "__main__":
    main()
