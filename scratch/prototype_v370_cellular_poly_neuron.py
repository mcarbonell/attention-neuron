import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json

# --- CONFIGURACIÓN DE DISPOSITIVO ---
device = torch.device('cpu')
try:
    import torch_directml
    device = torch_directml.device()
except ImportError:
    pass

# --- CARGA MNIST DESDE IDX CRUDOS (sin torchvision) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MNIST_DIR = os.path.join(SCRIPT_DIR, "data", "MNIST", "raw")

def _read_idx_images(path):
    with open(path, 'rb') as f:
        f.read(16)  # magic(4) + n(4) + rows(4) + cols(4)
        buf = f.read()
    data = torch.frombuffer(buf, dtype=torch.uint8).float()
    return data.view(-1, 28, 28)  # (N, 28, 28)

def _read_idx_labels(path):
    with open(path, 'rb') as f:
        f.read(8)  # magic(4) + n(4)
        buf = f.read()
    return torch.frombuffer(buf, dtype=torch.uint8).long().view(-1)

def load_mnist(n_train=2000, n_test=500):
    xt = _read_idx_images(os.path.join(MNIST_DIR, "train-images-idx3-ubyte"))[:n_train]
    yt = _read_idx_labels(os.path.join(MNIST_DIR, "train-labels-idx1-ubyte"))[:n_train]
    xe = _read_idx_images(os.path.join(MNIST_DIR, "t10k-images-idx3-ubyte"))[:n_test]
    ye = _read_idx_labels(os.path.join(MNIST_DIR, "t10k-labels-idx1-ubyte"))[:n_test]
    return (xt, yt), (xe, ye)

# --- REGLAS DE AUTÓMATA (outer-totalísticas, suaves/diferenciables) ---
# Cada regla: f(center c in [0,1], suma de 8 vecinos s in [0,8]) -> [0,1]
# Las versiones discretas clásicas son: Life=B3/S23, Seeds=B2/S-, HighLife=B36/S23,
# Replicator=B1357/S1357, Day&Night=B3678/S34678.

def _band(x, lo, hi, sharp=8.0):
    return torch.sigmoid(sharp * (x - lo)) * torch.sigmoid(sharp * (hi - x))

def _life(c, s):
    birth = _band(s, 2.5, 3.5)
    surv  = _band(s, 1.5, 3.5)
    return (1.0 - c) * birth + c * surv

def _seeds(c, s):
    birth = _band(s, 1.5, 2.5)
    return (1.0 - c) * birth  # sin supervivencia

def _highlife(c, s):
    birth = _band(s, 2.5, 3.5) + _band(s, 5.5, 6.5)
    surv  = _band(s, 1.5, 3.5)
    return (1.0 - c) * torch.clamp(birth, 0, 1) + c * surv

def _replicator(c, s):
    birth = _band(s, 0.5, 1.5) + _band(s, 2.5, 3.5) + _band(s, 4.5, 5.5) + _band(s, 6.5, 7.5)
    surv  = birth
    return (1.0 - c) * torch.clamp(birth, 0, 1) + c * surv

def _daynight(c, s):
    birth = _band(s, 2.5, 3.5) + _band(s, 5.5, 6.5) + _band(s, 6.5, 7.5) + _band(s, 7.5, 8.5)
    surv  = _band(s, 2.5, 4.5) + _band(s, 5.5, 6.5) + _band(s, 7.5, 8.5)
    return (1.0 - c) * torch.clamp(birth, 0, 1) + c * surv

def _largestatics(c, s):  # regla "aburrida" de control (muy estable)
    surv = _band(s, 3.5, 4.5)
    return c * surv

RULES = [
    ("Life",       _life),
    ("Seeds",      _seeds),
    ("HighLife",   _highlife),
    ("Replicator", _replicator),
    ("DayNight",   _daynight),
    ("Stable",     _largestatics),
]
K = len(RULES)

# --- CAPA NEURONA-AUTÓMATA POLIMÓRFICA ---
class CellularPolyLayer(nn.Module):
    """
    Procesa una grilla 2D. Cada celda mira su centro + 8 vecinos (3x3) y aplica
    una REGLA POLIMÓRFICA: interpola (lerp ponderado por alpha) entre K reglas
    candidatas. Auto-recurrente: itera el resultado sobre sí mismo `iters` veces.
    """
    def __init__(self, k=K):
        super().__init__()
        # alpha aprendible: mezcla las K reglas (softmax -> coeficientes de lerp)
        self.alpha = nn.Parameter(torch.ones(k))

    def forward(self, grid, iters=4):
        a = torch.softmax(self.alpha, dim=0)  # (K,)
        for _ in range(iters):
            # suma de los 8 vecinos vía convolución 3x3 (incluye centro, restamos)
            nb_all = F.conv2d(grid.unsqueeze(1), torch.ones(1, 1, 3, 3).to(grid.device),
                              padding=1).squeeze(1)
            s = nb_all - grid  # suma de 8 vecinos en [0,8]
            outs = []
            for _, fn in RULES:
                outs.append(fn(grid, s))
            outs = torch.stack(outs, dim=0)  # (K, B, H, W)
            grid = (a.view(-1, 1, 1, 1) * outs).sum(dim=0)
        return grid

class CellularPolyNet(nn.Module):
    def __init__(self, iters=2):
        super().__init__()
        self.ca = CellularPolyLayer()
        self.iters = iters
        # Head recibe estadísticas del grid CA + tinta original (señal residual)
        self.head = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 10)
        )

    def forward(self, x):
        # x: (B, 1, 28, 28) en [0,1]
        x0 = x.squeeze(1)                       # (B, 28, 28)
        g = self.ca(x0, iters=self.iters)        # (B, 28, 28) grid CA
        b = x0.size(0)
        mean_ca = g.reshape(b, -1).mean(1, keepdim=True)
        std_ca  = g.reshape(b, -1).std(1, keepdim=True)
        max_ca  = g.reshape(b, -1).max(1, keepdim=True)[0]
        mean_in = x0.reshape(b, -1).mean(1, keepdim=True)  # densidad de tinta
        feat = torch.cat([mean_ca, std_ca, max_ca, mean_in], dim=1)
        return self.head(feat)

# --- PRUEBA DIRECTA: ¿aprende el alpha a REPRODUCIR una regla objetivo? ---
def rule_recovery_probe(target_name="Life", n=300, size=12, iters=2, epochs=400):
    target_fn = dict(RULES)[target_name]
    layer = CellularPolyLayer().to(device)
    opt = torch.optim.Adam(layer.parameters(), lr=0.05)
    # grids binarios aleatorias
    torch.manual_seed(0)
    grids = (torch.rand(n, size, size) > 0.6).float().to(device)
    with torch.no_grad():
        nb_all = F.conv2d(grids.unsqueeze(1), torch.ones(1, 1, 3, 3).to(device),
                          padding=1).squeeze(1)
        s = nb_all - grids
        target = target_fn(grids, s)
    for _ in range(epochs):
        opt.zero_grad()
        pred = layer(grids, iters=iters)
        loss = F.mse_loss(pred, target)
        loss.backward()
        opt.step()
    a = torch.softmax(layer.alpha, dim=0).detach().cpu().tolist()
    names = [r[0] for r in RULES]
    return {"target": target_name, "alpha": dict(zip(names, [round(v, 4) for v in a])),
            "dominant": names[int(torch.argmax(layer.alpha))],
            "final_mse": round(loss.item(), 5)}

# --- ENGINE ---
def to_grid(images):
    # (N, 784) -> (N, 28, 28) binarizado suave
    g = images.view(-1, 28, 28) / 255.0
    return g

def main():
    (xt, yt), (xe, ye) = load_mnist()
    xt, yt, xe, ye = xt.to(device), yt.to(device), xe.to(device), ye.to(device)
    xt_g, xe_g = to_grid(xt), to_grid(xe)

    model = CellularPolyNet(iters=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    crit = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[v370] K={K} reglas | params={n_params} | device={device}")

    for epoch in range(6):
        model.train()
        opt.zero_grad()
        out = model(xt_g.unsqueeze(1))
        loss = crit(out, yt)
        loss.backward()
        opt.step()
        # eval parcial
        model.eval()
        with torch.no_grad():
            acc = (model(xe_g.unsqueeze(1)).argmax(1) == ye).float().mean().item()
        print(f"  epoch {epoch} | loss {loss.item():.4f} | val_acc {acc:.3f}")

    with torch.no_grad():
        final_acc = (model(xe_g.unsqueeze(1)).argmax(1) == ye).float().mean().item()

    a = torch.softmax(model.ca.alpha, dim=0).detach().cpu().tolist()
    recovery = rule_recovery_probe(target_name="Life")
    findings = {
        "id": "v370_cellular_poly_neuron",
        "k_rules": K,
        "rule_names": [r[0] for r in RULES],
        "params": n_params,
        "mnist": {
            "alpha_final": dict(zip([r[0] for r in RULES], [round(v, 4) for v in a])),
            "alpha_dominant": [r[0] for r in RULES][int(torch.argmax(model.ca.alpha))],
            "val_acc": round(final_acc, 4),
        },
        "rule_recovery_probe": recovery,
    }
    os.makedirs(os.path.join(SCRIPT_DIR, "..", "docs"), exist_ok=True)
    out_path = os.path.join(SCRIPT_DIR, "..", "docs", "v370_findings.json")
    with open(out_path, "w") as f:
        json.dump(findings, f, indent=2)
    print("[v370] findings ->", json.dumps(findings, indent=2))

if __name__ == "__main__":
    main()
