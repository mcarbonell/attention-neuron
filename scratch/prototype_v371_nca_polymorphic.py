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

def load_mnist(n_train=2000, n_test=500):
    xt = _read_idx_images(os.path.join(MNIST_DIR, "train-images-idx3-ubyte"))[:n_train]
    yt = _read_idx_labels(os.path.join(MNIST_DIR, "train-labels-idx1-ubyte"))[:n_train]
    xe = _read_idx_images(os.path.join(MNIST_DIR, "t10k-images-idx3-ubyte"))[:n_test]
    ye = _read_idx_labels(os.path.join(MNIST_DIR, "t10k-labels-idx1-ubyte"))[:n_test]
    return (xt, yt), (xe, ye)

# --- REGLA APRENDIDA (Neural Cellular Automata diferenciable) ---
class LearnableRule(nn.Module):
    """Regla de transición paramétrica: 9 entradas (vecindad 3x3) -> delta de estado.
    Inicializada ~0 => el CA arranca como identidad (preserva la señal de entrada)."""
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        # init en casi-cero: actualización residual nula al inicio
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, patches):
        # patches: (B, 9, H, W) -> (B*H*W, 9)
        B, _, H, W = patches.shape
        flat = patches.permute(0, 2, 3, 1).reshape(-1, 9)
        delta = self.net(flat).reshape(B, H, W)
        return torch.tanh(delta)  # delta acotado en [-1, 1]

class CellularPolyLayerNCA(nn.Module):
    """
    Capa neurona-autómata polimórfica con reglas APRENDIDAS (estilo Neural CA).
    Cada celda mira su vecindad 3x3; K reglas (MLP) producen un delta que se
    mezcla con alpha (softmax) y se aplica como actualización residual.
    Auto-recurrente: itera su salida `iters` veces.
    """
    def __init__(self, K=3, hidden=16):
        super().__init__()
        self.rules = nn.ModuleList([LearnableRule(hidden) for _ in range(K)])
        self.alpha = nn.Parameter(torch.ones(K))

    def forward(self, grid, iters=2):
        a = torch.softmax(self.alpha, dim=0)
        B, H, W = grid.shape
        for _ in range(iters):
            patches = F.unfold(grid.unsqueeze(1), 3, padding=1).view(B, 9, H, W)
            deltas = torch.stack([r(patches) for r in self.rules], dim=0)  # (K,B,H,W)
            delta = (a.view(-1, 1, 1, 1) * deltas).sum(dim=0)
            grid = torch.clamp(grid + delta, 0.0, 1.0)
        return grid

class NCAClassifier(nn.Module):
    def __init__(self, K=3, iters=2):
        super().__init__()
        self.ca = CellularPolyLayerNCA(K=K)
        self.iters = iters
        # Readout LOCAL (convolucional) en vez de pooling global
        self.head = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(8 * 4 * 4, 10)
        )

    def forward(self, x):
        x0 = x.squeeze(1)                          # (B,28,28)
        g = self.ca(x0, iters=self.iters)          # (B,28,28)
        return self.head(g.unsqueeze(1))

# --- PRUEBA DIRECTA: recuperar una regla Life discreta ---
def life_step(g):
    nb = F.conv2d(g.unsqueeze(1), torch.ones(1, 1, 3, 3).to(g.device),
                  padding=1).squeeze(1) - g
    return (((g == 1) & ((nb == 2) | (nb == 3))) |
            ((g == 0) & (nb == 3))).float()

def rule_recovery_probe(iters=1, epochs=1200):
    layer = CellularPolyLayerNCA(K=3).to(device)
    opt = torch.optim.Adam(layer.parameters(), lr=0.02)
    torch.manual_seed(0)
    grids = (torch.rand(300, 12, 12) > 0.6).float().to(device)
    target = grids
    for _ in range(iters):
        target = life_step(target)
    for _ in range(epochs):
        opt.zero_grad()
        pred = layer(grids, iters=iters)
        loss = F.mse_loss(pred, target)
        loss.backward()
        opt.step()
    a = torch.softmax(layer.alpha, dim=0).detach().cpu().tolist()
    return {"target": "Life(discreta)", "K": 3,
            "alpha": [round(v, 4) for v in a],
            "final_mse": round(loss.item(), 5)}

# --- ENGINE ---
def to_grid(images):
    return images.view(-1, 28, 28) / 255.0

def main():
    (xt, yt), (xe, ye) = load_mnist()
    xt, yt, xe, ye = xt.to(device), yt.to(device), xe.to(device), ye.to(device)
    xt_g, xe_g = to_grid(xt), to_grid(xe)

    model = NCAClassifier(K=3, iters=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    crit = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[v371] K=3 reglas APRENDIDAS | params={n_params} | device={device}")

    for epoch in range(12):
        model.train()
        opt.zero_grad()
        out = model(xt_g.unsqueeze(1))
        loss = crit(out, yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(xe_g.unsqueeze(1)).argmax(1) == ye).float().mean().item()
        print(f"  epoch {epoch:02d} | loss {loss.item():.4f} | val_acc {acc:.3f}")

    with torch.no_grad():
        final_acc = (model(xe_g.unsqueeze(1)).argmax(1) == ye).float().mean().item()
    a = torch.softmax(model.ca.alpha, dim=0).detach().cpu().tolist()
    recovery = rule_recovery_probe()

    findings = {
        "id": "v371_nca_polymorphic",
        "description": "Regla de CA APRENDIDA (MLP 9->h->1) + alpha polimórfico + readout local convolucional",
        "K": 3,
        "params": n_params,
        "mnist_val_acc": round(final_acc, 4),
        "alpha_final": [round(v, 4) for v in a],
        "rule_recovery_probe": recovery,
    }
    os.makedirs(os.path.join(SCRIPT_DIR, "..", "docs"), exist_ok=True)
    with open(os.path.join(SCRIPT_DIR, "..", "docs", "v371_findings.json"), "w") as f:
        json.dump(findings, f, indent=2)
    print("[v371] findings ->", json.dumps(findings, indent=2))

if __name__ == "__main__":
    main()
