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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MNIST_DIR = os.path.join(SCRIPT_DIR, "data", "MNIST", "raw")

def _read_idx_images(path):
    with open(path, 'rb') as f:
        f.read(16)
        buf = f.read()
    return torch.frombuffer(buf, dtype=torch.uint8).float().view(-1, 28, 28)

def load_one_digit(digit=2, n=5):
    """Toma n ejemplos del dígito como targets binarios 28x28."""
    xt = _read_idx_images(os.path.join(MNIST_DIR, "train-images-idx3-ubyte"))
    yt = torch.frombuffer(open(os.path.join(MNIST_DIR, "train-labels-idx1-ubyte"), 'rb').read()[8:],
                         dtype=torch.uint8).long().view(-1)
    idx = (yt == digit).nonzero().flatten()[:n]
    imgs = xt[idx] / 255.0
    return (imgs > 0.4).float()  # (n,28,28) binario

def make_disk(radius=9):
    """Target sólido (disco) — mucho más fácil y visual que un dígito fino."""
    y, x = torch.meshgrid(torch.arange(28), torch.arange(28), indexing='ij')
    cy = cx = 13.5
    return ( (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2 ).float().unsqueeze(0)  # (1,28,28)

# --- NEURAL CA ---
class NCA(nn.Module):
    """Estado por célula = vector C canales. Regla = conv local 3x3 + 1x1.
    Actualización residual, solo en células 'vivas' (máscara de percepción)."""
    def __init__(self, C=4, hidden=16):
        super().__init__()
        self.C = C
        self.conv1 = nn.Conv2d(C, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, C, 1)
        # init en ~0 => arranca casi identidad (evita colapso)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def alive_mask(self, s):
        # célula viva si ella o algún vecino 3x3 tiene canal visible > 0.1
        vis = torch.relu(s[:, 0:1] - 0.1)
        return F.max_pool2d(vis, 3, padding=1, stride=1) > 0

    def step(self, s):
        h = torch.relu(self.conv1(s))
        delta = self.conv2(h)
        alive = self.alive_mask(s)
        s = s + delta
        s = s * alive.float()
        return s

    def forward(self, s0, steps=24):
        s = s0
        for _ in range(steps):
            s = self.step(s)
        return s

# --- AUXILIARES ---
def seed_state(target, B, C=4):
    s = torch.zeros(B, C, 28, 28, device=device)
    for i in range(B):
        s[i, 0, 14, 14] = 1.0  # semilla en el centro
    return s

def damage(s, hole=8):
    s = s.clone()
    b, c, h, w = s.shape
    for i in range(b):
        y = torch.randint(4, h - hole - 4, (1,)).item()
        x = torch.randint(4, w - hole - 4, (1,)).item()
        s[i, :, y:y + hole, x:x + hole] = 0.0
    return s

# --- ENTRENAMIENTO: crecer -> (a veces dañar) -> curar, pérdida sobre trayectoria ---
def train(nca, target, epochs=200, B=16, T=30):
    opt = torch.optim.Adam(nca.parameters(), lr=0.002)
    tgt = target.to(device)  # (1,28,28)
    for ep in range(epochs):
        opt.zero_grad()
        loss = 0.0
        n_steps = 0
        for _ in range(B):
            s = seed_state(tgt, 1, nca.C)
            # crecer un número aleatorio de pasos
            grow = torch.randint(5, T - 10, (1,)).item()
            for t in range(T):
                s = nca.step(s)
                if t == grow and torch.rand(1).item() < 0.7:
                    s = damage(s, hole=torch.randint(6, 11, (1,)).item())
                if t >= T - 12:
                    loss = loss + F.mse_loss(s[:, 0], tgt)
                    n_steps += 1
        loss = loss / max(n_steps, 1)
        loss.backward()
        opt.step()
        if ep % 40 == 0:
            print(f"  epoch {ep:03d} | loss {loss.item():.4f}")
    return nca

# --- EVALUACIÓN ---
def evaluate(nca, target, T=24):
    tgt = target.to(device)
    # 1) CRECIMIENTO desde semilla
    s = seed_state(tgt, 1, nca.C)
    s = nca(s, steps=T)
    grow_mse = F.mse_loss(s[:, 0], tgt).item()
    # 2) REPARACIÓN: dañar el crecido y dejar curar
    s_dam = damage(s, hole=10)
    s_rep = nca(s_dam, steps=T)
    rep_mse = F.mse_loss(s_rep[:, 0], tgt).item()
    # fracción de píxeles recuperados en la zona dañada (IoU-ish)
    with torch.no_grad():
        vis0 = (s_dam[:, 0] > 0.3).float()
        visR = (s_rep[:, 0] > 0.3).float()
        tgt_b = (tgt > 0.3).float()
        hole_mask = (vis0 < 0.5) & (tgt_b > 0.5)
        recovered = (visR[hole_mask] > 0.5).float().mean().item() if hole_mask.sum() > 0 else 1.0
    return {"grow_mse": round(grow_mse, 4),
            "repair_mse": round(rep_mse, 4),
            "hole_recovered_frac": round(recovered, 4)}

def main():
    targets = make_disk(radius=9).to(device)  # (1,28,28) disco sólido
    print(f"[v373] target shape {tuple(targets.shape)} | device {device}")
    nca = NCA(C=4, hidden=16).to(device)
    n_params = sum(p.numel() for p in nca.parameters())
    print(f"[v373] NCA params = {n_params}")
    train(nca, targets, epochs=200, B=16, T=30)
    metrics = evaluate(nca, targets, T=30)
    findings = {
        "id": "v373_nca_morphogenesis",
        "description": "Neural CA: crece un patron desde semilla y lo auto-repara (regla local aprendida).",
        "params": n_params,
        "metrics": metrics,
    }
    os.makedirs(os.path.join(SCRIPT_DIR, "..", "docs"), exist_ok=True)
    with open(os.path.join(SCRIPT_DIR, "..", "docs", "v373_findings.json"), "w") as f:
        json.dump(findings, f, indent=2)
    print("[v373] findings ->", json.dumps(findings, indent=2))

if __name__ == "__main__":
    main()
