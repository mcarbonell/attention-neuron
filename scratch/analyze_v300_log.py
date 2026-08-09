"""
Analiza v300_log.txt (log de entrenamiento V300, posiblemente incompleto)
y genera gráficas de:
  - Loss por época (por modelo y lr, para cada combinación d_k / pairs)
  - Best Acc Final (heatmap modelo x pairs, para cada d_k)

Uso:
    python scratch/analyze_v300_log.py
"""
import re
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = "v300_log.txt"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Parseo del log
# ----------------------------------------------------------------------
EPOCH_RE = re.compile(
    r"\[(\w+)\s+\|\s*lr=([\d.]+)\]\s*Epoch\s+(\d+)/\d+\s*Complete\s*\|\s*Loss\s*=\s*([\d.]+)"
)
BEST_ACC_RE = re.compile(
    r"\[(\w+)\s+\|\s*d_k=(\d+)\]\s*Pairs=\s*(\d+)\s*\(L=\s*\d+\)\s*->\s*Best Acc:\s*([\d.]+)%\s*\(lr=([\d.]+)\)"
)
SWEEP_RE = re.compile(r"=== SWEEP d_k = (\d+) \(d_model = \d+\) ===")
PAIRS_RE = re.compile(r">>> Pre-generating GPU Datasets for Load: (\d+) Pairs \(L=\d+\) <<<")

# loss[(d_k, pairs, modelo, lr)] = {epoch: loss}
loss = defaultdict(dict)
# best_acc[(d_k, pairs, modelo)] = (acc, lr)
best_acc = {}

current_dk = None
current_pairs = None

with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        m = SWEEP_RE.search(line)
        if m:
            current_dk = int(m.group(1))
            continue

        m = PAIRS_RE.search(line)
        if m:
            current_pairs = int(m.group(1))
            continue

        m = EPOCH_RE.search(line)
        if m and current_dk is not None and current_pairs is not None:
            modelo, lr, epoch, loss_val = m.groups()
            key = (current_dk, current_pairs, modelo, lr)
            loss[key][int(epoch)] = float(loss_val)
            continue

        m = BEST_ACC_RE.search(line)
        if m:
            modelo, dk, pairs, acc, lr = m.groups()
            best_acc[(int(dk), int(pairs), modelo)] = (float(acc), lr)

# ----------------------------------------------------------------------
# 2. Resumen en consola
# ----------------------------------------------------------------------
print("=" * 70)
print("RESUMEN DEL LOG (v300_log.txt)")
print("=" * 70)

dk_sweeps = sorted({k[0] for k in loss} | {k[0] for k in best_acc})
pairs_set = sorted({k[1] for k in loss} | {k[1] for k in best_acc})
modelos = sorted({k[2] for k in loss} | {k[2] for k in best_acc})

print(f"Sweeps d_k encontrados: {dk_sweeps}")
print(f"Configuraciones de pairs encontradas: {pairs_set}")
print(f"Modelos: {modelos}")
print()

# Tabla de Best Acc Final
print("-" * 70)
print("BEST ACC FINAL (por d_k, pairs y modelo)")
print("-" * 70)
for dk in dk_sweeps:
    print(f"\n=== d_k = {dk} ===")
    header = "Modelo".ljust(32) + "".join(f"{p:>12}" for p in pairs_set)
    print(header)
    print("-" * len(header))
    for modelo in modelos:
        row = modelo.ljust(32)
        for p in pairs_set:
            if (dk, p, modelo) in best_acc:
                acc, lr = best_acc[(dk, p, modelo)]
                row += f"{acc:>7.2f}%({lr})".rjust(12)
            else:
                row += f"{'--':>12}"
        print(row)

# ----------------------------------------------------------------------
# 3. Gráfica 1: Loss por época
# ----------------------------------------------------------------------
# Una figura por cada (d_k, pairs), con un subplot por modelo.
# Cada subplot muestra las curvas de loss para los 3 lr.
lrs = ["0.0020", "0.0040", "0.0080"]
colors = {"0.0020": "tab:blue", "0.0040": "tab:orange", "0.0080": "tab:green"}

for dk in dk_sweeps:
    for p in pairs_set:
        # ¿Hay datos de loss para esta combinación?
        keys = [k for k in loss if k[0] == dk and k[1] == p]
        if not keys:
            continue

        n_modelos = len(modelos)
        fig, axes = plt.subplots(1, n_modelos, figsize=(5 * n_modelos, 4.5), squeeze=False)
        fig.suptitle(f"Loss por época — d_k={dk}, Pairs={p} (L={p * 8})", fontsize=13)

        for ax, modelo in zip(axes[0], modelos):
            ax.set_title(modelo, fontsize=11)
            ax.set_xlabel("Época")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            for lr in lrs:
                key = (dk, p, modelo, lr)
                if key in loss and loss[key]:
                    epochs = sorted(loss[key])
                    vals = [loss[key][e] for e in epochs]
                    ax.plot(epochs, vals, marker="o", markersize=3,
                            linewidth=1.5, color=colors[lr],
                            label=f"lr={float(lr):.4f}")
            ax.legend(fontsize=8)
            ax.set_yscale("log")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(OUT_DIR, f"v300_loss_dk{dk}_pairs{p}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[OK] Gráfica loss guardada: {out}")

# ----------------------------------------------------------------------
# 4. Gráfica 2: Best Acc Final (heatmap modelo x pairs por d_k)
# ----------------------------------------------------------------------
for dk in dk_sweeps:
    # Solo pairs que tengan al menos un best_acc para este dk
    ps = [p for p in pairs_set if any((dk, p, m) in best_acc for m in modelos)]
    if not ps:
        continue

    data = np.full((len(modelos), len(ps)), np.nan)
    for i, modelo in enumerate(modelos):
        for j, p in enumerate(ps):
            if (dk, p, modelo) in best_acc:
                data[i, j] = best_acc[(dk, p, modelo)][0]

    fig, ax = plt.subplots(figsize=(max(4, 1.6 * len(ps)), max(3, 0.9 * len(modelos))))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels([f"Pairs={p}\n(L={p * 8})" for p in ps])
    ax.set_yticks(range(len(modelos)))
    ax.set_yticklabels(modelos)
    ax.set_title(f"Best Acc Final (%) — d_k={dk}", fontsize=13)

    # Anotar valores
    for i in range(len(modelos)):
        for j in range(len(ps)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}%", ha="center", va="center",
                        fontsize=9, color="black",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

    fig.colorbar(im, ax=ax, label="Best Acc (%)")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"v300_bestacc_dk{dk}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] Gráfica best acc guardada: {out}")

print("\nAnálisis completado.")