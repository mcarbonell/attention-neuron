"""
analyze_v302_log.py
===================
Parses v302_log.txt (Multi-hop MQAR benchmark log) and produces:
  - Console summary tables grouped by d_k, hops, and chains.
  - Heatmaps for best accuracy per d_k scale.
  - Line plots comparing Complex vs Real vs Softmax MHA across hops and chains.

Usage:
    python scratch/analyze_v302_log.py
"""

import re, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = "v302_log.txt"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# Regex patterns
SWEEP_DK_RE = re.compile(r"=== SWEEP d_k=(\d+)")
HOPS_RE     = re.compile(r"--- HOPS = (\d+) ---")
DATA_GEN_RE = re.compile(r">>> Generating data: (\d+) chains x (\d+) hops = (\d+) pairs, L=(\d+)")
EPOCH_RE    = re.compile(r"\[([\w\s]+)\s+\|\s*lr=([\d.]+)\s+\|\s*[\d,]+p\]\s*Epoch\s+(\d+)/(\d+)\s*\|\s*AvgLoss\s*=\s*([\d.]+)\s*\|\s*EpTime\s*=\s*([\d.]+)s")
RESULT_RE   = re.compile(r"\*\*\* RESULT: \[([\w\s]+)\]\s*d_k=(\d+)\s*hops=(\d+)\s*chains=(\d+)\s*->\s*Acc:\s*([\d.]+)%\s*\(lr=([\d.]+)\)")

# Storage
# loss_data[(dk, hops, chains, model, lr)] = {epoch: loss}
loss_data = defaultdict(dict)
# results[(dk, hops, chains, model)] = (best_acc, best_lr)
results = {}

current_dk = None
current_hops = None
current_chains = None

if not os.path.exists(LOG_PATH):
    print(f"Error: {LOG_PATH} not found!")
    exit(1)

with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        m = SWEEP_DK_RE.search(line)
        if m:
            current_dk = int(m.group(1))
            continue
            
        m = HOPS_RE.search(line)
        if m:
            current_hops = int(m.group(1))
            continue

        m = DATA_GEN_RE.search(line)
        if m:
            current_chains = int(m.group(1))
            continue

        m = EPOCH_RE.search(line)
        if m and current_dk and current_hops and current_chains:
            model = m.group(1).strip()
            lr = float(m.group(2))
            ep = int(m.group(3))
            loss_val = float(m.group(5))
            loss_data[(current_dk, current_hops, current_chains, model, lr)][ep] = loss_val
            continue

        m = RESULT_RE.search(line)
        if m:
            model = m.group(1).strip()
            dk = int(m.group(2))
            hops = int(m.group(3))
            chains = int(m.group(4))
            acc = float(m.group(5))
            lr = float(m.group(6))
            results[(dk, hops, chains, model)] = (acc, lr)

# ----------------------------------------------------------------------
# 1. Summary Report in Console & File
# ----------------------------------------------------------------------
report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))

log("=" * 80)
log("RESUMEN DEL BENCHMARK V302 (v302_log.txt)")
log("=" * 80)

all_dks = sorted(list({k[0] for k in results}))
all_hops = sorted(list({k[1] for k in results}))
all_chains = sorted(list({k[2] for k in results}))
all_models = sorted(list({k[3] for k in results}))

log(f"Sweeps d_k encontrados: {all_dks}")
log(f"Hops evaluados:         {all_hops}")
log(f"Cadenas evaluadas:      {all_chains}")
log(f"Modelos evaluados:      {all_models}")
log(f"Total celdas procesadas: {len(results)}")

for dk in all_dks:
    log("\n" + "=" * 80)
    log(f"TABLA DE RESULTADOS — BEST ACC FINAL (%) [d_k = {dk}]")
    log("=" * 80)
    
    cells = sorted(list({(k[1], k[2]) for k in results if k[0] == dk}))
    
    header = f"{'Modelo':36s}" + "".join(f" | h{h}_c{c:<3d}" for h, c in cells)
    log(header)
    log("-" * len(header))
    
    for model in all_models:
        row = f"{model:36s}"
        for h, c in cells:
            key = (dk, h, c, model)
            if key in results:
                acc, lr = results[key]
                row += f" | {acc:6.2f}%({lr})"
            else:
                row += f" | {'--':>11s}"
        log(row)

with open(os.path.join(OUT_DIR, "v302_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log(f"\n[OK] Reporte escrito en {os.path.join(OUT_DIR, 'v302_summary.txt')}")

# ----------------------------------------------------------------------
# 2. Heatmaps per d_k
# ----------------------------------------------------------------------
for dk in all_dks:
    cells = sorted(list({(k[1], k[2]) for k in results if k[0] == dk}))
    if not cells:
        continue
    
    cell_labels = [f"h={h}\nc={c}" for h, c in cells]
    data = np.full((len(all_models), len(cells)), np.nan)
    
    for i, model in enumerate(all_models):
        for j, (h, c) in enumerate(cells):
            key = (dk, h, c, model)
            if key in results:
                data[i, j] = results[key][0]

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(cells)), max(3.5, 0.8 * len(all_models))))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cell_labels, fontsize=8)
    ax.set_yticks(range(len(all_models)))
    ax.set_yticklabels(all_models, fontsize=8)
    ax.set_title(f"Best Acc Final (%) — d_k={dk} (Multi-hop MQAR)", fontsize=11, pad=10)
    
    for i in range(len(all_models)):
        for j in range(len(cells)):
            if not np.isnan(data[i, j]):
                val = data[i, j]
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=7.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6))

    cbar = fig.colorbar(im, ax=ax, label="Best Acc (%)")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"v302_bestacc_dk{dk}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"[OK] Heatmap guardado: {out_path}")

# ----------------------------------------------------------------------
# 3. Line plot: Hops vs Acc
# ----------------------------------------------------------------------
for dk in all_dks:
    for c in [16, 32]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        has_data = False
        for model in all_models:
            acc_list = []
            hops_present = []
            for h in [1, 2, 3]:
                key = (dk, h, c, model)
                if key in results:
                    acc_list.append(results[key][0])
                    hops_present.append(h)
            if hops_present:
                has_data = True
                ax.plot(hops_present, acc_list, marker="o", linewidth=2, label=model)
        
        if has_data:
            ax.set_title(f"Accuracy vs Hops (d_k={dk}, Chains={c})", fontsize=11)
            ax.set_xlabel("Number of Hops", fontsize=10)
            ax.set_ylabel("Accuracy (%)", fontsize=10)
            ax.set_xticks([1, 2, 3])
            ax.set_ylim(-5, 105)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            out_path = os.path.join(OUT_DIR, f"v302_scaling_hops_dk{dk}_c{c}.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            log(f"[OK] Grafica hops scaling guardada: {out_path}")

log("\nAnalisis v302 finalizado con exito!")
