"""
analyze_v303_log.py
===================
Parses v303_log.txt (Overwrite MQAR benchmark log) and produces:
  - Console summary tables grouped by d_k, overwrite ratio, and n_keys.
  - Heatmaps for best accuracy per d_k scale.
  - Line plots comparing Complex vs Real vs Softmax MHA across overwrite ratios (0%, 30%, 60%).

Usage:
    python scratch/analyze_v303_log.py
"""

import re, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = "v303_log.txt"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# Regex patterns
SWEEP_DK_RE = re.compile(r"=== SWEEP d_k=(\d+)")
OW_RATIO_RE = re.compile(r"--- OVERWRITE RATIO = ([\d.]+) \((\d+)% keys rewritten\) ---")
DATA_GEN_RE = re.compile(r">>> Generating data: (\d+) unique keys, (\d+) overwrites = (\d+) pairs, L=(\d+)")
EPOCH_RE    = re.compile(r"\[([\w\s]+)\s+\|\s*lr=([\d.]+)\s+\|\s*[\d,]+p\]\s*Epoch\s+(\d+)/(\d+)\s*\|\s*AvgLoss\s*=\s*([\d.]+)\s*\|\s*EpTime\s*=\s*([\d.]+)s")
RESULT_RE   = re.compile(r"\*\*\* RESULT: \[([\w\s]+)\]\s*d_k=(\d+)\s*ow_ratio=([\d.]+)\s*n_keys=(\d+)\s*->\s*Acc:\s*([\d.]+)%\s*\(lr=([\d.]+)\)")

# Storage
loss_data = defaultdict(dict)
results = {}

if not os.path.exists(LOG_PATH):
    print(f"Error: {LOG_PATH} not found!")
    exit(1)

current_dk = None
current_ratio = None
current_keys = None

with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        m = SWEEP_DK_RE.search(line)
        if m:
            current_dk = int(m.group(1))
            continue
            
        m = OW_RATIO_RE.search(line)
        if m:
            current_ratio = float(m.group(1))
            continue

        m = DATA_GEN_RE.search(line)
        if m:
            current_keys = int(m.group(1))
            continue

        m = EPOCH_RE.search(line)
        if m and current_dk is not None and current_ratio is not None and current_keys is not None:
            model = m.group(1).strip()
            lr = float(m.group(2))
            ep = int(m.group(3))
            loss_val = float(m.group(5))
            loss_data[(current_dk, current_ratio, current_keys, model, lr)][ep] = loss_val
            continue

        m = RESULT_RE.search(line)
        if m:
            model = m.group(1).strip()
            dk = int(m.group(2))
            ow_ratio = float(m.group(3))
            n_keys = int(m.group(4))
            acc = float(m.group(5))
            lr = float(m.group(6))
            results[(dk, ow_ratio, n_keys, model)] = (acc, lr)

# ----------------------------------------------------------------------
# 1. Summary Report
# ----------------------------------------------------------------------
report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))

log("=" * 80)
log("RESUMEN DEL BENCHMARK V303 OVERWRITE MQAR (v303_log.txt)")
log("=" * 80)

all_dks = sorted(list({k[0] for k in results}))
all_ratios = sorted(list({k[1] for k in results}))
all_keys = sorted(list({k[2] for k in results}))
all_models = sorted(list({k[3] for k in results}))

log(f"Sweeps d_k encontrados:    {all_dks}")
log(f"Overwrite Ratios evaluados: {all_ratios}")
log(f"Unique Keys evaluadas:     {all_keys}")
log(f"Modelos evaluados:         {all_models}")
log(f"Total celdas procesadas:    {len(results)}")

for dk in all_dks:
    log("\n" + "=" * 80)
    log(f"TABLA DE RESULTADOS — BEST ACC FINAL (%) [d_k = {dk}]")
    log("=" * 80)
    
    cells = sorted(list({(k[1], k[2]) for k in results if k[0] == dk}))
    header = f"{'Modelo':36s}" + "".join(f" | ow{int(r*100):02d}_k{k:<2d}" for r, k in cells)
    log(header)
    log("-" * len(header))
    
    for model in all_models:
        row = f"{model:36s}"
        for r, k in cells:
            key = (dk, r, k, model)
            if key in results:
                acc, lr = results[key]
                row += f" | {acc:6.2f}%({lr})"
            else:
                row += f" | {'--':>11s}"
        log(row)

with open(os.path.join(OUT_DIR, "v303_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log(f"\n[OK] Reporte escrito en {os.path.join(OUT_DIR, 'v303_summary.txt')}")

# ----------------------------------------------------------------------
# 2. Line Plot: Accuracy vs Overwrite Ratio
# ----------------------------------------------------------------------
for dk in all_dks:
    for n_k in all_keys:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        has_data = False
        for model in all_models:
            acc_list = []
            ratios_present = []
            for r in all_ratios:
                key = (dk, r, n_k, model)
                if key in results:
                    acc_list.append(results[key][0])
                    ratios_present.append(r * 100)
            if ratios_present:
                has_data = True
                ax.plot(ratios_present, acc_list, marker="o", linewidth=2, label=model)
        
        if has_data:
            ax.set_title(f"Accuracy vs Overwrite Ratio % (d_k={dk}, Keys={n_k})", fontsize=11)
            ax.set_xlabel("Overwrite Ratio (% of keys re-written)", fontsize=10)
            ax.set_ylabel("Accuracy (%)", fontsize=10)
            ax.set_ylim(-5, 105)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            out_path = os.path.join(OUT_DIR, f"v303_overwrite_scaling_dk{dk}_k{n_k}.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            log(f"[OK] Grafica overwrite scaling guardada: {out_path}")

log("\nAnalisis v303 finalizado con exito!")
