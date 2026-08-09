"""
analyze_v304_log.py
===================
Parses v304_log.txt (Tiny Language Modeling benchmark log) and produces:
  - Console summary tables for Validation Loss and Perplexity (PPL).
  - Bar charts and line plots comparing Complex vs Real vs Softmax MHA on text LM.

Usage:
    python scratch/analyze_v304_log.py
"""

import re, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = "v304_log.txt"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# Regex patterns
SWEEP_DK_RE = re.compile(r"=== SWEEP d_k=(\d+)")
EPOCH_RE    = re.compile(r"\[([\w\s]+)\s+\|\s*lr=([\d.]+)\s+\|\s*[\d,]+p\]\s*Epoch\s+(\d+)/(\d+)\s*\|\s*TrainLoss\s*=\s*([\d.]+)\s*\|\s*TrainPPL\s*=\s*([\d.]+)")
RESULT_RE   = re.compile(r"\*\*\* RESULT: \[([\w\s]+)\]\s*d_k=(\d+)\s*->\s*ValLoss:\s*([\d.]+)\s*\|\s*ValPPL:\s*([\d.]+)\s*\(lr=([\d.]+)\)")

results = {}

if not os.path.exists(LOG_PATH):
    print(f"Error: {LOG_PATH} not found!")
    exit(1)

with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        m = SWEEP_DK_RE.search(line)
        if m:
            current_dk = int(m.group(1))
            continue

        m = RESULT_RE.search(line)
        if m:
            model = m.group(1).strip()
            dk = int(m.group(2))
            val_loss = float(m.group(3))
            val_ppl = float(m.group(4))
            lr = float(m.group(5))
            results[(dk, model)] = (val_loss, val_ppl, lr)

# ----------------------------------------------------------------------
# 1. Summary Report
# ----------------------------------------------------------------------
report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))

log("=" * 80)
log("RESUMEN DEL BENCHMARK V304 TINY LANGUAGE MODELING (v304_log.txt)")
log("=" * 80)

all_dks = sorted(list({k[0] for k in results}))
all_models = sorted(list({k[1] for k in results}))

log(f"Sweeps d_k encontrados: {all_dks}")
log(f"Modelos evaluados:      {all_models}")
log(f"Total celdas procesadas: {len(results)}")

for dk in all_dks:
    log("\n" + "=" * 80)
    log(f"RESULTADOS — PERPLEXITY (PPL) Y VAL LOSS [d_k = {dk}]")
    log("=" * 80)
    
    header = f"  {'Modelo':40s} | {'Val Loss':>12s} | {'Val PPL':>12s} | {'Best LR':>8s}"
    log(header)
    log("  " + "-" * len(header))
    
    for model in all_models:
        key = (dk, model)
        if key in results:
            val_loss, val_ppl, lr = results[key]
            log(f"  {model:40s} | {val_loss:12.4f} | {val_ppl:12.2f} | {lr:8.4f}")

with open(os.path.join(OUT_DIR, "v304_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log(f"\n[OK] Reporte escrito en {os.path.join(OUT_DIR, 'v304_summary.txt')}")

# ----------------------------------------------------------------------
# 2. Bar Chart: PPL per Model
# ----------------------------------------------------------------------
for dk in all_dks:
    models_present = [m for m in all_models if (dk, m) in results]
    ppl_vals = [results[(dk, m)][1] for m in models_present]
    
    if models_present:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(models_present, ppl_vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(models_present)])
        
        ax.set_title(f"Validation Perplexity (PPL) — d_k={dk} (Tiny Shakespeare)", fontsize=11)
        ax.set_ylabel("Validation PPL (lower is better)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=15, ha="right", fontsize=9)
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

        fig.tight_layout()
        out_path = os.path.join(OUT_DIR, f"v304_val_ppl_dk{dk}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log(f"[OK] Grafica PPL guardada: {out_path}")

log("\nAnalisis v304 finalizado con exito!")
