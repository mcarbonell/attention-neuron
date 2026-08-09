"""
analyze_v306_log.py
===================
Parses v306_log.txt (Tiny LM Iso-Parametric Multi-Seed log) and produces:
  - Console summary table with Mean Val Loss +- SE and Mean Val PPL +- SE.
  - Bar plot with error bars (Standard Error) comparing Complex vs Real Iso-Param vs Softmax MHA.

Usage:
    python scratch/analyze_v306_log.py
"""

import re, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


LOG_PATH = "v306_log.txt"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(LOG_PATH):
    print(f"Error: {LOG_PATH} not found!")
    exit(1)

SUMMARY_RE = re.compile(r"\*\*\* SUMMARY \[([\w\s]+)\]:\s*ValLoss\s*=\s*([\d.]+)\s*\+-\s*([\d.]+)\s*\|\s*ValPPL\s*=\s*([\d.]+)\s*\+-\s*([\d.]+)")

results = {}

with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        m = SUMMARY_RE.search(line)
        if m:
            model = m.group(1).strip()
            mean_loss = float(m.group(2))
            se_loss = float(m.group(3))
            mean_ppl = float(m.group(4))
            se_ppl = float(m.group(5))
            results[model] = (mean_loss, se_loss, mean_ppl, se_ppl)

print("=" * 80)
print("RESUMEN DEL BENCHMARK V306 TINY LM ISO-PARAMETRIC MULTI-SEED (v306_log.txt)")
print("=" * 80)

header = f"  {'Model':35s} | {'Mean ValLoss +- SE':>22s} | {'Mean ValPPL +- SE':>22s}"
print(header)
print("  " + "-" * len(header))

best_model = min(results.keys(), key=lambda m: results[m][0]) if results else None

for m in sorted(results.keys()):
    mean_l, se_l, mean_p, se_p = results[m]
    star = " 🌟" if m == best_model else ""
    loss_str = f"{mean_l:.4f} +- {se_l:.4f}"
    ppl_str = f"{mean_p:.2f} +- {se_p:.2f}{star}"
    print(f"  {m:35s} | {loss_str:>22s} | {ppl_str:>22s}")

if results:
    models = sorted(list(results.keys()))
    ppls = [results[m][2] for m in models]
    ses = [results[m][3] for m in models]
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(models, ppls, yerr=ses, capsize=5, color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(models)])
    ax.set_title("Validation Perplexity (PPL +- SE) — Level 2 ANCLA (5 Seeds)", fontsize=11)
    ax.set_ylabel("Validation PPL (lower is better)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "v306_val_ppl_multiseed.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[OK] Grafica guardada en {out_path}")

print("\nAnalisis v306 finalizado con exito!")
