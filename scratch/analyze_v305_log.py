"""
analyze_v305_log.py
===================
Parses v305_log.txt (MQAR Harness Certification log) and generates:
  - Console summary table comparing WITH_Warmup vs NO_Warmup across sequence lengths L.
  - Heatmap/Bar chart of accuracy recovery.

Usage:
    python scratch/analyze_v305_log.py
"""

import re, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_PATH = "v305_log.txt"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(LOG_PATH):
    print(f"Error: {LOG_PATH} not found!")
    exit(1)

RESULT_RE = re.compile(r"\[([\w\s]+)\s*\|\s*(WITH_Warmup|NO_Warmup)\]\s*L=(\d+)\s*->\s*Acc:\s*([\d.]+)%")

results = defaultdict(dict)

with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        m = RESULT_RE.search(line)
        if m:
            model = m.group(1).strip()
            mode = m.group(2).strip()
            seq_len = int(m.group(3))
            acc = float(m.group(4))
            results[seq_len][f"{model}_{mode}"] = acc

print("=" * 80)
print("RESUMEN DEL BENCHMARK V305 MQAR HARNESS CERTIFICATION (v305_log.txt)")
print("=" * 80)

all_lens = sorted(list(results.keys()))
all_keys = sorted(list(results[all_lens[0]].keys())) if all_lens else []

header = f"  {'Model & Mode':50s}" + "".join(f" | L={l:<4d}" for l in all_lens)
print(header)
print("  " + "-" * len(header))

for key in all_keys:
    row = f"  {key:50s}"
    for l in all_lens:
        acc = results[l].get(key, 0.0)
        row += f" | {acc:6.2f}%"
    print(row)

print("\nAnalisis v305 finalizado con exito!")
