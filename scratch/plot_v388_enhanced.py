"""
Visualization script for v388: Enhanced Rate-Distortion & Spectral JPEG Figures
"""

import os
import json
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '.')

from scratch.prototype_v388_jpeg_spectral_quantization import (
    TinyShakespeareDataset,
    train_base_model,
    get_dct_basis
)

def main():
    raw_path = "results/raw/v388_jpeg_spectral_quantization.json"
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    topo_sum = data["Topographic_Full_eps2e-3"]["quant_summary"]
    std_sum = data["Standard_AdamW_Baseline"]["quant_summary"]

    quant_order = [
        "fp32",
        "spatial_int4",
        "spectral_uniform_int4",
        "spatial_int2",
        "jpeg_adaptive_1.7bpp",
        "jpeg_aggressive_0.9bpp"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ----------------------------------------------------
    # Panel 1: Rate-Distortion Frontier (Log Y-Scale)
    # ----------------------------------------------------
    ax1 = axes[0, 0]

    # Plot spectral curves (FP32 -> JPEG Adaptive -> JPEG Aggressive)
    spec_keys = ["fp32", "jpeg_adaptive_1.7bpp", "jpeg_aggressive_0.9bpp"]
    bpp_t_spec = [topo_sum[k]["bpp"] for k in spec_keys]
    ppl_t_spec = [topo_sum[k]["mean_ppl"] for k in spec_keys]
    err_t_spec = [topo_sum[k]["std_ppl"] for k in spec_keys]

    bpp_s_spec = [std_sum[k]["bpp"] for k in spec_keys]
    ppl_s_spec = [std_sum[k]["mean_ppl"] for k in spec_keys]
    err_s_spec = [std_sum[k]["std_ppl"] for k in spec_keys]

    ax1.errorbar(bpp_t_spec, ppl_t_spec, yerr=err_t_spec, fmt='o-', color='#1f77b4', lw=2.5,
                 markersize=8, capsize=4, label='Topographic (JPEG Spectral Frontier)')
    ax1.errorbar(bpp_s_spec, ppl_s_spec, yerr=err_s_spec, fmt='s--', color='#d62728', lw=2.0,
                 markersize=7, capsize=4, label='Standard AdamW (JPEG Spectral Collapse)')

    # Add Spatial INT4 and INT2 points as distinct markers
    ax1.scatter([topo_sum["spatial_int4"]["bpp"]], [topo_sum["spatial_int4"]["mean_ppl"]],
                color='#2ca02c', marker='^', s=120, zorder=5, label='Topographic Spatial INT4 (4.0 bpp)')
    ax1.scatter([std_sum["spatial_int4"]["bpp"]], [std_sum["spatial_int4"]["mean_ppl"]],
                color='#ff7f0e', marker='^', s=120, zorder=5, label='Standard Spatial INT4 (4.0 bpp)')

    ax1.scatter([topo_sum["spatial_int2"]["bpp"]], [topo_sum["spatial_int2"]["mean_ppl"]],
                color='#9467bd', marker='X', s=130, zorder=5, label='Topographic Spatial INT2 (506 PPL, Collapse)')
    ax1.scatter([std_sum["spatial_int2"]["bpp"]], [std_sum["spatial_int2"]["mean_ppl"]],
                color='#8c564b', marker='X', s=130, zorder=5, label='Standard Spatial INT2 (121 PPL, Collapse)')

    # Annotations
    ax1.annotate("JPEG Adaptive (1.68 bpp)\nPPL = 10.24", (1.68, 10.24),
                 textcoords="offset points", xytext=(15, -15), fontsize=9, fontweight="bold", color='#1f77b4',
                 arrowprops=dict(arrowstyle="->", color='#1f77b4', lw=1.2))
    ax1.annotate("JPEG Aggressive (0.91 bpp)\nPPL = 12.72", (0.91, 12.72),
                 textcoords="offset points", xytext=(20, 10), fontsize=9, fontweight="bold", color='#1f77b4',
                 arrowprops=dict(arrowstyle="->", color='#1f77b4', lw=1.2))
    ax1.annotate("Standard JPEG Adaptive\nPPL = 58.29 (8.3x degradation)", (1.68, 58.29),
                 textcoords="offset points", xytext=(20, 10), fontsize=9, fontweight="bold", color='#d62728',
                 arrowprops=dict(arrowstyle="->", color='#d62728', lw=1.2))

    ax1.set_yscale('log')
    ax1.set_ylim(5, 700)
    ax1.set_xlabel("Bits Per Parameter (bpp, Linear Projections)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Validation Perplexity (Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_title("Rate-Distortion Frontier: PPL vs Bit-Budget", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8.5, loc="upper right")

    # ----------------------------------------------------
    # Panel 2: 2D-DCT Power Spectrum with JPEG Bands
    # ----------------------------------------------------
    ax2 = axes[0, 1]
    ds = TinyShakespeareDataset("data/tinyshakespeare.txt", seq_len=128)
    D128 = get_dct_basis(128)
    m_topo, _ = train_base_model({"name": "t", "eps_attn": 2e-3, "eps_ffn": 2e-3}, 42, ds, max_steps=200, lr=2e-3, batch_size=32)
    w_topo = m_topo.blocks[0].attn.q.weight.detach().cpu()
    w_topo_dct = (D128 @ w_topo @ D128.T).abs().numpy()

    im = ax2.imshow(np.log10(w_topo_dct + 1e-5), cmap="magma", aspect="auto")
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label("Log10 |2D-DCT Amplitude|", fontsize=10)

    # Concentric circles for JPEG bands
    theta = np.linspace(0, np.pi/2, 100)
    for r_norm, col, lbl in [(0.15, "#00ffff", "Band 0: 8-bit (r<=0.15)"),
                             (0.35, "#ffff00", "Band 1: 4-bit (r<=0.35)"),
                             (0.60, "#00ff00", "Band 2: 2-bit (r<=0.60)")]:
        r_pixel = r_norm * 128 * np.sqrt(2.0)
        ax2.plot(r_pixel * np.sin(theta), r_pixel * np.cos(theta), color=col, lw=2.2, label=lbl)

    # Annotate Band 3
    ax2.text(90, 90, "Band 3:\n0-bit (Truncated)", color="white", fontsize=9, fontweight="bold", ha="center")

    ax2.set_xlim(0, 127)
    ax2.set_ylim(127, 0)
    ax2.set_title("Topographic Transformer 2D-DCT Energy & JPEG Bands", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Frequency Index u (Columns)", fontsize=10)
    ax2.set_ylabel("Frequency Index v (Rows)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8.5)

    # ----------------------------------------------------
    # Panel 3: Weight Reconstruction Relative Error (%)
    # ----------------------------------------------------
    ax3 = axes[1, 0]
    bar_modes = ["jpeg_adaptive_1.7bpp", "jpeg_aggressive_0.9bpp", "spatial_int4", "spatial_int2", "spectral_uniform_int4"]
    labels = ["JPEG Adaptive\n(1.68 bpp)", "JPEG Aggressive\n(0.91 bpp)", "Spatial INT4\n(4.00 bpp)", "Spatial INT2\n(2.00 bpp)", "Spectral INT4\n(4.00 bpp)"]

    x = np.arange(len(bar_modes))
    width = 0.35

    err_t = [topo_sum[k]["mean_rel_err"] * 100 for k in bar_modes]
    err_s = [std_sum[k]["mean_rel_err"] * 100 for k in bar_modes]

    rects1 = ax3.bar(x - width/2, err_t, width, label='Topographic', color='#1f77b4', alpha=0.85)
    rects2 = ax3.bar(x + width/2, err_s, width, label='Standard AdamW', color='#d62728', alpha=0.85)

    for r in rects1:
        h = r.get_height()
        ax3.text(r.get_x() + r.get_width()/2, h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=8, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax3.text(r.get_x() + r.get_width()/2, h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=8, fontweight="bold")

    ax3.set_ylabel("Reconstruction Relative Error (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Weight Reconstruction Distortion Across Schemes", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_ylim(0, 115)
    ax3.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax3.legend(fontsize=10)

    # ----------------------------------------------------
    # Panel 4: Physical Memory Footprint (KB)
    # ----------------------------------------------------
    ax4 = axes[1, 1]
    mem_modes = ["fp32", "spatial_int4", "spatial_int2", "jpeg_adaptive_1.7bpp", "jpeg_aggressive_0.9bpp"]
    mem_labels = ["FP32 Base", "Spatial INT4", "Spatial INT2", "JPEG Adaptive", "JPEG Aggressive"]
    mem_bpps = [32.0, 4.0, 2.0, 1.6816, 0.9134]
    mem_kb = [262144 * (b / 8.0) / 1024.0 for b in mem_bpps]
    colors = ['#7f7f7f', '#ff7f0e', '#8c564b', '#2ca02c', '#17becf']

    bars = ax4.bar(mem_labels, mem_kb, color=colors, alpha=0.85)
    for bar, kb, b in zip(bars, mem_kb, mem_bpps):
        ratio = 32.0 / b
        ax4.text(bar.get_x() + bar.get_width()/2, kb + 18, f"{kb:.1f} KB\n({ratio:.1f}x)",
                 ha='center', va='bottom', fontsize=8.5, fontweight="bold")

    ax4.set_ylabel("Physical Footprint (KB)", fontsize=11, fontweight="bold")
    ax4.set_title("Linear Weights Physical Footprint (262,144 params)", fontsize=12, fontweight="bold")
    ax4.set_xticks(range(len(mem_labels)))
    ax4.set_xticklabels(mem_labels, fontsize=9.5)
    ax4.set_ylim(0, 1220)
    ax4.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    out_fig = "results/figures/v388_jpeg_spectral_quantization.png"
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print("Enhanced figure successfully saved to:", out_fig)

if __name__ == "__main__":
    main()
