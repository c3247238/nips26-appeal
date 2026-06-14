"""
Figure 3: EDA Scaling — Layer and Architecture Sweep

Scatter+line plot of EDA_delta across 11 SAE configurations:
  - Primary jb suite (L2-L10): positive EDA_delta, peaks at L4/L6
  - AJT suite (L6, 3 variants): NEGATIVE EDA_delta (-0.176 to -0.217)
  - Width suite (L8, 12k-98k): EDA_delta decreases with width

Output: writing/figures/fig3_eda_scaling.pdf
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(WORKSPACE), "exp", "results", "full", "B2_scaling_curve.json")
OUT_PATH = os.path.join(WORKSPACE, "figures", "fig3_eda_scaling.pdf")

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE_PRIMARY = "#2166ac"
RED_AJT      = "#d6604d"
GREEN_WIDTH  = "#1a9641"
GRAY_ZERO    = "#aaaaaa"

# ── Load data ──────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    data = json.load(f)

results = data["all_results"]

primary = [r for r in results if r["group"] == "primary"]
ajt     = [r for r in results if r["group"] == "ajt"]
width   = [r for r in results if r["group"] == "width"]

# Sort primary by layer
primary = sorted(primary, key=lambda r: r["layer"])
width   = sorted(width,   key=lambda r: r["width"])

fig, ax = plt.subplots(figsize=(6.5, 3.5))
fig.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.90)

# Zero line
ax.axhline(0, color=GRAY_ZERO, lw=0.8, linestyle="--", zorder=1)

# ── Primary jb suite (connect by layer) ──────────────────────────────────
primary_layers   = [r["layer"] for r in primary]
primary_eda_d    = [r["eda_delta"] for r in primary]
primary_aurocs   = [r["eda_auroc"] for r in primary]

ax.plot(primary_layers, primary_eda_d, color=BLUE_PRIMARY, lw=1.8,
        zorder=2, label="Standard/L1 SAE (jb suite, L2–L10)")
ax.scatter(primary_layers, primary_eda_d, s=60, color=BLUE_PRIMARY,
           zorder=5, marker="o")

# Annotate AUROC next to each point
for layer, eda_d, auroc in zip(primary_layers, primary_eda_d, primary_aurocs):
    offset = 0.004 if eda_d >= 0 else -0.006
    ax.text(layer, eda_d + offset, f"{auroc:.3f}", fontsize=6.5, color=BLUE_PRIMARY,
            ha="center", va="bottom" if eda_d >= 0 else "top")

# ── AJT variants (at layer 6) ─────────────────────────────────────────────
# Jitter x slightly to avoid overlap
ajt_x     = [5.7, 6.0, 6.3]
ajt_eda_d = [r["eda_delta"] for r in ajt]
ajt_l0    = [r["l0"] for r in ajt]
ajt_auroc = [r["eda_auroc"] for r in ajt]

ax.scatter(ajt_x, ajt_eda_d, s=80, color=RED_AJT, zorder=5,
           marker="^", label="AJT-trained SAE (L6, 3 variants)")
for x, d, auroc in zip(ajt_x, ajt_eda_d, ajt_auroc):
    ax.text(x, d - 0.010, f"{auroc:.3f}", fontsize=6.5, color=RED_AJT,
            ha="center", va="top")

# ── Width suite (at layer 8) — show as separate subplot inset ─────────────
width_widths = [r["width"] for r in width]  # 12288, 24576, 49152, 98304
width_eda_d  = [r["eda_delta"] for r in width]
width_x = [8.5, 8.7, 8.9, 9.1][:len(width_eda_d)]   # offset on x axis
ax.scatter(width_x, width_eda_d, s=60, color=GREEN_WIDTH, zorder=5,
           marker="s", label="Width sweep (L8, 12k–98k)")

# ── Axis formatting ───────────────────────────────────────────────────────
ax.set_xlabel("Layer (GPT-2 Small, 0-indexed)", fontsize=9)
ax.set_ylabel("EDA$_\\Delta$ (letter − non-letter mean)", fontsize=9)
ax.set_title("EDA Scaling Across Architectures and Layers", fontsize=10, fontweight="bold")
ax.set_xticks([2, 4, 6, 8, 10])
ax.set_xticklabels(["L2", "L4", "L6", "L8", "L10"])

# Width annotation
ax.annotate("Width sweep\n(L8: 12k→98k)",
            xy=(8.8, np.mean(width_eda_d)), xytext=(9.5, 0.035),
            fontsize=7, color=GREEN_WIDTH,
            arrowprops=dict(arrowstyle="->", color=GREEN_WIDTH, lw=0.8))

# AJT annotation
ax.annotate("AJT SAEs\n(reversed polarity)",
            xy=(6.0, np.mean(ajt_eda_d)), xytext=(7.0, -0.14),
            fontsize=7, color=RED_AJT,
            arrowprops=dict(arrowstyle="->", color=RED_AJT, lw=0.8))

ax.legend(loc="upper right", fontsize=7.5, frameon=True)
ax.set_ylim(-0.28, 0.10)

fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
print(f"Saved: {OUT_PATH}")
plt.close()
