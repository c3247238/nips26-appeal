"""
Figure 5: Absorption Phase Stability

Scatter plot: x = 1/L0, y = absorption_rate across all 11 SAE configs.
Overlay sigmoid and linear fits. Show Spearman rho annotation.

Output: writing/figures/fig5_phase_stability.pdf
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
import matplotlib.patches as mpatches

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_E1 = os.path.join(os.path.dirname(WORKSPACE), "exp", "results", "full",
                       "E1_phase_transition.json")
DATA_B2 = os.path.join(os.path.dirname(WORKSPACE), "exp", "results", "full",
                       "B2_scaling_curve.json")
OUT_PATH = os.path.join(WORKSPACE, "figures", "fig5_phase_stability.pdf")

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
GRAY_FIT     = "#888888"
ORANGE_SIG   = "#e6811a"

with open(DATA_E1) as f:
    d_e1 = json.load(f)

results = d_e1["all_results"]

# Extract per-config data
primary = sorted([r for r in results if r["group"] == "primary"], key=lambda r: r["l0"])
ajt     = [r for r in results if r["group"] == "ajt"]
width   = sorted([r for r in results if r["group"] == "width"], key=lambda r: r["width"])

fig, ax = plt.subplots(figsize=(6.5, 3.8))
fig.subplots_adjust(left=0.11, right=0.97, bottom=0.13, top=0.88)

# ── Horizontal reference line at mean ─────────────────────────────────────
all_abs = [r["absorption_rate"] for r in results]
mean_abs = np.mean(all_abs)
ax.axhline(mean_abs, color=GRAY_FIT, lw=0.8, linestyle="--", zorder=1,
           label=f"Mean = {mean_abs:.3f}")

# ── Primary suite ─────────────────────────────────────────────────────────
ax.scatter([r["inv_l0"] for r in primary],
           [r["absorption_rate"] for r in primary],
           s=70, color=BLUE_PRIMARY, marker="o", zorder=5,
           label="Standard/L1 SAE (jb suite)", edgecolors="black", linewidths=0.4)

# ── AJT suite ─────────────────────────────────────────────────────────────
ax.scatter([r["inv_l0"] for r in ajt],
           [r["absorption_rate"] for r in ajt],
           s=70, color=RED_AJT, marker="^", zorder=5,
           label="AJT-trained SAE", edgecolors="black", linewidths=0.4)

# ── Width suite ───────────────────────────────────────────────────────────
ax.scatter([r["inv_l0"] for r in width],
           [r["absorption_rate"] for r in width],
           s=70, color=GREEN_WIDTH, marker="s", zorder=5,
           label="Width sweep (L8)", edgecolors="black", linewidths=0.4)

# ── Sigmoid fit curve (from E1 curve_fits) ────────────────────────────────
# Parameters from E1: L=0.663, k=8569, x0=0.01228, b=1.5e-5
L_sig, k_sig, x0_sig = 0.663468, 8569.37, 0.012282
x_fit = np.linspace(0.009, 0.057, 200)
y_sig = L_sig / (1 + np.exp(-k_sig * (x_fit - x0_sig))) + 1.518e-5

ax.plot(x_fit, y_sig, color=ORANGE_SIG, lw=1.5, linestyle="-.",
        zorder=3, label=f"Sigmoid fit (LRT p = 0.456)", alpha=0.8)

# ── Linear fit ────────────────────────────────────────────────────────────
# slope=1.725, intercept=0.602 (from B2)
slope, intercept = 1.7253, 0.6021
y_lin = slope * x_fit + intercept
ax.plot(x_fit, y_lin, color=GRAY_FIT, lw=1.0, linestyle=":",
        zorder=2, label="Linear fit", alpha=0.7)

# ── Annotations ───────────────────────────────────────────────────────────
ax.text(0.97, 0.06,
        "Spearman $\\rho$ = 0.191, $p$ = 0.574\n"
        "Sigmoid vs. linear: BIC diff = −3.22\n"
        "LRT $p$ = 0.456 (sigmoid not preferred)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#cccccc", alpha=0.95))

# Label layer/width for primary points
layer_labels = [2, 4, 6, 8, 10]
for r, lbl in zip(primary, layer_labels):
    ax.text(r["inv_l0"] + 0.0005, r["absorption_rate"] + 0.003,
            f"L{lbl}", fontsize=6.5, color=BLUE_PRIMARY, ha="left")

ax.set_xlabel("1 / L0 (sparsity proxy)", fontsize=9)
ax.set_ylabel("Absorption rate", fontsize=9)
ax.set_title("Absorption Phase Stability Across 11 SAE Configurations", fontsize=10,
             fontweight="bold")
ax.set_ylim(0.84, 1.01)
ax.legend(loc="lower right", fontsize=7.5, frameon=True)

fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
print(f"Saved: {OUT_PATH}")
plt.close()
