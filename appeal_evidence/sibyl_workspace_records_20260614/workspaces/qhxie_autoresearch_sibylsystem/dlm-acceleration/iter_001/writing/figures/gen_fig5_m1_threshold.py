"""
Figure 5: M1 Speedup and AccRet vs. Entropy Threshold (Section 5.3)
Dual y-axis: speedup (left) and GSM8K AccRet (right).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style_config import apply_style, COLORS, FIGSIZE_WIDE, DPI

apply_style()

# Data from m1_pareto_full.json (3-seed, full benchmark)
# eta, combined_speedup, gsm8k_acc_retention
eta_vals   = [0.5,   1.0,   2.0,   3.0]
speedups   = [0.553, 0.573, 1.380, 1.702]
acc_rets   = [0.934, 0.862, 0.550, 0.145]
chr_vals   = [0.503, 0.527, 0.602, 0.633]   # cache hit rates from full_m1

fig, ax1 = plt.subplots(figsize=(5.0, 3.5))

color_speed = COLORS["M1"]
color_acc   = COLORS["M3"]

# --- Left axis: Speedup ---
ax1.set_xlabel(r"Entropy Threshold $\eta$", fontsize=9)
ax1.set_ylabel("Speedup over Baseline", color=color_speed, fontsize=9)
l1, = ax1.plot(eta_vals, speedups, color=color_speed, marker="o", linewidth=1.4,
               markersize=6, label="Speedup", zorder=4)
ax1.tick_params(axis="y", labelcolor=color_speed)

# Overhead zone (speedup < 1.0)
ax1.axhspan(0.0, 1.0, color="#DDDDFF", alpha=0.35, zorder=1)
ax1.text(0.7, 0.7, "OVERHEAD\nZONE", ha="center", va="center",
         color=color_speed, fontsize=6.5, alpha=0.7)

# Speedup = 1 reference line
ax1.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)

# Critical threshold marker
ax1.axvline(2.0, color="#AA8800", linestyle="--", linewidth=0.9, alpha=0.8,
            label=r"Critical $\eta=2.0$")
ax1.text(2.05, 0.25, r"$\eta^*=2.0$", color="#AA8800", fontsize=7.5)

# --- Right axis: AccRet ---
ax2 = ax1.twinx()
ax2.set_ylabel("GSM8K Accuracy Retention", color=color_acc, fontsize=9)
l2, = ax2.plot(eta_vals, acc_rets, color=color_acc, marker="s", linewidth=1.4,
               linestyle="--", markersize=6, label="AccRet (GSM8K)", zorder=4)
ax2.tick_params(axis="y", labelcolor=color_acc)
ax2.set_ylim(-0.05, 1.20)

ax1.set_xlim(0.2, 3.3)
ax1.set_ylim(-0.05, 2.0)
ax1.set_xticks(eta_vals)
ax1.set_title(r"M1 (EntropyCache): Speedup and AccRet vs. $\eta$", fontsize=9)

# Combined legend
lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, fontsize=7.5, loc="upper left", framealpha=0.9)
ax1.grid(alpha=0.2, linewidth=0.4)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "fig5_m1_threshold.pdf")
plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
print(f"Saved: {out_path}")
