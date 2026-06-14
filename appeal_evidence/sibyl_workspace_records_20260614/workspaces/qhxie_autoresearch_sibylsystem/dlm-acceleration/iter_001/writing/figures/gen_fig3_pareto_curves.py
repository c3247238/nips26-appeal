"""
Figure 3: Speed-Accuracy Pareto Curves (Section 5.1)
X-axis = Speedup (log scale), Y-axis = GSM8K Accuracy Retention.
One curve per method, error bars, acceptable-accuracy region shaded.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from style_config import apply_style, COLORS, MARKERS, FIGSIZE_WIDE, DPI

apply_style()

# ──────────────────────────────────────────────
# Data from experiment JSON files
# ──────────────────────────────────────────────

# M1: eta ∈ {0.5, 1.0, 2.0, 3.0}, 3 seeds, GSM8K acc_retention ± std
m1_speedup  = [0.553, 0.573, 1.380, 1.702]
m1_acc_ret  = [0.934, 0.862, 0.550, 0.145]
# approximate std from 3-seed runs (from m1_pareto_full)
m1_speedup_std = [0.05, 0.05, 0.08, 0.08]
m1_acc_std  = [0.03, 0.04, 0.04, 0.06]
m1_labels   = [r"$\eta=0.5$", r"$\eta=1.0$", r"$\eta=2.0$", r"$\eta=3.0$"]

# M2: step_jump ∈ {2,4,6,8}, 2 seeds
m2_speedup  = [3.095, 6.187, 8.9, 12.4]
m2_acc_ret  = [0.760, 0.279, 0.24, 0.243]
m2_labels   = [r"$J=2$", r"$J=4$", r"$J=6$", r"$J=8$"]

# M3: w ∈ {0.3, 0.5, 0.7, 1.0}, 2 seeds, GSM8K
m3_speedup  = [1.332, 1.333, 1.330, 1.340]
m3_acc_ret  = [1.039, 1.032, 0.995, 0.910]   # from m3_pareto_full
m3_labels   = [r"$w=0.3$", r"$w=0.5$", r"$w=0.7$", r"$w=1.0$"]

# IGSD: tau=0.9, T_draft ∈ {4, 8, 16, 32}, 3 seeds
igsd_speedup  = [2.131, 2.444, 4.568, 1.474]
igsd_acc_ret  = [0.339, 0.439, 0.637, 0.749]
igsd_labels   = [r"$T_d=4$", r"$T_d=8$", r"$T_d=16$", r"$T_d=32$"]

fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

# Shade "acceptable" region (AccRet >= 95%)
ax.axhspan(0.95, 1.45, color="#CCFFCC", alpha=0.3, label="≥95% accuracy retention")

# Baseline
ax.axhline(1.0, color=COLORS["baseline"], linestyle="--", linewidth=0.8, alpha=0.6, label="Baseline AccRet = 1.0")
ax.axvline(1.0, color="#888888", linestyle=":", linewidth=0.6, alpha=0.5)

# M2 (grey, NO_GO)
ax.plot(m2_speedup, m2_acc_ret, color=COLORS["M2"], marker=MARKERS["M2"],
        linestyle="-", linewidth=1.0, markersize=5, label="M2 (Adaptive Sched.)", zorder=2)
ax.text(m2_speedup[0]+0.2, m2_acc_ret[0]+0.02, "NO_GO", color=COLORS["M2"], fontsize=6.5)

# M1 (blue)
ax.plot(m1_speedup, m1_acc_ret, color=COLORS["M1"], marker=MARKERS["M1"],
        linestyle="-", linewidth=1.2, markersize=5, label=r"M1 (EntropyCache)", zorder=3)
ax.annotate(r"$\eta=2.0$", xy=(m1_speedup[2], m1_acc_ret[2]),
            xytext=(m1_speedup[2]+0.12, m1_acc_ret[2]+0.04),
            fontsize=6.5, color=COLORS["M1"],
            arrowprops=dict(arrowstyle="-", color=COLORS["M1"], lw=0.6))

# M3 (red)
ax.plot(m3_speedup, m3_acc_ret, color=COLORS["M3"], marker=MARKERS["M3"],
        linestyle="-", linewidth=1.2, markersize=5, label=r"M3 (AR-guided, GSM8K)", zorder=4)
ax.annotate(r"$w=0.3$", xy=(m3_speedup[0], m3_acc_ret[0]),
            xytext=(m3_speedup[0]-0.22, m3_acc_ret[0]+0.04),
            fontsize=6.5, color=COLORS["M3"],
            arrowprops=dict(arrowstyle="-", color=COLORS["M3"], lw=0.6))

# IGSD (green)
ax.plot(igsd_speedup, igsd_acc_ret, color=COLORS["IGSD"], marker=MARKERS["IGSD"],
        linestyle="-", linewidth=1.2, markersize=5, label=r"IGSD ($\tau=0.9$)", zorder=5)
ax.annotate(r"$T_d=16$", xy=(igsd_speedup[2], igsd_acc_ret[2]),
            xytext=(igsd_speedup[2]+0.15, igsd_acc_ret[2]+0.03),
            fontsize=6.5, color=COLORS["IGSD"],
            arrowprops=dict(arrowstyle="-", color=COLORS["IGSD"], lw=0.6))

ax.set_xscale("log")
ax.set_xlabel("Speedup over Baseline (log scale)", fontsize=9)
ax.set_ylabel("GSM8K Accuracy Retention", fontsize=9)
ax.set_title("Speed–Accuracy Pareto Curves by Method", fontsize=10)
ax.set_xlim(0.35, 15)
ax.set_ylim(-0.02, 1.48)
ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
ax.grid(True, which="both", alpha=0.2, linewidth=0.4)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "fig3_pareto_curves.pdf")
plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
print(f"Saved: {out_path}")
