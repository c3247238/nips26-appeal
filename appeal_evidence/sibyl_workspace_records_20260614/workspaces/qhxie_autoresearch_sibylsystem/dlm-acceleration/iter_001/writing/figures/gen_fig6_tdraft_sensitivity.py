"""
Figure 6: IGSD T_draft Sensitivity Analysis (Section 5.4)
Dual y-axis: Speedup (left, decreasing) and AccRet (right, increasing) vs. T_draft.
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

# Data from igsd_ablation.json (tau=0.9, seeds [42, 123])
# IGSD-full ablation by T_draft
t_draft_vals = [4,     8,     16,    32]
speedups     = [1.884, 2.304, 2.664, 2.085]  # combined (GSM8K + HumanEval)
acc_rets     = [0.208, 0.278, 0.359, 0.405]  # combined acc retention

# Per-seed ranges for error bars (from igsd_ablation.json all_results)
speed_per_seed = {
    4:  [1.813, 1.955],
    8:  [2.242, 2.366],
    16: [2.665, 2.663],
    32: [2.066, 2.104],
}
acc_per_seed = {
    4:  [0.193, 0.224],
    8:  [0.247, 0.309],
    16: [0.339, 0.378],
    32: [0.382, 0.428],
}
speed_err = [abs(speed_per_seed[t][1] - speed_per_seed[t][0]) / 2 for t in t_draft_vals]
acc_err   = [abs(acc_per_seed[t][1]   - acc_per_seed[t][0])   / 2 for t in t_draft_vals]

color_speed = COLORS["IGSD"]
color_acc   = "#AA3377"

fig, ax1 = plt.subplots(figsize=(5.0, 3.5))

ax1.set_xlabel(r"$T_{\mathrm{draft}}$ (draft denoising steps)", fontsize=9)
ax1.set_ylabel("Combined Speedup", color=color_speed, fontsize=9)

l1 = ax1.errorbar(t_draft_vals, speedups, yerr=speed_err,
                  color=color_speed, marker="D", linewidth=1.4, markersize=6,
                  capsize=3, capthick=0.8, label="Speedup")
ax1.tick_params(axis="y", labelcolor=color_speed)

# Highlight T_draft=16
ax1.axvline(16, color="#AA8800", linestyle="--", linewidth=0.9, alpha=0.8)
ax1.text(16.3, 2.85, r"$T_d^*=16$", color="#AA8800", fontsize=7.5)

ax2 = ax1.twinx()
ax2.set_ylabel("Combined Accuracy Retention", color=color_acc, fontsize=9)
l2 = ax2.errorbar(t_draft_vals, acc_rets, yerr=acc_err,
                  color=color_acc, marker="s", linewidth=1.4, markersize=6,
                  linestyle="--", capsize=3, capthick=0.8, label="AccRet")
ax2.tick_params(axis="y", labelcolor=color_acc)

ax1.set_xlim(2, 35)
ax1.set_xticks(t_draft_vals)
ax1.set_ylim(1.5, 3.1)
ax2.set_ylim(0.15, 0.47)
ax1.set_title(r"IGSD Sensitivity to $T_{\mathrm{draft}}$ ($\tau=0.9$)", fontsize=9)

lines = [l1, l2]
labels = ["Speedup", "AccRet"]
ax1.legend(lines, labels, fontsize=7.5, loc="center left", framealpha=0.9)
ax1.grid(alpha=0.2, linewidth=0.4)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "fig6_tdraft_sensitivity.pdf")
plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
print(f"Saved: {out_path}")
