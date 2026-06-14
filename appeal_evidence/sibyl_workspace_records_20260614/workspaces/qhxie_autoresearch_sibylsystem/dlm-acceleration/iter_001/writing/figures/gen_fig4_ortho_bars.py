"""
Figure 4: Orthogonality Comparison Bar Chart (Section 5.2)
Grouped bars per method pair, one bar per benchmark. Horizontal threshold at Ortho=1.0.
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

# ──────────────────────────────────────────────
# Data from full_pairwise_ortho.json (2-seed: seeds 42, 123)
# ──────────────────────────────────────────────
# Per-seed ortho values
# M1+IGSD: seed42=1.292, seed123=1.478
# M3+IGSD: seed42=0.462, seed123=0.524
# M1+M3:   seed42=0.289, seed123=0.312

pairs       = ["M1 + IGSD", "M3 + IGSD", "M1 + M3"]
ortho_mean  = [1.385, 0.493, 0.301]
ortho_std   = [
    (1.478 - 1.292) / 2,   # 0.093
    (0.524 - 0.462) / 2,   # 0.031
    (0.312 - 0.289) / 2,   # 0.012
]
# Pair-level QAS for annotation
qas_vals    = [1.654, 0.826, 0.504]

pair_colors = [COLORS["M1+IGSD"], COLORS["M3+IGSD"], COLORS["M1+M3"]]

fig, ax = plt.subplots(figsize=(5.5, 3.5))

x = np.arange(len(pairs))
width = 0.55

bars = ax.bar(x, ortho_mean, width,
              yerr=ortho_std,
              color=pair_colors,
              capsize=4,
              edgecolor="black",
              linewidth=0.6,
              error_kw={"linewidth": 0.8, "capthick": 0.8},
              label=pairs,
              zorder=3)

# Multiplicative threshold line
ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0,
           label="Multiplicative threshold (Ortho = 1.0)", zorder=4)

# Synergy annotation
ax.fill_betweenx([1.0, 1.65], -0.4, 0.4, color="#CCFFCC", alpha=0.25, zorder=1)
ax.text(0.0, 1.42, "SYNERGY", ha="center", va="bottom",
        color="#228833", fontsize=7.5, fontweight="bold")

# Interference shading
ax.fill_betweenx([0.0, 0.8], 0.6, 2.4, color="#FFCCCC", alpha=0.20, zorder=1)
ax.text(1.5, 0.55, "INTERFERENCE", ha="center", va="bottom",
        color="#CC3311", fontsize=7.5, fontweight="bold")

# Value labels and QAS annotation on top of bars
for i, (bar, ortho, qas) in enumerate(zip(bars, ortho_mean, qas_vals)):
    y_top = bar.get_height() + ortho_std[i] + 0.03
    ax.text(bar.get_x() + bar.get_width() / 2, y_top,
            f"Ortho={ortho:.3f}\nQAS={qas:.3f}",
            ha="center", va="bottom", fontsize=6.5, linespacing=1.3)

ax.set_xticks(x)
ax.set_xticklabels(pairs, fontsize=8.5)
ax.set_ylabel("Orthogonality (Ortho)", fontsize=9)
ax.set_title("Pairwise Orthogonality Scores\n(2-seed estimate, 200 GSM8K + 164 HumanEval)", fontsize=9)
ax.set_ylim(0, 1.75)
ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
ax.grid(axis="y", alpha=0.25, linewidth=0.4)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "fig4_ortho_bars.pdf")
plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
print(f"Saved: {out_path}")
