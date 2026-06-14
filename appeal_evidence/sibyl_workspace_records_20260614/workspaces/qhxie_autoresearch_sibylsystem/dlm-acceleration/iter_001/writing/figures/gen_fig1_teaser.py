"""
Figure 1: Teaser — Composability Landscape for ComposeAccel paper.
Generates a figure showing (left) the Ortho score heatmap/bar chart for all pairs,
and (right) a scatter of Speedup vs. QAS for individual methods and pairs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from style_config import apply_style, COLORS, MARKERS, FIGSIZE_WIDE, DPI

apply_style()

# ---- Data from full_pairwise_ortho.json ----
# Pairwise Ortho scores (2-seed mean)
ortho_data = {
    "M1+IGSD": 1.385,
    "M3+IGSD": 0.493,
    "M1+M3":   0.301,
}
# QAS for each individual method (3-seed full from outline)
individual_qas = {
    "M1":   0.836,
    "M3":   1.675,
    "IGSD": 1.194,
}
individual_speedup = {
    "M1":   1.38,
    "M3":   1.33,
    "IGSD": 3.40,
}
# Combined (from pairwise)
combined_qas = {
    "M1+IGSD": 1.654,
    "M3+IGSD": 0.826,
    "M1+M3":   0.504,
}
combined_speedup = {
    "M1+IGSD": 5.13,
    "M3+IGSD": 2.34,
    "M1+M3":   0.93,
}

fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

# ---- Left: Orthogonality bar chart ----
ax = axes[0]
pairs = list(ortho_data.keys())
ortho_vals = [ortho_data[p] for p in pairs]
bar_colors = [COLORS["synergy"] if v >= 1.0 else COLORS["interference"] for v in ortho_vals]

x = np.arange(len(pairs))
bars = ax.bar(x, ortho_vals, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.6, zorder=3)

# Multiplicative threshold line
ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.0, label="Multiplicative threshold (Ortho=1.0)", zorder=4)

# Annotate bars
for bar, val in zip(bars, ortho_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.03,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(pairs, fontsize=8)
ax.set_ylabel("Orthogonality Score (Ortho)", fontsize=8.5)
ax.set_ylim(0, 1.65)
ax.set_title("(a) Pairwise Composability Landscape", fontsize=9)

synergy_patch = mpatches.Patch(color=COLORS["synergy"], label="SYNERGY (Ortho>1.0)")
interference_patch = mpatches.Patch(color=COLORS["interference"], label="INTERFERENCE (Ortho<0.8)")
ax.legend(handles=[synergy_patch, interference_patch], fontsize=7, loc="upper right")
ax.grid(axis="y", alpha=0.3, linewidth=0.4)

# ---- Right: Speedup vs. QAS scatter ----
ax2 = axes[1]

# Plot individual methods
for m, sp in individual_speedup.items():
    q = individual_qas[m]
    ax2.scatter(sp, q, color=COLORS[m], marker=MARKERS[m], s=80, zorder=5,
                label=m, edgecolors="black", linewidths=0.5)
    ax2.annotate(m, (sp, q), textcoords="offset points", xytext=(6, 2), fontsize=7.5)

# Plot combined methods
for combo, sp in combined_speedup.items():
    q = combined_qas[combo]
    color = COLORS[combo]
    marker = MARKERS[combo]
    ax2.scatter(sp, q, color=color, marker=marker, s=100, zorder=5,
                label=combo, edgecolors="black", linewidths=0.5)
    offset = (6, 2)
    if combo == "M1+M3":
        offset = (6, -10)
    ax2.annotate(combo, (sp, q), textcoords="offset points", xytext=offset, fontsize=7.5)

# Baseline point
ax2.scatter(1.0, 1.0, color=COLORS["baseline"], marker=MARKERS["baseline"], s=80, zorder=5,
            label="Baseline", edgecolors="black", linewidths=0.5)
ax2.annotate("Baseline", (1.0, 1.0), textcoords="offset points", xytext=(6, 2), fontsize=7.5)

# Reference lines
ax2.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)
ax2.axvline(x=1.0, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)

ax2.set_xlabel("Speedup (×)", fontsize=8.5)
ax2.set_ylabel("Quality-Adjusted Speedup (QAS)", fontsize=8.5)
ax2.set_title("(b) Speed–Quality Landscape", fontsize=9)
ax2.legend(fontsize=6.5, loc="upper left", ncol=2)
ax2.grid(alpha=0.2, linewidth=0.4)

plt.tight_layout(pad=0.8)

out_path = os.path.join(os.path.dirname(__file__), "fig1_teaser.pdf")
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
