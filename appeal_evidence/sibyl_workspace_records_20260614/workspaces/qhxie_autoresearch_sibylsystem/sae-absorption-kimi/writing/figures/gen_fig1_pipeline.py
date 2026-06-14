"""Generate Figure 1: Experimental pipeline diagram."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set paper style
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})

fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.5)
ax.axis("off")

# Colors
colors = {
    "data": "#2E5AAC",
    "variant": "#5DAE6E",
    "baseline": "#D95F43",
    "metric": "#9467BD",
    "arrow": "#888888",
}

# === LEFT PANEL: Data Generation ===
box_data = FancyBboxPatch((0.3, 0.8), 2.8, 3.0, boxstyle="round,pad=0.1",
                           facecolor=colors["data"], alpha=0.15, edgecolor=colors["data"], linewidth=2)
ax.add_patch(box_data)
ax.text(1.7, 3.5, "SynthSAEBench-16k", fontsize=11, fontweight="bold", ha="center", color=colors["data"])
ax.text(1.7, 3.0, r"$F = 16{,}384$ features", fontsize=9, ha="center")
ax.text(1.7, 2.5, r"$F_h = 10{,}884$ hierarchical", fontsize=9, ha="center")
ax.text(1.7, 2.0, "128 trees, depth 3", fontsize=9, ha="center")
ax.text(1.7, 1.5, r"$|\mathcal{H}| = 992$ pairs", fontsize=9, ha="center")
ax.text(1.7, 1.05, "Known ground-truth", fontsize=8, ha="center", style="italic", color=colors["data"])

# === CENTER PANEL: SAE Variants ===
box_variants = FancyBboxPatch((3.6, 0.5), 4.4, 3.6, boxstyle="round,pad=0.1",
                               facecolor=colors["variant"], alpha=0.12, edgecolor=colors["variant"], linewidth=2)
ax.add_patch(box_variants)
ax.text(5.8, 3.85, "6 SAE Variants (1 component each)", fontsize=11, fontweight="bold", ha="center", color=colors["variant"])

variant_y = 3.3
variant_items = [
    ("Baseline", "ReLU + L1", colors["baseline"]),
    ("+TopK", "k=50 hard sparsity", colors["variant"]),
    ("+MultiScale", "Nested dictionaries", colors["variant"]),
    ("+Orthogonality", "Decoder ortho. penalty", colors["variant"]),
    ("+Gating", "Decoupled detection", colors["variant"]),
    ("+Full Matryoshka", "All combined", "#7f7f7f"),
]

for name, desc, color in variant_items:
    ax.add_patch(FancyBboxPatch((3.8, variant_y - 0.18), 0.25, 0.25, boxstyle="round,pad=0.02",
                                 facecolor=color, alpha=0.7, edgecolor="none"))
    ax.text(4.15, variant_y, name, fontsize=8.5, fontweight="bold", va="center")
    ax.text(5.8, variant_y, desc, fontsize=8, va="center", color="#555555")
    variant_y -= 0.45

# === RIGHT PANEL: Metrics ===
box_metrics = FancyBboxPatch((8.4, 0.8), 3.3, 3.0, boxstyle="round,pad=0.1",
                              facecolor=colors["metric"], alpha=0.15, edgecolor=colors["metric"], linewidth=2)
ax.add_patch(box_metrics)
ax.text(10.05, 3.5, "Ground-Truth Metrics", fontsize=11, fontweight="bold", ha="center", color=colors["metric"])

metrics = [
    ("Absorption rate", "Direct from known pairs"),
    ("Feature recovery MCC", "Hungarian matching"),
    ("Reconstruction MSE", "Standard MSE"),
    ("L0 sparsity", "Active latents per sample"),
    ("Hedging score", "Opposite failure mode"),
]

metric_y = 3.0
for name, desc in metrics:
    ax.text(8.6, metric_y, "•", fontsize=12, va="center", color=colors["metric"])
    ax.text(8.85, metric_y, name, fontsize=9, fontweight="bold", va="center")
    ax.text(10.05, metric_y - 0.18, desc, fontsize=8, ha="center", color="#666666", style="italic")
    metric_y -= 0.5

# === ARROWS ===
# Data -> Variants
ax.annotate("", xy=(3.5, 2.3), xytext=(3.1, 2.3),
            arrowprops=dict(arrowstyle="->", color=colors["arrow"], lw=2))

# Variants -> Metrics
ax.annotate("", xy=(8.3, 2.3), xytext=(8.0, 2.3),
            arrowprops=dict(arrowstyle="->", color=colors["arrow"], lw=2))

# Bottom label
ax.text(6.0, 0.15, "Component-isolated causal attribution: varying one architectural component at a time",
        fontsize=9, ha="center", style="italic", color="#666666")

plt.tight_layout()
plt.savefig("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig1_pipeline.pdf", format="pdf")
plt.savefig("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig1_pipeline.png", format="png")
print("Figure 1 saved.")
