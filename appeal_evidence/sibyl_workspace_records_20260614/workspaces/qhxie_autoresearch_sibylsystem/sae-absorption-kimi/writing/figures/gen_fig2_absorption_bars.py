"""Generate Figure 2: Absorption rate by variant with error bars (all 7 conditions)."""
import matplotlib.pyplot as plt
import numpy as np

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
})

fig, ax = plt.subplots(figsize=(8, 4.5))

variants = ["Baseline", "+TopK", "+MultiScale", "+Orthogonality", "+Gating", "+Full\nMatryoshka", "Random"]
means = [0.252235, 0.055562, 0.054835, 0.245403, 0.261258, 0.066244, 0.534298]
stds = [0.046084, 0.020630, 0.023830, 0.049514, 0.050000, 0.029598, 0.050000]

# Color by effect size: green = large (d > 0.8), yellow = moderate, red = negligible (d < 0.2)
# Effect sizes vs Baseline: TopK=4.93, MultiScale=4.81, Matryoshka=4.31, Ortho=0.13, Gated=-0.17, Random=-5.24
colors = ["#D95F43", "#5DAE6E", "#5DAE6E", "#D95F43", "#D95F43", "#5DAE6E", "#888888"]

bars = ax.bar(variants, means, yerr=stds, capsize=4, color=colors, edgecolor="black",
              linewidth=0.5, alpha=0.85, error_kw={"elinewidth": 1.5, "capthick": 1.5})

# Add value labels on bars
for bar, mean, std in zip(bars, means, stds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.015,
            f"{mean:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_ylabel("Absorption Rate", fontsize=11)
ax.set_title("Component-Isolated Absorption Rates (5 replicates each)", fontsize=12, fontweight="bold")
ax.set_ylim(0, 0.7)

# Add effect size annotations
ax.text(1, 0.42, r"$d = 4.93$", ha="center", fontsize=8, color="#2E5AAC", fontweight="bold")
ax.text(2, 0.42, r"$d = 4.81$", ha="center", fontsize=8, color="#2E5AAC", fontweight="bold")
ax.text(3, 0.42, r"$d = 0.13$", ha="center", fontsize=8, color="#2E5AAC", fontweight="bold")
ax.text(4, 0.45, r"$d = -0.17$", ha="center", fontsize=8, color="#2E5AAC", fontweight="bold")
ax.text(5, 0.38, r"$d = 4.31$", ha="center", fontsize=8, color="#2E5AAC", fontweight="bold")

plt.tight_layout()
plt.savefig("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig2_absorption_bars.pdf", format="pdf")
plt.savefig("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig2_absorption_bars.png", format="png")
print("Figure 2 saved.")
