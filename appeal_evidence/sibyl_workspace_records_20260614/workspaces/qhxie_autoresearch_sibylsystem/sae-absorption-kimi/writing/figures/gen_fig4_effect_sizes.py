"""Generate Figure 4: Effect sizes (Cohen's d) with thresholds (all 7 variants)."""
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

fig, ax = plt.subplots(figsize=(6.5, 4))

# All 7 variants with effect sizes vs. Baseline
variants = ["+TopK", "+MultiScale", "+Full\nMatryoshka", "+Orthogonality", "+Gating", "Random"]
effect_sizes = [4.93, 4.81, 4.31, 0.13, -0.17, -5.24]
# Color by magnitude: green = large positive, red = negligible/negative, gray = random
colors = ["#5DAE6E", "#5DAE6E", "#5DAE6E", "#D95F43", "#D95F43", "#888888"]

# Horizontal bar chart
y_pos = np.arange(len(variants))
bars = ax.barh(y_pos, effect_sizes, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85, height=0.5)

# Threshold lines
ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
ax.axvline(x=0.2, color="#888888", linestyle=":", linewidth=1, alpha=0.7)
ax.axvline(x=0.5, color="#888888", linestyle=":", linewidth=1, alpha=0.7)
ax.axvline(x=0.8, color="#888888", linestyle=":", linewidth=1, alpha=0.7)

# Threshold labels
ax.text(0.2, 5.6, "small", fontsize=7, ha="center", color="#888888")
ax.text(0.5, 5.6, "medium", fontsize=7, ha="center", color="#888888")
ax.text(0.8, 5.6, "large", fontsize=7, ha="center", color="#888888")

# Value labels
for bar, d in zip(bars, effect_sizes):
    width = bar.get_width()
    if width >= 0:
        ax.text(width + 0.15, bar.get_y() + bar.get_height()/2.,
                f"$d = {d:.2f}$", ha="left", va="center", fontsize=9, fontweight="bold")
    else:
        ax.text(width - 0.15, bar.get_y() + bar.get_height()/2.,
                f"$d = {d:.2f}$", ha="right", va="center", fontsize=9, fontweight="bold")

ax.set_yticks(y_pos)
ax.set_yticklabels(variants)
ax.set_xlabel("Cohen's $d$ (effect size vs. Baseline)", fontsize=11)
ax.set_title("Component Effect Sizes", fontsize=12, fontweight="bold")
ax.set_xlim(-6.5, 6.5)

plt.tight_layout()
plt.savefig("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig4_effect_sizes.pdf", format="pdf")
plt.savefig("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig4_effect_sizes.png", format="png")
print("Figure 4 saved.")
