"""Generate Figure 6: Component interaction - additive vs. observed for Full Matryoshka."""
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

fig, ax = plt.subplots(figsize=(7, 4.5))

# Data: Baseline, TopK, MultiScale, Expected (additive), Observed (Full Matryoshka)
categories = ["Baseline", "+TopK", "+MultiScale", "Expected\n(additive)", "Observed\n(Full Matryoshka)"]
values = [0.252235, 0.055562, 0.054835, 0.0, 0.066244]
stds = [0.046084, 0.020630, 0.023830, 0.0, 0.029598]

# Colors: Baseline = coral, individual components = green, expected = blue, observed = orange
colors = ["#D95F43", "#5DAE6E", "#5DAE6E", "#2E5AAC", "#ff7f0e"]

bars = ax.bar(categories, values, yerr=stds, capsize=5, color=colors, edgecolor="black",
              linewidth=0.5, alpha=0.85, error_kw={"elinewidth": 1.5, "capthick": 1.5})

# Add horizontal line at zero
ax.axhline(y=0, color="#888888", linestyle="-", linewidth=0.8)

# Add value labels
for bar, val, std in zip(bars, values, stds):
    height = bar.get_height()
    if height >= 0:
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    else:
        ax.text(bar.get_x() + bar.get_width()/2., height - std - 0.025,
                f"{val:.3f}", ha="center", va="top", fontsize=9, fontweight="bold")

# Add annotation for antagonistic interaction
ax.annotate("Antagonistic:\nobserved > best\nindividual",
            xy=(4, 0.066), xytext=(3.3, 0.15),
            arrowprops=dict(arrowstyle="->", color="#D95F43", lw=1.5),
            fontsize=9, color="#D95F43", fontweight="bold",
            ha="center")

ax.set_ylabel("Absorption Rate", fontsize=11)
ax.set_title("Full Matryoshka: Antagonistic Component Interaction", fontsize=12, fontweight="bold")
ax.set_ylim(-0.05, 0.35)

plt.tight_layout()
plt.savefig("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig6_interaction.pdf", format="pdf")
plt.savefig("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig6_interaction.png", format="png")
print("Figure 6 saved.")
