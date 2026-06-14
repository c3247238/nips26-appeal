"""Figure 1: Pareto front scatter plot — Absorption vs. Hedging vs. Reconstruction."""
import json
import matplotlib.pyplot as plt
import numpy as np
from style_config import set_paper_style, COLORS

set_paper_style()

with open("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current/exp/results/full/e1_full_gpt2_pareto_results.json") as f:
    data = json.load(f)

checkpoints = data["checkpoints"]
archs = sorted({c["architecture"] for c in checkpoints})

fig, ax = plt.subplots(figsize=(5.5, 4.2))

for arch in archs:
    pts = [c for c in checkpoints if c["architecture"] == arch]
    x = [p["absorption_score"] for p in pts]
    y = [p["hedging_score"] for p in pts]
    sizes = [50 + 400 * p["explained_variance"] for p in pts]
    color = COLORS.get(arch, "#333333")
    label = arch.replace("_", " ").title()
    ax.scatter(x, y, s=sizes, c=color, alpha=0.7, edgecolors="white", linewidth=0.5, label=label)
    # Mark Pareto-optimal points with a star
    for p in pts:
        if p.get("is_pareto", False):
            ax.scatter(p["absorption_score"], p["hedging_score"], s=120, c=color, marker="*", edgecolors="black", linewidth=0.8, zorder=5)

ax.set_xlabel("Absorption rate ($\\alpha$")
ax.set_ylabel("Hedging rate ($h$")
ax.set_title("No architecture dominates the full Pareto front")
ax.legend(loc="upper right", frameon=False)

# Add annotation
ax.annotate("Point size = explained variance", xy=(0.95, 0.02), xycoords="axes fraction", ha="right", va="bottom", fontsize=8, color="gray")

plt.tight_layout()
fig.savefig("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current/writing/figures/fig1_pareto.pdf")
print("Saved fig1_pareto.pdf")
