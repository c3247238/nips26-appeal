"""Generate Figure 5: Reconstruction-absorption Pareto frontier (all 7 variants)."""
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

fig, ax = plt.subplots(figsize=(6.5, 4.5))

# Data: (MSE, absorption, label, color) - all 7 variants
points = [
    (0.01044, 0.252235, "Baseline", "#D95F43"),
    (0.00768, 0.055562, "+TopK", "#5DAE6E"),
    (0.00710, 0.054835, "+MultiScale", "#5DAE6E"),
    (0.00003, 0.245403, "+Orthogonality", "#D95F43"),
    (0.00821, 0.261258, "+Gating", "#D95F43"),
    (0.00710, 0.066244, "+Full Matryoshka", "#ff7f0e"),
    (0.649, 0.534, "Random", "#888888"),
]

for mse, abs_val, label, color in points:
    ax.scatter(mse, abs_val, s=120, c=color, edgecolors="black", linewidth=0.5, alpha=0.85, zorder=3)
    # Adjust offsets to avoid overlap
    if label == "+MultiScale":
        offset = (8, -15)
    elif label == "+Full Matryoshka":
        offset = (8, -15)
    elif label == "+Orthogonality":
        offset = (8, 5)
    elif label == "Random":
        offset = (-50, 5)
    else:
        offset = (8, 5)
    ax.annotate(label, (mse, abs_val), textcoords="offset points", xytext=offset, fontsize=8)

# Pareto frontier: points that are not dominated (lower MSE and lower absorption)
# MultiScale (0.00710, 0.0548), TopK (0.00768, 0.0556), Full Matryoshka (0.00710, 0.0662)
# MultiScale dominates Full Matryoshka (same MSE, lower absorption)
# So frontier is: MultiScale -> TopK (MultiScale has lower MSE and absorption than TopK)
# Actually need to check: TopK has higher MSE but also higher absorption than MultiScale
# So MultiScale dominates TopK. Frontier is just MultiScale.
# But let's include Orthogonality since it has extremely low MSE
pareto_points = [(0.00710, 0.054835), (0.00003, 0.245403)]
pareto_x = [p[0] for p in pareto_points]
pareto_y = [p[1] for p in pareto_points]
ax.plot(pareto_x, pareto_y, "--", color="#2E5AAC", linewidth=2, alpha=0.7, label="Pareto frontier")

# Ideal point
ax.scatter([0], [0], s=150, c="#2E5AAC", marker="*", edgecolors="black", linewidth=0.5, zorder=4)
ax.annotate("Ideal (0, 0)", (0, 0), textcoords="offset points", xytext=(10, 5), fontsize=8, color="#2E5AAC")

ax.set_xlabel("Reconstruction MSE", fontsize=11)
ax.set_ylabel("Absorption Rate", fontsize=11)
ax.set_title("Reconstruction--Absorption Pareto Frontier", fontsize=12, fontweight="bold")
ax.set_xlim(-0.005, 0.12)
ax.set_ylim(0, 0.65)
ax.legend(loc="upper right", frameon=False)

plt.tight_layout()
plt.savefig("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig5_pareto.pdf", format="pdf")
plt.savefig("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig5_pareto.png", format="png")
print("Figure 5 saved.")
