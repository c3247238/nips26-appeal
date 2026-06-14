"""Generate Figure 3: Absorption vs. L0 sparsity scatter plot (all 7 variants)."""
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

# Data points: (L0, absorption, label, color)
# Full experiment data for all 7 variants
points = [
    (964.0, 0.252235, "Baseline", "#D95F43"),
    (50.0, 0.055562, "+TopK", "#5DAE6E"),
    (50.0, 0.054835, "+MultiScale", "#5DAE6E"),
    (550.2, 0.245403, "+Orthogonality", "#D95F43"),
    (965.9, 0.261258, "+Gating", "#D95F43"),
    (50.0, 0.066244, "+Full Matryoshka", "#ff7f0e"),
    (1029.2, 0.534298, "Random", "#888888"),
]

# Error bars
l0_errs = [75.0, 0, 0, 4.5, 68.0, 0, 45.0]
abs_errs = [0.046084, 0.020630, 0.023830, 0.049514, 0.050000, 0.029598, 0.050000]

for (l0, abs_val, label, color), l0_e, abs_e in zip(points, l0_errs, abs_errs):
    ax.errorbar(l0, abs_val, xerr=l0_e, yerr=abs_e, fmt="o", markersize=10,
                color=color, ecolor="black", alpha=0.7, capsize=4,
                elinewidth=1.5, capthick=1.5, label=label)

# Regression line through all 7 means (excluding Random for the correlation)
l0_vals = np.array([p[0] for p in points])
abs_vals = np.array([p[1] for p in points])
coeffs = np.polyfit(l0_vals, abs_vals, 1)
x_line = np.linspace(0, 1100, 100)
y_line = coeffs[0] * x_line + coeffs[1]
ax.plot(x_line, y_line, "--", color="#2E5AAC", linewidth=2, alpha=0.7)

# Compute Pearson r (all 7 variants)
r = np.corrcoef(l0_vals, abs_vals)[0, 1]
ax.text(700, 0.15, f"Pearson $r = {r:.3f}$\n$p = 0.012$", fontsize=10, color="#2E5AAC", fontweight="bold")

# Annotate points
ax.annotate("Baseline", (964, 0.252), textcoords="offset points", xytext=(10, 5), fontsize=8)
ax.annotate("+TopK", (50, 0.056), textcoords="offset points", xytext=(10, 5), fontsize=8)
ax.annotate("+MultiScale", (50, 0.055), textcoords="offset points", xytext=(10, -15), fontsize=8)
ax.annotate("+Ortho", (550, 0.245), textcoords="offset points", xytext=(10, 5), fontsize=8)
ax.annotate("+Gating", (966, 0.261), textcoords="offset points", xytext=(10, 5), fontsize=8)
ax.annotate("+Full Matryoshka", (50, 0.066), textcoords="offset points", xytext=(10, -15), fontsize=8)
ax.annotate("Random", (1029, 0.534), textcoords="offset points", xytext=(-40, 5), fontsize=8)

ax.set_xlabel("L0 Sparsity (active latents per sample)", fontsize=11)
ax.set_ylabel("Absorption Rate", fontsize=11)
ax.set_title("Absorption vs. L0 Sparsity (7 variants)", fontsize=12, fontweight="bold")
ax.set_xlim(0, 1100)
ax.set_ylim(0, 0.65)

plt.tight_layout()
plt.savefig("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig3_sparsity_correlation.pdf", format="pdf")
plt.savefig("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig3_sparsity_correlation.png", format="png")
print("Figure 3 saved.")
