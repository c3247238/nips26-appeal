"""Generate Figure 3: EDA AUROC heatmap across SAE configurations.

Shows where EDA works (mid-layers, narrow SAEs) and where it does not,
with a separate sub-panel for GPT-2 results.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Data from Phase 1 (Gemma Scope) and Phase 5 (GPT-2) ---

# Gemma Scope: rows = layers, cols = widths
gemma_layers = [5, 12, 19]
gemma_widths = ["16k", "65k"]
gemma_auroc = np.array([
    [0.698, 0.617],   # L5
    [0.776, 0.468],   # L12
    [0.458, 0.562],   # L19
])
gemma_ci_lo = np.array([
    [0.637, 0.532],
    [0.700, 0.315],
    [0.317, 0.438],
])
gemma_ci_hi = np.array([
    [0.779, 0.725],
    [0.863, 0.620],
    [0.590, 0.683],
])
gemma_pass = gemma_auroc >= 0.65

# GPT-2
gpt2_configs = ["L6", "L10"]
gpt2_auroc_eda = [0.629, 0.336]
gpt2_auroc_deda = [0.656, 0.762]
gpt2_ci_eda = [[0.561, 0.692], [0.245, 0.435]]
gpt2_ci_deda = [[0.597, 0.714], [0.686, 0.830]]

# --- Style ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2),
                          gridspec_kw={"width_ratios": [3, 2], "wspace": 0.35})

# --- Panel (a): Gemma Scope heatmap ---
ax = axes[0]
cmap = plt.cm.RdYlGn
norm = matplotlib.colors.Normalize(vmin=0.35, vmax=0.85)

im = ax.imshow(gemma_auroc, cmap=cmap, norm=norm, aspect="auto")

for i in range(len(gemma_layers)):
    for j in range(len(gemma_widths)):
        val = gemma_auroc[i, j]
        ci_text = f"[{gemma_ci_lo[i,j]:.2f}, {gemma_ci_hi[i,j]:.2f}]"
        color = "white" if val < 0.50 else "black"
        fontweight = "bold" if gemma_pass[i, j] else "normal"
        ax.text(j, i - 0.12, f"{val:.3f}", ha="center", va="center",
                fontsize=11, fontweight=fontweight, color=color)
        ax.text(j, i + 0.18, ci_text, ha="center", va="center",
                fontsize=7, color=color, alpha=0.8)
        if gemma_pass[i, j]:
            rect = mpatches.FancyBboxPatch(
                (j - 0.45, i - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.02", linewidth=2,
                edgecolor="green", facecolor="none"
            )
            ax.add_patch(rect)

ax.set_xticks(range(len(gemma_widths)))
ax.set_xticklabels([f"$d_{{\\mathrm{{SAE}}}}$ = {w}" for w in gemma_widths])
ax.set_yticks(range(len(gemma_layers)))
ax.set_yticklabels([f"Layer {l}" for l in gemma_layers])
ax.set_title("(a) Gemma Scope (EDA AUROC)", fontsize=11, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("AUROC", fontsize=9)

# --- Panel (b): GPT-2 bar chart ---
ax2 = axes[1]
x = np.arange(len(gpt2_configs))
width_bar = 0.35

bars_eda = ax2.bar(x - width_bar/2, gpt2_auroc_eda, width_bar, label="EDA",
                    color="#4C72B0", alpha=0.85)
bars_deda = ax2.bar(x + width_bar/2, gpt2_auroc_deda, width_bar, label="D-EDA",
                     color="#DD8452", alpha=0.85)

# Error bars
for i, (eda, deda) in enumerate(zip(gpt2_ci_eda, gpt2_ci_deda)):
    ax2.errorbar(i - width_bar/2, gpt2_auroc_eda[i],
                 yerr=[[gpt2_auroc_eda[i] - eda[0]], [eda[1] - gpt2_auroc_eda[i]]],
                 fmt="none", color="black", capsize=3, linewidth=1)
    ax2.errorbar(i + width_bar/2, gpt2_auroc_deda[i],
                 yerr=[[gpt2_auroc_deda[i] - deda[0]], [deda[1] - gpt2_auroc_deda[i]]],
                 fmt="none", color="black", capsize=3, linewidth=1)

ax2.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Chance")
ax2.axhline(y=0.65, color="green", linestyle=":", linewidth=1.0, alpha=0.6, label="Pass (0.65)")

ax2.set_xticks(x)
ax2.set_xticklabels([f"GPT-2 {c}" for c in gpt2_configs])
ax2.set_ylabel("AUROC")
ax2.set_ylim(0.2, 0.9)
ax2.set_title("(b) GPT-2 Small", fontsize=11, fontweight="bold")
ax2.legend(fontsize=7.5, loc="upper left")

plt.tight_layout()
plt.savefig("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current/writing/figures/fig3_eda_heatmap.pdf",
            bbox_inches="tight", dpi=300)
plt.close()
print("Saved fig3_eda_heatmap.pdf")
