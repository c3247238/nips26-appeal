"""
Generate Figure 5: CMI Dimension Sensitivity
Shows Spearman rho between CMI and absorption rate across subspace dimensions d'.
Highlights the instability -- negative correlation at d'=10 reverses at higher dimensions.
"""
import sys
sys.path.insert(0, '/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current/writing/figures')
from style_config import *
import numpy as np

# Data from geometric_constant.json subspace_dim_comparison
dims = [10, 20, 30, 50]
rhos = [-0.383, 0.048, 0.299, 0.197]
p_values = [0.059, 0.818, 0.147, 0.345]

# Bonferroni-corrected significance threshold
alpha_bonf = 0.05 / len(dims)  # 0.0125

fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT * 0.85))

# Plot rho values
ax.plot(dims, rhos, 'o-', color=COLORS['ours'], linewidth=LINE_WIDTH * 1.2,
        markersize=8, zorder=5)

# Highlight d'=10 (the only negative point)
ax.plot(dims[0], rhos[0], 'o', color=COLORS['highlight'], markersize=12,
        markeredgecolor='black', markeredgewidth=1.2, zorder=6)

# Zero line
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

# Annotate each point with p-value
for d, r, p in zip(dims, rhos, p_values):
    offset_y = -0.08 if r > 0 else 0.06
    ax.annotate(f'p={p:.3f}', (d, r + offset_y),
                ha='center', fontsize=FONT_SIZE - 2, color='#555555')

# Annotate the key finding
ax.annotate(r"$d'=10$: $\rho_s = -0.383$" + "\n(theory-consistent)",
            xy=(10, -0.383), xytext=(25, -0.35),
            arrowprops=dict(arrowstyle='->', color=COLORS['highlight'], lw=1.2),
            fontsize=FONT_SIZE - 1, color=COLORS['highlight'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=COLORS['highlight'], alpha=0.9))

# Shaded region where correlation reverses
ax.axhspan(0, 0.4, alpha=0.06, color=COLORS['ablation'])
ax.text(40, 0.32, 'Sign reversal\nregion', fontsize=FONT_SIZE - 2,
        color=COLORS['ablation'], ha='center', style='italic', alpha=0.8)

ax.set_xlabel("Subspace dimension $d'$")
ax.set_ylabel("Spearman $\\rho_s$ (CMI vs. absorption rate)")
ax.set_xticks(dims)
ax.set_xlim(5, 55)
ax.set_ylim(-0.55, 0.45)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
outpath = '/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current/writing/figures/cmi_dimension_sensitivity.pdf'
fig.savefig(outpath)
print(f"Saved: {outpath}")
plt.close()
