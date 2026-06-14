"""Generate Figure 2: Theorem 1 Regime Illustration."""
import sys
sys.path.insert(0, '.')
from style_config import setup_style, COLORS, FONT_SIZE, FIG_WIDTH, DPI

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

setup_style()

fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.8))

# Parameters
ais = np.linspace(0, 1, 500)
lambda_bar = 0.6
stability_cost = 0.15  # constant line

# Alignment benefit curve: monotonically increasing, concave
alignment_benefit = lambda_bar * ais**0.7 * 0.45

# Find crossing point
idx_cross = np.argmin(np.abs(alignment_benefit - stability_cost))
ais_star = ais[idx_cross]

# Shaded regions
ax.fill_between(ais[:idx_cross+1], -0.05, 0.5,
                alpha=0.08, color='#F44336', zorder=0)
ax.fill_between(ais[idx_cross:], -0.05, 0.5,
                alpha=0.08, color='#4CAF50', zorder=0)

# Curves
ax.plot(ais, alignment_benefit, color='#2196F3', linewidth=2.2,
        label=r'Alignment benefit $\propto \mathrm{AIS} \cdot \bar{\lambda}$',
        zorder=3)
ax.axhline(y=stability_cost, color='#F44336', linewidth=2.0,
           linestyle='--',
           label=r'Stability cost $\propto \frac{C\sigma^2}{n} \cdot \Delta\mathrm{CSI} / \bar{\lambda}$',
           zorder=3)

# AIS* threshold vertical line
ax.axvline(x=ais_star, color='#555555', linewidth=1.2, linestyle=':',
           zorder=2)
ax.annotate(r'$\mathrm{AIS}^*$',
            xy=(ais_star, -0.04), fontsize=FONT_SIZE + 1,
            ha='center', va='top', fontweight='bold', color='#333333')

# Region labels
ax.text(ais_star / 2, 0.28, 'Constant WD\noptimal',
        ha='center', va='top', fontsize=FONT_SIZE,
        color='#C62828', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#F44336', alpha=0.9))

ax.text((1 + ais_star) / 2, 0.42, 'Adaptive WD\noptimal',
        ha='center', va='top', fontsize=FONT_SIZE,
        color='#2E7D32', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#4CAF50', alpha=0.9))

# CIFAR-10 BN experiments annotation
cifar_ais = np.array([0.18, 0.25, 0.32, 0.40])
cifar_y = lambda_bar * cifar_ais**0.7 * 0.45
ax.scatter(cifar_ais, cifar_y, color='#E91E63', s=50, zorder=5,
           edgecolors='white', linewidths=0.8, marker='D')

# Bracket / annotation for CIFAR-10 cluster
ax.annotate('CIFAR-10 BN\nexperiments',
            xy=(0.29, np.mean(cifar_y) - 0.02),
            xytext=(0.55, 0.06),
            fontsize=FONT_SIZE - 1,
            ha='center',
            arrowprops=dict(arrowstyle='->', color='#E91E63',
                            lw=1.3, connectionstyle='arc3,rad=-0.2'),
            color='#E91E63', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#E91E63', alpha=0.9))

# Crossing point marker
ax.plot(ais_star, stability_cost, 'ko', markersize=7, zorder=5)

# Axes
ax.set_xlabel('AIS (Alignment Informativeness Score)', fontsize=FONT_SIZE + 1)
ax.set_ylabel('Net benefit / cost', fontsize=FONT_SIZE + 1)
ax.set_xlim(0, 1)
ax.set_ylim(-0.05, 0.5)
ax.legend(loc='upper left', fontsize=FONT_SIZE - 1, framealpha=0.95)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig('theorem1_regime.pdf', bbox_inches='tight')
fig.savefig('theorem1_regime.png', bbox_inches='tight', dpi=DPI)
print("Saved theorem1_regime.pdf and .png")
plt.close()
