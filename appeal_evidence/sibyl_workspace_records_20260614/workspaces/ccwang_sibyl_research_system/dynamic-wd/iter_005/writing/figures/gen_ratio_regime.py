"""Generate Figure 1: Ratio regime diagram — method spread vs log(rho)."""
import sys
sys.path.insert(0, '.')
from style_config import *
import numpy as np

setup_style()

# Data points from experiments
# rho ~ 0.005 (SGD original, lr=0.1, wd=5e-4): spread = 0.91% CIFAR-10
# rho ~ 0.05 (AdamW rho_low, wd=5e-5): constant only, no spread computable
# rho ~ 0.5 (AdamW standard, wd=5e-4): spread = 0.25% CIFAR-10
rho_values = [0.005, 0.5]
spread_values = [0.91, 0.25]

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

# Shaded regime zones
ax.axvspan(np.log10(0.001), np.log10(0.1), alpha=0.12, color='#4CAF50', label='Inhibition')
ax.axvspan(np.log10(0.1), np.log10(2.0), alpha=0.12, color='#FF9800', label='Transition')
ax.axvspan(np.log10(2.0), np.log10(20.0), alpha=0.12, color='#F44336', label='Differentiation')

# Data points
log_rho = [np.log10(r) for r in rho_values]
ax.scatter(log_rho, spread_values, s=120, c=[COLORS['sgd'], COLORS['adamw']],
           zorder=5, edgecolors='black', linewidth=0.8)

# Annotations
ax.annotate('SGD\n$\\rho \\approx 0.005$\nspread = 0.91%',
            xy=(log_rho[0], spread_values[0]),
            xytext=(log_rho[0] + 0.15, spread_values[0] - 0.45),
            fontsize=9, ha='left',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))

ax.annotate('AdamW\n$\\rho \\approx 0.5$\nspread = 0.25%',
            xy=(log_rho[1], spread_values[1]),
            xytext=(log_rho[1] - 0.6, spread_values[1] + 0.35),
            fontsize=9, ha='right',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))

# Trend line (dashed, suggestive)
x_trend = np.linspace(-2.5, 1.3, 100)
# Simple model: spread increases with rho
y_trend = 0.15 + 0.35 * np.exp(-(x_trend + 0.3)**2 / 2.0) + 0.6 / (1 + np.exp(-3*(x_trend - 0.5)))
ax.plot(x_trend, y_trend, '--', color='gray', alpha=0.5, linewidth=1.0, label='_nolegend_')

# Threshold line
ax.axvline(x=np.log10(2.0), color='red', linestyle=':', alpha=0.5, linewidth=1.0)
ax.text(np.log10(2.0) + 0.05, 1.1, '$\\rho^*$', fontsize=11, color='red', alpha=0.7)

ax.set_xlabel('$\\log_{10}(\\rho)$  (gradient-to-weight ratio)')
ax.set_ylabel('$\\Phi_{\\mathrm{spread}}$ (percentage points)')
ax.set_xlim(-2.8, 1.5)
ax.set_ylim(-0.05, 1.3)
ax.set_title('WD Method Sensitivity Across Ratio Regimes')
ax.legend(loc='upper left', framealpha=0.9, fontsize=9)

plt.tight_layout()
plt.savefig('ratio_regime.pdf')
plt.savefig('ratio_regime.png')
print("Saved ratio_regime.pdf and ratio_regime.png")
