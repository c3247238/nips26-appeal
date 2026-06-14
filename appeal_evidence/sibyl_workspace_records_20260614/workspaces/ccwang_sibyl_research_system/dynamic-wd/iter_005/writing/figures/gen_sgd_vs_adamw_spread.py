"""Generate Figure 5: Phi spread comparison across optimizers and datasets."""
import sys
sys.path.insert(0, '.')
from style_config import *
import numpy as np

setup_style()

configs = ['AdamW\nCIFAR-10', 'AdamW\nCIFAR-100', 'SGD\nCIFAR-10', 'SGD\nCIFAR-100']
spreads = [0.25, 0.76, 0.91, 1.71]
rho_labels = ['$\\rho \\approx 0.5$', '$\\rho \\approx 0.5$',
              '$\\rho \\approx 0.005$', '$\\rho \\approx 0.005$']
colors_bar = [COLORS['adamw'], COLORS['adamw'], COLORS['sgd'], COLORS['sgd']]

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

x = np.arange(len(configs))
bars = ax.bar(x, spreads, color=colors_bar, edgecolor='black', linewidth=0.8, alpha=0.85, width=0.6)

# Add rho annotations
for i, (bar, rho) in enumerate(zip(bars, rho_labels)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
            rho, ha='center', va='bottom', fontsize=9, style='italic')

# Add spread values on bars
for i, (bar, spread) in enumerate(zip(bars, spreads)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            f'{spread:.2f}%', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Ratio annotation
ax.annotate('3.7$\\times$', xy=(1.5, 0.91), xytext=(1.5, 1.3),
            fontsize=12, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
ax.text(1.5, 1.38, 'SGD/AdamW ratio\n(CIFAR-10)', ha='center', fontsize=8, color='red')

ax.set_ylabel('$\\Phi_{\\mathrm{spread}}$ (percentage points)')
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10)
ax.set_ylim(0, 2.0)
ax.set_title('WD Method Sensitivity: AdamW vs. SGD')

plt.tight_layout()
plt.savefig('sgd_vs_adamw_spread.pdf')
plt.savefig('sgd_vs_adamw_spread.png')
print("Saved sgd_vs_adamw_spread.pdf and .png")
