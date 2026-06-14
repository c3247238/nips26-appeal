"""Generate BEM vs accuracy scatter plot."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from style_config import *
import numpy as np

setup_style()

# Data from instrumented experiments (means across 3 seeds)
methods = ['no_wd', 'constant', 'cosine_schedule', 'cwd_hard', 'swd', 'pmpwd']
bem_values = [1.000, 0.000, 0.502, 0.507, 0.900, 0.508]
best_acc = [90.10, 89.80, 89.90, 89.98, 90.14, 90.29]
acc_std = [0.15, 0.31, 0.12, 0.41, 0.20, 0.12]

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

for i, m in enumerate(methods):
    ax.errorbar(bem_values[i], best_acc[i], yerr=acc_std[i], 
                fmt='o', color=COLORS[m], markersize=10, capsize=4,
                label=METHOD_LABELS[m], zorder=5)

# Fit and show trend line
z = np.polyfit(bem_values, best_acc, 1)
p = np.poly1d(z)
x_line = np.linspace(-0.05, 1.05, 100)
ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.5, linewidth=1)

# Annotate correlation
from scipy import stats
r, pval = stats.pearsonr(bem_values, best_acc)
ax.text(0.02, 89.5, f'$r = {r:.2f}$, $p = {pval:.2f}$', fontsize=FONT_SIZE - 1, color='gray')

ax.set_xlabel('BEM (Budget Equivalence Metric)')
ax.set_ylabel('Best Test Accuracy (%)')
ax.set_title('BEM vs. Accuracy (CIFAR-10, ResNet-20)')
ax.legend(fontsize=FONT_SIZE - 2, loc='lower right')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(89.2, 90.8)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'bem_accuracy_scatter.pdf'))
print("Saved bem_accuracy_scatter.pdf")
