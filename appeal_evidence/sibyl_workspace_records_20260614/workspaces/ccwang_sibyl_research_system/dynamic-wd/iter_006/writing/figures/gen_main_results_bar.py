"""Generate main results bar chart: best test accuracy across methods on CIFAR-10."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from style_config import *
import numpy as np

setup_style()

methods = ['no_wd', 'constant', 'cosine_schedule', 'cwd_hard', 'swd', 'pmpwd']
# CIFAR-10 best test accuracy (3 seeds each)
data = {
    'no_wd':           [90.10, 89.99, 90.21],
    'constant':        [89.72, 90.15, 89.54],
    'cosine_schedule': [90.00, 89.94, 89.77],
    'cwd_hard':        [89.71, 89.78, 90.46],
    'swd':             [90.08, 89.98, 90.37],
    'pmpwd':           [90.16, 90.34, 90.38],
}

means = [np.mean(data[m]) for m in methods]
stds = [np.std(data[m]) for m in methods]
labels = [METHOD_LABELS[m] for m in methods]
colors = [COLORS[m] for m in methods]

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

x = np.arange(len(methods))
bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

# Add value labels
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.05, f'{m:.2f}', ha='center', va='bottom', fontsize=FONT_SIZE - 1)

ax.set_ylabel('Best Test Accuracy (%)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha='right')
ax.set_ylim(89.0, 91.0)

# Add narrow-band annotation
ax.axhspan(min(means) - 0.1, max(means) + 0.1, alpha=0.08, color='blue', zorder=0)
ax.annotate('0.49pp spread', xy=(4.5, max(means) + 0.12), fontsize=FONT_SIZE - 2, 
            color='#1565C0', style='italic')

ax.set_title('CIFAR-10 / ResNet-20: Best Test Accuracy (3 seeds)')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'main_results_bar.pdf'))
print("Saved main_results_bar.pdf")
