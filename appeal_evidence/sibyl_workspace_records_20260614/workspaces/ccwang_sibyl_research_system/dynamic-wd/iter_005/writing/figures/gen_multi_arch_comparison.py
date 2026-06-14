"""Generate Figure 4: Multi-architecture accuracy comparison (ResNet-20 and VGG-16-BN)."""
import sys
sys.path.insert(0, '.')
from style_config import *
import numpy as np

setup_style()

methods = METHOD_ORDER
method_labels = [METHOD_NAMES[m] for m in methods]

# ResNet-20 CIFAR-10 AdamW (iter_003)
resnet_means = [90.13, 90.12, 90.06, 89.88, 90.09, 90.12, 90.08]
resnet_stds  = [0.31,  0.07,  0.24,  0.25,  0.28,  0.30,  0.31]

# VGG-16-BN CIFAR-10 AdamW
vgg_means = [92.05, 91.99, 92.06, 92.11, 92.15, 92.05, 92.03]
vgg_stds  = [0.06,  0.32,  0.26,  0.28,  0.13,  0.27,  0.04]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(FIG_WIDTH_FULL * 0.8, FIG_HEIGHT))

bars1 = ax.bar(x - width/2, resnet_means, width, yerr=resnet_stds,
               label='ResNet-20 (270K)', color=COLORS['adamw'], alpha=0.8,
               capsize=3, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, vgg_means, width, yerr=vgg_stds,
               label='VGG-16-BN (15M)', color=COLORS['sgd'], alpha=0.8,
               capsize=3, edgecolor='black', linewidth=0.5)

# Constant WD reference lines
ax.axhline(y=90.13, color=COLORS['adamw'], linestyle='--', alpha=0.4, linewidth=1.0)
ax.axhline(y=92.05, color=COLORS['sgd'], linestyle='--', alpha=0.4, linewidth=1.0)

ax.set_xlabel('WD Method')
ax.set_ylabel('Best Test Accuracy (%)')
ax.set_xticks(x)
ax.set_xticklabels(method_labels, rotation=15, ha='right')
ax.legend(loc='lower left', framealpha=0.9)

# Annotate Phi spreads
ax.text(6.3, 90.13, f'ResNet spread: 0.25%', fontsize=8, color=COLORS['adamw'], va='center')
ax.text(6.3, 92.05, f'VGG spread: 0.16%', fontsize=8, color=COLORS['sgd'], va='center')

# Use broken y-axis effect by setting limits
ax.set_ylim(89.4, 92.6)
ax.set_title('Accuracy Comparison Across Architectures (CIFAR-10, AdamW)')

plt.tight_layout()
plt.savefig('multi_arch_comparison.pdf')
plt.savefig('multi_arch_comparison.png')
print("Saved multi_arch_comparison.pdf and .png")
