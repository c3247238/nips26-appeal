"""
Regenerate Figures 3 and 4 for the Phi Invariance paper.
Uses only the 7 methods from the paper's method catalog (Table 1).
No PMP-WD. Data sourced from iter_003 (Tables 2, 4, 5) and iter_005 (VGG).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# 7 methods from paper Table 1
METHODS = ['constant', 'cosine_schedule', 'cwd_hard', 'swd', 'half_lambda', 'random_mask', 'no_wd']
LABELS = ['Constant', 'Cosine', 'CWD', 'SWD', r'Half-$\lambda$', 'Random', 'No WD']
COLORS_LIST = ['#1f77b4', '#ff7f0e', '#8c564b', '#e377c2', '#d62728', '#2ca02c', '#9467bd']

# Data from iter_003 (Table 2: AdamW + ResNet-20, CIFAR-10)
resnet_adamw_means = [90.13, 90.12, 90.06, 89.88, 90.09, 90.12, 90.08]
resnet_adamw_stds  = [0.31,  0.07,  0.24,  0.25,  0.28,  0.30,  0.31]

# Data from iter_005 (Table 5: SGD + VGG-16-BN, CIFAR-10)
vgg_sgd_means = [92.05, 91.99, 92.06, 92.11, 92.15, 92.05, 92.03]
vgg_sgd_stds  = [0.05,  0.26,  0.21,  0.23,  0.11,  0.22,  0.04]

# BEM values from Table 6
bem_values = [0.000, 0.503, 0.503, 0.900, 0.500, 0.500, 1.000]

# CIFAR-10 AdamW accuracy for BEM scatter
cifar10_means = resnet_adamw_means
cifar10_stds  = resnet_adamw_stds

# CIFAR-100 AdamW data (iter_003 Table 2)
cifar100_means = [63.01, 63.44, 62.67, 63.09, 63.19, 63.33, 62.69]
cifar100_stds  = [0.38,  0.06,  0.79,  0.46,  0.24,  0.66,  0.53]

outdir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Figure 3: Multi-architecture comparison
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

x = np.arange(len(METHODS))
width = 0.6

# Panel (a): ResNet-20 / CIFAR-10 / AdamW
bars1 = ax1.bar(x, resnet_adamw_means, width, yerr=resnet_adamw_stds,
                color=COLORS_LIST, alpha=0.85, capsize=3, edgecolor='black', linewidth=0.5)
ax1.axhline(y=90.13, color='gray', linestyle='--', alpha=0.4, linewidth=1.0)
ax1.set_xlabel('WD Method')
ax1.set_ylabel('Best Test Accuracy (%)')
ax1.set_xticks(x)
ax1.set_xticklabels(LABELS, rotation=20, ha='right')
ax1.set_ylim(89.4, 90.7)
ax1.set_title('(a) ResNet-20 / CIFAR-10 / AdamW')
spread_a = max(resnet_adamw_means) - min(resnet_adamw_means)
ax1.text(0.98, 0.95, f'Spread: {spread_a:.2f}pp', transform=ax1.transAxes,
         fontsize=9, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# Panel (b): VGG-16-BN / CIFAR-10 / SGD
bars2 = ax2.bar(x, vgg_sgd_means, width, yerr=vgg_sgd_stds,
                color=COLORS_LIST, alpha=0.85, capsize=3, edgecolor='black', linewidth=0.5)
ax2.axhline(y=92.05, color='gray', linestyle='--', alpha=0.4, linewidth=1.0)
ax2.set_xlabel('WD Method')
ax2.set_ylabel('Best Test Accuracy (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(LABELS, rotation=20, ha='right')
ax2.set_ylim(91.5, 92.6)
ax2.set_title('(b) VGG-16-BN / CIFAR-10 / SGD')
spread_b = max(vgg_sgd_means) - min(vgg_sgd_means)
ax2.text(0.98, 0.95, f'Spread: {spread_b:.2f}pp', transform=ax2.transAxes,
         fontsize=9, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(outdir, 'multi_arch_comparison.png'), bbox_inches='tight')
plt.savefig(os.path.join(outdir, 'multi_arch_comparison.pdf'), bbox_inches='tight')
print("Saved multi_arch_comparison.png/pdf")
plt.close()

# ============================================================
# Figure 4: BEM vs. accuracy scatter
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

for i, m in enumerate(METHODS):
    ax1.errorbar(bem_values[i], cifar10_means[i], yerr=cifar10_stds[i],
                 fmt='o', color=COLORS_LIST[i], markersize=8, capsize=4, label=LABELS[i],
                 markeredgecolor='black', markeredgewidth=0.5)
    ax2.errorbar(bem_values[i], cifar100_means[i], yerr=cifar100_stds[i],
                 fmt='o', color=COLORS_LIST[i], markersize=8, capsize=4, label=LABELS[i],
                 markeredgecolor='black', markeredgewidth=0.5)

# Trend lines
z1 = np.polyfit(bem_values, cifar10_means, 1)
p1 = np.poly1d(z1)
xr = np.linspace(-0.05, 1.05, 50)
ax1.plot(xr, p1(xr), '--', color='gray', alpha=0.5, linewidth=1)
ax1.set_xlabel('Budget Equivalence Metric (BEM)')
ax1.set_ylabel('Best Test Accuracy (%)')
ax1.set_title(f'(a) CIFAR-10 (slope = {z1[0]:.3f}%/BEM)')
ax1.legend(loc='lower left', fontsize=8, ncol=2)
ax1.set_xlim(-0.05, 1.1)
ax1.set_ylim(89.4, 90.7)

z2 = np.polyfit(bem_values, cifar100_means, 1)
p2 = np.poly1d(z2)
ax2.plot(xr, p2(xr), '--', color='gray', alpha=0.5, linewidth=1)
ax2.set_xlabel('Budget Equivalence Metric (BEM)')
ax2.set_ylabel('Best Test Accuracy (%)')
ax2.set_title(f'(b) CIFAR-100 (slope = {z2[0]:.3f}%/BEM)')
ax2.legend(loc='lower left', fontsize=8, ncol=2)
ax2.set_xlim(-0.05, 1.1)
ax2.set_ylim(62.0, 64.0)

plt.tight_layout()
plt.savefig(os.path.join(outdir, 'fig3_bem_vs_accuracy.png'), bbox_inches='tight')
plt.savefig(os.path.join(outdir, 'fig3_bem_vs_accuracy.pdf'), bbox_inches='tight')
print("Saved fig3_bem_vs_accuracy.png/pdf")
plt.close()

print("All figures regenerated successfully with correct 7-method data.")
