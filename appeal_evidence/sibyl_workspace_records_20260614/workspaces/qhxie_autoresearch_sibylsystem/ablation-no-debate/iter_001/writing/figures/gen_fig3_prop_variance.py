#!/usr/bin/env python3
"""
Generate Figure 3: Proportional variance of child contributions by condition.
Shows trained SAE has more asymmetric child contributions.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from pilot_summary.md
conditions = ['Trained SAE', 'Random Decoder', 'Shuffled Features', 'Permuted Encoder']
means = [0.1154, 0.0040, 0.1057, 0.1031]
stds = [0.0072, 0.0011, 0.0281, 0.0304]

colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(conditions))
bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.8)

# Highlight trained SAE
bars[0].set_edgecolor('#1a5f7a')
bars[0].set_linewidth(2)

ax.set_ylabel('Proportional Variance', fontsize=12)
ax.set_xlabel('Condition', fontsize=12)
ax.set_title('Proportional Variance of Child Contributions\nTrained SAE Shows Asymmetric Substitution', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=11)
ax.set_ylim(0, 0.17)

# Value labels
for bar, val, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add annotation
ax.annotate('Random decoder:\nnear-zero variance\n(no structure)', xy=(1, 0.005), xytext=(1.6, 0.045),
            fontsize=9, ha='left',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current/writing/figures/fig3_prop_variance.pdf', dpi=300, bbox_inches='tight')
print("Figure 3 saved.")
