#!/usr/bin/env python3
"""Generate Figure 4: DFDA Per-Pair Results with caveat annotation."""

import matplotlib.pyplot as plt
import numpy as np

# Data from f5_dfda_scale_results.json
pairs = [f'Pair {i+1}' for i in range(8)]
baseline_mse = [0.102, 9.993, 39.183, 0.100, 0.430, 4.654, 3.552, 0.230]
compensated_mse = [1.5e-10, 5.6e-10, 2.2e-08, 8.7e-12, 0.016, 8.7e-07, 5.5e-11, 8.8e-10]
improvement = [99.999, 99.999, 99.999, 99.999, 96.225, 99.999, 99.999, 99.999]

x = np.arange(len(pairs))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))

# Use log scale for MSE
bars1 = ax.bar(x - width/2, baseline_mse, width, label='Baseline MSE', color='#C55A11', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, compensated_mse, width, label='Compensated MSE', color='#70AD47', edgecolor='black', linewidth=0.5)

ax.set_yscale('log')
ax.set_ylabel('MSE (log scale)', fontsize=12)
ax.set_title('DFDA Per-Pair MSE Reduction (Preliminary Results)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(pairs, fontsize=10)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Add improvement percentage annotations
for i, imp in enumerate(improvement):
    ax.annotate(f'{imp:.1f}%',
                xy=(i, max(baseline_mse[i], compensated_mse[i])),
                xytext=(0, 8),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkgreen')

# Caveat box
caveat_text = (
    "CAVEAT: Metric measures near-zero prediction on child-dominant examples,\n"
    "not true absorption recovery on parent-positive examples."
)
props = dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=2)
ax.text(0.5, 0.15, caveat_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='center', horizontalalignment='center', bbox=props, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-negative/writing/figures/fig4.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-negative/writing/figures/fig4.png', dpi=300, bbox_inches='tight')
print("Figure 4 saved.")
