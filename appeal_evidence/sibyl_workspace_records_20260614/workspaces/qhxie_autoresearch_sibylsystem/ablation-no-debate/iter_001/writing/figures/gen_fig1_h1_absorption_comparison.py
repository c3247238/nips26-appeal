#!/usr/bin/env python3
"""
Generate Figure 1: Multi-child proportional absorption rates by condition.
H1 main results - bar chart comparing trained SAE vs three baselines.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from h1_statistics.json
conditions = ['Trained SAE', 'Random Decoder', 'Shuffled Features', 'Permuted Encoder']
absorption_rates = [0.5000, 0.0590, 0.4870, 0.4840]
stds = [0.0000, 0.0694, 0.0336, 0.0367]

# Colors: trained SAE highlighted, baselines in muted colors
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(conditions))
bars = ax.bar(x, absorption_rates, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.8)

# Highlight trained SAE
bars[0].set_edgecolor('#1a5f7a')
bars[0].set_linewidth(2)

ax.set_ylabel('Absorption Rate', fontsize=12)
ax.set_xlabel('Condition', fontsize=12)
ax.set_title('H1: Multi-Child Proportional Absorption\nTrained SAE vs. Baseline Methods', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=11)
ax.set_ylim(0, 0.65)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Trained SAE level')

# Add value labels on bars
for bar, val in zip(bars, absorption_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add significance annotation
ax.annotate('', xy=(0, 0.52), xytext=(1, 0.52),
            arrowprops=dict(arrowstyle='-', color='black', lw=1.5))
ax.text(0.5, 0.545, f'd=8.94\np<10⁻¹³³', ha='center', va='bottom', fontsize=9, color='black')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current/writing/figures/fig1_h1_absorption_comparison.pdf', dpi=300, bbox_inches='tight')
print("Figure 1 saved.")
