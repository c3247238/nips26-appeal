#!/usr/bin/env python3
"""Generate Figure 5: H3 falsification — single-pass routing failure."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(5, 4))

categories = ['Low-$E_a$\n(n=19)', 'High-$E_a$\n(n=11)']
accuracies = [68.4, 63.6]
colors = ['#2E86AB', '#E94F37']

bars = ax.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=0.8, width=0.5)

# Threshold line
ax.axhline(y=75.0, color='#F5A623', linestyle='--', linewidth=2.0, alpha=0.9,
           label='75% routing threshold')

# Value labels
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, acc + 1.0,
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# "FALSIFIED" annotation
ax.annotate('H3 FALSIFIED', xy=(0.5, 68.4), xytext=(1.5, 55),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=11, color='red', fontweight='bold',
            ha='center')

ax.set_ylabel('Single-Pass Accuracy (%)', fontsize=12)
ax.set_title('H3: Ea-Based Routing Fails\n(Low-$E_a$ accuracy 68.4% < 75% threshold)', fontsize=11)
ax.set_ylim(0, 90)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# Add count annotations
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, 3,
            f'({accuracies.index(acc) == 0 and "n=19" or "n=11"})',
            ha='center', va='bottom', fontsize=9, color='white')

outpath = '/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/writing/figures/fig5_h3_routing_failure.pdf'
plt.tight_layout()
plt.savefig(outpath, dpi=150)
print(f"Saved: {outpath}")
