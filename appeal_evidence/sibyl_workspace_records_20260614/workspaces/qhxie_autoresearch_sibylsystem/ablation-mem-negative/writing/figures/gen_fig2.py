#!/usr/bin/env python3
"""Generate Figure 2: UAD Performance Summary - grouped bar chart."""

import matplotlib.pyplot as plt
import numpy as np

# Data from experiments
conditions = ['Pilot\n(layer 8)', 'Full\n(layer 8)', 'Multi-seed\n(mean)', 'Cross-layer\n(mean)']
precision = [0.543, 0.569, 0.569, 0.335]  # cross-layer: avg of 0.276, 0.543, 0.377
recall = [1.0, 1.0, 1.0, 1.0]
f1 = [0.704, 0.725, 0.725, 0.561]

x = np.arange(len(conditions))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(x - width, precision, width, label='Precision', color='#4472C4', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, recall, width, label='Recall', color='#ED7D31', edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width, f1, width, label='F1', color='#70AD47', edgecolor='black', linewidth=0.5)

# Target threshold line
ax.axhline(y=0.6, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='F1 target (0.6)')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('UAD Performance Across Experimental Conditions', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=10)
ax.set_ylim(0, 1.15)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.5)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-negative/writing/figures/fig2.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-negative/writing/figures/fig2.png', dpi=300, bbox_inches='tight')
print("Figure 2 saved.")
