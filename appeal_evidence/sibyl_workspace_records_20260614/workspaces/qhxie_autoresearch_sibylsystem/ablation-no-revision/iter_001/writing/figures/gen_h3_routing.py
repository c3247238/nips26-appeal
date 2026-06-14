#!/usr/bin/env python3
"""
Figure 5: H3 Single-Pass Routing Failure
Bar chart showing Ea-based routing fails to achieve 75% threshold
"""

import numpy as np
import matplotlib.pyplot as plt

# Data from pilot_summary_round4.json
categories = ['Low-$E_a$', 'High-$E_a$']
accuracies = [68.4, 63.6]
threshold = 75.0

# Create figure
fig, ax = plt.subplots(figsize=(6, 5))

# Bar positions
x = np.arange(len(categories))
bar_width = 0.5

# Create bars
bars = ax.bar(x, accuracies, width=bar_width, color=['#f87171', '#fb923c'],
               edgecolor='white', linewidth=2, alpha=0.8)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.annotate(f'{acc:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add threshold line
ax.axhline(y=threshold, color='green', linestyle='--', linewidth=2.5, alpha=0.8)
ax.text(-0.35, threshold + 1.5, f'75% threshold', fontsize=11, color='green', fontweight='bold')

# Labels and title
ax.set_ylabel('Single-Pass Accuracy (%)', fontsize=12)
ax.set_title('H3: Ea-Based Routing Fails\n(Low-$E_a$ problems: 68.4% < 75% threshold)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)

# Configure axes
ax.set_ylim(0, 90)
ax.set_xlim(-0.6, 1.6)
ax.grid(True, alpha=0.3, axis='y')

# Add X mark on threshold line
ax.scatter([0.25], [threshold], marker='x', s=200, c='green', linewidths=3, zorder=5)

# Add falsification badge
ax.text(0.95, 0.95, 'FALSIFIED', transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='#ef4444', alpha=0.8),
        color='white', fontweight='bold')

# Add explanation
explanation = "Ea measures consistency,\nnot correctness"
ax.text(0.5, 0.15, explanation, transform=ax.transAxes, fontsize=10,
        ha='center', verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/iter_001/writing/figures/h3_routing.pdf', dpi=150, bbox_inches='tight')
print("Saved: h3_routing.pdf")
