#!/usr/bin/env python3
"""Generate Figure 3: Cross-Layer F1 Comparison."""

import matplotlib.pyplot as plt
import numpy as np

layers = ['Layer 4', 'Layer 8', 'Layer 10']
f1_scores = [0.432, 0.704, 0.548]
precision = [0.276, 0.543, 0.377]
recall = [1.0, 1.0, 1.0]

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(layers, f1_scores, color=['#C55A11', '#70AD47', '#FFC000'], edgecolor='black', linewidth=0.5, width=0.6)

# Threshold line
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Minimum threshold (0.5)')

# Add precision/recall annotations
for i, (layer, f1, prec, rec) in enumerate(zip(layers, f1_scores, precision, recall)):
    ax.annotate(f'F1={f1:.3f}\nP={prec:.3f}, R={rec:.3f}',
                xy=(i, f1),
                xytext=(0, 8),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('UAD Performance Across Layers (GPT-2 Small)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-negative/writing/figures/fig3.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-negative/writing/figures/fig3.png', dpi=300, bbox_inches='tight')
print("Figure 3 saved.")
