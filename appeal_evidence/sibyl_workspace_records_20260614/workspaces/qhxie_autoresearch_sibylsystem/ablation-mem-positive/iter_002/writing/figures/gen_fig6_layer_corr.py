#!/usr/bin/env python3
"""Generate Figure 6: Layer-Dependent Correlation Pattern — GPT-2"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/current/exp/results/pilots/h2v2_analysis/h2v2_analysis.json', 'r') as f:
    data = json.load(f)

layers = data['layer_pattern_analysis']['layers']
rhos = data['layer_pattern_analysis']['rho_projection']
pvals = data['layer_pattern_analysis']['pval_projection']

# Create figure
fig, ax = plt.subplots(figsize=(6.5, 4.5))

# Color by significance
colors = ['#ED7D31' if p < 0.05 else '#A5A5A5' for p in pvals]

# Line plot
ax.plot(layers, rhos, 'o-', color='#4472C4', linewidth=2, markersize=10,
        markerfacecolor='white', markeredgewidth=2, zorder=2)

# Fill significant points
for layer, rho, p, color in zip(layers, rhos, pvals, colors):
    if p < 0.05:
        ax.plot(layer, rho, 'o', color=color, markersize=10, zorder=3)

# Zero line
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

# Significance region shading
ax.axhspan(-0.2, 0.2, alpha=0.1, color='gray', label='Near-zero region')

# Labels for each point
for layer, rho, p in zip(layers, rhos, pvals):
    offset = 0.12 if rho > 0 else -0.18
    sig = '*' if p < 0.05 else 'ns'
    ax.annotate(f'rho = {rho:+.3f}\n(p = {p:.3f}) {sig}',
                xy=(layer, rho), xytext=(layer, rho + offset),
                ha='center', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

# Labels and styling
ax.set_xlabel('Layer', fontsize=11)
ax.set_ylabel(r'Spearman $\rho$ ($A_j$ vs $A_{\mathrm{proj}}$)', fontsize=11)
ax.set_title(r'Layer-Dependent $A_j$ Correlation Pattern (GPT-2 ReLU)', fontsize=12, fontweight='bold')
ax.set_xticks(layers)
ax.set_ylim(-1.0, 1.0)
ax.grid(alpha=0.3)
ax.legend(loc='lower left', fontsize=9)

# Add interpretation text
ax.text(0.02, 0.02, 'Sign flip: negative at layers 5 and 10,\npositive at layer 8 (mid-layer peak)',
        transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.6))

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/current/writing/figures/fig6_layer_corr.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/current/writing/figures/fig6_layer_corr.png', dpi=300, bbox_inches='tight')
print("Figure 6 saved.")
