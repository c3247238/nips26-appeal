#!/usr/bin/env python3
"""Generate Figure 1: Experimental Pipeline Overview."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis('off')

# Colors
colors = {
    'p1': '#E3F2FD',  # blue tint
    'p2': '#E8F5E9',  # green tint
    'p3': '#FFF3E0',  # orange tint
    'p4': '#FFEBEE',  # red tint
    'border1': '#2E86AB',
    'border2': '#4CAF50',
    'border3': '#F18F01',
    'border4': '#C73E1D',
    'highlight': '#C73E1D',
}

# Phase boxes
phases = [
    (1.5, 2.8, 'Phase 1\nAbsorption\nDetection', colors['p1'], colors['border1']),
    (4.8, 2.8, 'Phase 2\nFeature\nSteering', colors['p2'], colors['border2']),
    (8.1, 2.8, 'Phase 3\nSparse\nProbing', colors['p3'], colors['border3']),
    (11.4, 2.8, 'Phase 4\nCorrelation\nAnalysis', colors['p4'], colors['border4']),
]

for x, y, text, facecolor, edgecolor in phases:
    box = FancyBboxPatch((x-1.1, y-0.7), 2.2, 1.4,
                         boxstyle="round,pad=0.05,rounding_size=0.15",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9,
            fontweight='bold', color='#333333')

# Data boxes below phases
data_boxes = [
    (1.5, 1.5, r'$A(f)$ per feature', colors['border1']),
    (4.8, 1.5, r'$S(f, s)$, $\Delta S(f, s)$', colors['border2']),
    (8.1, 1.5, r'F1$(f, k)$ per $k$', colors['border3']),
    (11.4, 1.5, r'$r$, $p$, $R^2$, $\beta$', colors['border4']),
]

for x, y, text, edgecolor in data_boxes:
    box = FancyBboxPatch((x-1.0, y-0.25), 2.0, 0.5,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor='white', edgecolor=edgecolor,
                         linewidth=1.0, linestyle='--')
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=7.5,
            color='#555555')

# Arrows between phases
arrow_style = dict(arrowstyle='->', color='#333333', lw=1.5,
                   connectionstyle='arc3,rad=0')
for i in range(3):
    x1 = phases[i][0] + 1.1
    x2 = phases[i+1][0] - 1.1
    y = 2.8
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

# Arrow labels
ax.text(3.15, 3.15, 'feature IDs', ha='center', va='center', fontsize=7, color='#666666')
ax.text(6.45, 3.15, 'same features', ha='center', va='center', fontsize=7, color='#666666')
ax.text(9.75, 3.15, 'task metrics', ha='center', va='center', fontsize=7, color='#666666')

# Vertical dashed lines (phase to data)
for x, y, _, _, _ in phases:
    ax.plot([x, x], [y - 0.7, y - 0.95], 'k--', alpha=0.3, lw=0.8)

# Input arrow
ax.text(0.2, 2.8, 'Pre-trained SAE\n(SAELens)', ha='center', va='center',
        fontsize=8, color='#555555')
ax.annotate('', xy=(0.4, 2.8), xytext=(1.1, 2.8),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

# Output arrow
ax.text(13.5, 2.8, 'H1--H3\ntested', ha='center', va='center',
        fontsize=8, color='#555555')
ax.annotate('', xy=(13.3, 2.8), xytext=(12.5, 2.8),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

# Random baseline highlight on Phase 2
ax.text(4.8, 3.75, 'Random baseline\ncontrol', ha='center', va='center',
        fontsize=7, color=colors['highlight'], fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3F3',
                  edgecolor=colors['highlight'], alpha=0.8))
ax.annotate('', xy=(4.8, 3.45), xytext=(4.8, 3.55),
            arrowprops=dict(arrowstyle='->', color=colors['highlight'], lw=1.2))

# Title
ax.text(7, 4.5, 'Four-Phase Experimental Pipeline', ha='center', va='center',
        fontsize=13, fontweight='bold', color='#333333')

# Bottom caption box
caption_box = FancyBboxPatch((0.5, 0.2), 13, 0.5,
                              boxstyle="round,pad=0.05,rounding_size=0.1",
                              facecolor='#F5F5F5', edgecolor='#999999',
                              linewidth=0.8)
ax.add_patch(caption_box)
ax.text(7, 0.45,
        'Training-free methodology: pre-trained SAEs → absorption detection → '
        'task evaluation (with random baseline control) → statistical hypothesis testing',
        ha='center', va='center', fontsize=8, color='#555555')

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/writing/figures/fig1_pipeline.pdf',
            format='pdf')
plt.close()
print('Figure 1 saved.')
