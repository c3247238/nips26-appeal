#!/usr/bin/env python3
"""Generate Figure 1: UAD Detection Pipeline - horizontal flow diagram."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4.5)
ax.axis('off')

# Color palette (consistent with other figures)
blue = '#4472C4'
orange = '#ED7D31'
green = '#70AD47'
red = '#C5504B'
light_blue = '#D6E3F8'
light_green = '#E2EFDA'
light_red = '#F8D6D6'

# Step boxes
steps = [
    ("1. Extract\nActivations", "Corpus + SAE\n→ Matrix A", blue, light_blue),
    ("2. Compute\nCo-Occurrence", "AᵀA\n→ Matrix C", blue, light_blue),
    ("3. Phi\nCoefficient", "Normalize\n→ Matrix R", blue, light_blue),
    ("4. Hierarchical\nClustering", "Ward linkage\n→ 50 clusters", blue, light_blue),
    ("5. Candidate\nPairs", "Same-cluster\n→ P_cand", green, light_green),
    ("6. Validate", "Chanin labels\n→ P, R, F1", red, light_red),
]

box_width = 1.8
box_height = 1.4
start_x = 0.6
spacing = 2.2
y_center = 2.8

for i, (title, subtitle, edge_color, face_color) in enumerate(steps):
    x = start_x + i * spacing
    # Main box
    box = FancyBboxPatch((x, y_center - box_height/2), box_width, box_height,
                         boxstyle="round,pad=0.05,rounding_size=0.15",
                         facecolor=face_color, edgecolor=edge_color,
                         linewidth=2)
    ax.add_patch(box)
    # Title
    ax.text(x + box_width/2, y_center + 0.25, title,
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=edge_color)
    # Subtitle
    ax.text(x + box_width/2, y_center - 0.35, subtitle,
            ha='center', va='center', fontsize=7.5, color='#333333')

    # Arrow to next step
    if i < len(steps) - 1:
        ax.annotate('', xy=(x + box_width + 0.35, y_center),
                    xytext=(x + box_width + 0.05, y_center),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

# Key annotation below
ax.text(7, 0.7,
        "No ground truth required for Steps 1–5. Validation (Step 6) uses supervised labels only for evaluation, not for detection.",
        ha='center', va='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8DC', edgecolor='#DAA520', linewidth=1.2))

# Title
ax.text(7, 4.1, 'UAD Detection Pipeline', ha='center', va='center',
        fontsize=14, fontweight='bold', color='#1a1a1a')

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-negative/writing/figures/fig1.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-negative/writing/figures/fig1.png', dpi=300, bbox_inches='tight')
print("Figure 1 saved.")
