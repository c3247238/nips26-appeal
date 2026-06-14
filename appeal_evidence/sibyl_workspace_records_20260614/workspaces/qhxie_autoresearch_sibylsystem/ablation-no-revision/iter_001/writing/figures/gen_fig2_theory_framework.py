#!/usr/bin/env python3
"""Generate Figure 2: Theory Framework Diagram — mapping chemical kinetics to LLM reasoning."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_title('Activation Energy Theory: Chemical Kinetics ↔ LLM Reasoning', fontsize=14, fontweight='bold', pad=20)

# Left box: Chemical Kinetics
left_box = FancyBboxPatch((0.3, 1.0), 4.2, 5.5, boxstyle="round,pad=0.1",
                           facecolor='#E8F4F8', edgecolor='#2E86AB', linewidth=2)
ax.add_patch(left_box)
ax.text(2.4, 6.2, 'Chemical Kinetics', ha='center', va='center', fontsize=12,
        fontweight='bold', color='#2E86AB')

# Chemical terms
chem_terms = [
    ('Activation energy $E_a$', 5.3),
    ('Temperature $T$', 4.6),
    ('Catalyst', 3.9),
    ('Pre-exponential factor $A$', 3.2),
    ('Reaction rate', 2.5),
    ('Arrhenius equation', 1.8),
]
for term, y in chem_terms:
    ax.text(0.7, y, '•', fontsize=14, color='#2E86AB', va='center')
    ax.text(1.0, y, term, ha='left', va='center', fontsize=10, color='#1a1a2e')

# Right box: LLM Reasoning
right_box = FancyBboxPatch((5.5, 1.0), 4.2, 5.5, boxstyle="round,pad=0.1",
                            facecolor='#FDF2E9', edgecolor='#E94F37', linewidth=2)
ax.add_patch(right_box)
ax.text(7.6, 6.2, 'LLM Reasoning', ha='center', va='center', fontsize=12,
        fontweight='bold', color='#E94F37')

# LLM terms
llm_terms = [
    ('Problem difficulty', 5.3),
    ('Sampling diversity / compute', 4.6),
    ('Tools or external knowledge', 3.9),
    ('Model base capability', 3.2),
    ('$P$(\\textit{correct}) per sample', 2.5),
    ('$P_k = P_\\infty(1 - e^{-k/k_0})$', 1.8),
]
for term, y in llm_terms:
    ax.text(5.7, y, '•', fontsize=14, color='#E94F37', va='center')
    ax.text(6.0, y, term, ha='left', va='center', fontsize=10, color='#1a1a2e')

# Arrows connecting terms
for ly in [5.3, 4.6, 3.9, 3.2, 2.5]:
    ax.annotate('', xy=(5.35, ly), xytext=(4.6, ly),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Bottom row: Arrhenius equation (center)
ax.text(5.0, 0.4, '$P_k = P_\\infty \\cdot (1 - e^{-k/k_0})$',
        ha='center', va='center', fontsize=13, style='italic',
        bbox=dict(boxstyle='round', facecolor='#FFF3E0', edgecolor='#F5A623', lw=1.5))

# Legend box
legend_box = FancyBboxPatch((0.3, 0.05), 9.4, 0.2, boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor='none', alpha=0.0)
ax.add_patch(legend_box)

plt.tight_layout()
outpath = '/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/writing/figures/fig2_theory_framework.pdf'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved: {outpath}")
