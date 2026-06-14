#!/usr/bin/env python3
"""Generate Figure 2: Construct validity scatter plot (first-letter vs semantic-hierarchy absorption)."""

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Load style config
import sys
sys.path.insert(0, '/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures')
from style_config import set_paper_style, COLORS, ARCH_LABELS

set_paper_style()

# Load data
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/iter_003/exp/results/full/statistical_analysis_summary.json') as f:
    data = json.load(f)

archs = data['per_architecture_scores']
# Exclude Random SAE
trained = [a for a in archs if a['family'] != 'Random']

x = [a['first_letter_absorption'] for a in trained]
y = [a['semantic_hierarchy_absorption'] for a in trained]
labels = [a['family'] for a in trained]

fig, ax = plt.subplots(figsize=(5.5, 4.2))

# Color mapping
color_map = {
    'BatchTopK': COLORS['batchtopk'],
    'GatedSAE': COLORS['gatedsae'],
    'JumpRelu': COLORS['jumprelu'],
    'MatryoshkaBatchTopK': COLORS['matryoshka'],
    'PAnneal': COLORS['pannael'],
    'Standard': COLORS['standard'],
    'TopK': COLORS['topk'],
}

colors = [color_map.get(l, '#333333') for l in labels]

# Scatter points
for xi, yi, li, ci in zip(x, y, labels, colors):
    ax.scatter(xi, yi, c=ci, s=120, zorder=3, edgecolors='white', linewidths=1.5)
    # Offset labels to avoid overlap
    offset_x = 0.015
    offset_y = 0.012
    if li == 'GatedSAE':
        offset_x = -0.12
        offset_y = 0.015
    elif li == 'PAnneal':
        offset_x = -0.11
        offset_y = -0.025
    elif li == 'Standard':
        offset_x = -0.13
        offset_y = -0.02
    elif li == 'MatryoshkaBatchTopK':
        offset_x = 0.015
        offset_y = -0.025
    ax.annotate(ARCH_LABELS.get(li.lower(), li), (xi, yi),
                xytext=(offset_x, offset_y), textcoords='offset fontsize',
                fontsize=8, ha='left')

# Regression line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
x_line = np.linspace(min(x) - 0.02, max(x) + 0.02, 100)
ax.plot(x_line, p(x_line), 'k--', linewidth=1.2, alpha=0.6, zorder=1)

# Bootstrap CI band (approximate from summary stats)
r = data['h1_construct_validity']['excludes_random_sae']['pearson_r']
ci_low = data['h1_construct_validity']['excludes_random_sae']['bootstrap_95_ci_lower']
ci_high = data['h1_construct_validity']['excludes_random_sae']['bootstrap_95_ci_upper']

# Annotation
ax.annotate(f'Pearson $r$ = {r:.3f}\n95% CI: [{ci_low:.3f}, {ci_high:.3f}]',
            xy=(0.95, 0.05), xycoords='axes fraction',
            fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.3, edgecolor='none'))

ax.set_xlabel('First-Letter Absorption Score')
ax.set_ylabel('Semantic-Hierarchy Absorption Score')
ax.set_xlim(-0.02, 0.62)
ax.set_ylim(-0.02, 0.42)
ax.set_title('Construct Validity: First-Letter vs. Semantic-Hierarchy Absorption')

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig2_scatter.pdf', format='pdf')
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig2_scatter.png', dpi=300)
print('Figure 2 saved.')
