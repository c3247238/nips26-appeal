#!/usr/bin/env python3
"""Generate Figure 3: Hierarchy specificity test (semantic vs non-hierarchy control)."""

import json
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, '/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures')
from style_config import set_paper_style, COLORS, ARCH_LABELS

set_paper_style()

with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/iter_003/exp/results/full/statistical_analysis_summary.json') as f:
    data = json.load(f)

archs = data['per_architecture_scores']
# Exclude Random for this plot (trained only)
trained = [a for a in archs if a['family'] != 'Random']

names = [a['family'] for a in trained]
x = np.arange(len(names))
width = 0.35

semantic = [a['semantic_hierarchy_absorption'] for a in trained]
nonhier = [a['nonhierarchy_control_absorption'] for a in trained]

fig, ax = plt.subplots(figsize=(7.5, 4.2))

bars1 = ax.bar(x - width/2, semantic, width, label='Semantic-Hierarchy',
               color='#E74C3C', alpha=0.85, edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, nonhier, width, label='Non-Hierarchy Control',
               color='#2ECC71', alpha=0.85, edgecolor='white', linewidth=0.5)

# Mean difference line
mean_sem = np.mean(semantic)
mean_nh = np.mean(nonhier)
ax.axhline(y=mean_sem, color='#E74C3C', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y=mean_nh, color='#2ECC71', linestyle='--', linewidth=1, alpha=0.5)

# Stats annotation
t = data['h2_hierarchy_specificity']['t_statistic']
p = data['h2_hierarchy_specificity']['p_value']
ax.annotate(f'Paired t-test: $t$ = {t:.2f}, $p$ = {p:.4f}',
            xy=(0.98, 0.95), xycoords='axes fraction',
            fontsize=9, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.25, edgecolor='none'))

ax.set_ylabel('Absorption Score')
ax.set_title('Hierarchy Specificity Test')
ax.set_xticks(x)
ax.set_xticklabels([ARCH_LABELS.get(n.lower(), n) for n in names], rotation=25, ha='right')
ax.legend(loc='upper right', frameon=False)
ax.set_ylim(0, 0.48)

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig3_specificity.pdf', format='pdf')
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig3_specificity.png', dpi=300)
print('Figure 3 saved.')
