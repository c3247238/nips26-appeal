#!/usr/bin/env python3
"""Generate Figure 4: Robustness across tau_fs thresholds."""

import json
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, '/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures')
from style_config import set_paper_style

set_paper_style()

with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/iter_003/exp/results/full/statistical_analysis_summary.json') as f:
    data = json.load(f)

tau_data = data['tau_fs_robustness']

tau_vals = [t['tau_fs'] for t in tau_data]
r_vals = [t['pearson_r_firstletter_semantic'] for t in tau_data]
ci_low = [t['bootstrap_ci_lower'] for t in tau_data]
ci_high = [t['bootstrap_ci_upper'] for t in tau_data]

# Error bar lengths
err_low = [r - l for r, l in zip(r_vals, ci_low)]
err_high = [h - r for r, h in zip(r_vals, ci_high)]

fig, ax = plt.subplots(figsize=(5.5, 4.2))

ax.errorbar(tau_vals, r_vals, yerr=[err_low, err_high],
            fmt='o-', color='#1f77b4', ecolor='#1f77b4',
            capsize=6, capthick=1.5, linewidth=1.5, markersize=8,
            elinewidth=1.5, zorder=3)

# Reference lines
ax.axhline(y=0.6, color='red', linestyle='--', linewidth=1, alpha=0.6, label='$r$ = 0.6 (H1 threshold)')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.4)
ax.axhspan(-0.4, 0, alpha=0.08, color='red')

# Annotations
for tv, rv in zip(tau_vals, r_vals):
    ax.annotate(f'{rv:.3f}', (tv, rv), xytext=(0, 12),
                textcoords='offset points', ha='center', fontsize=9)

ax.set_xlabel(r'Feature-Splitting Threshold $\tau_{\mathrm{fs}}$')
ax.set_ylabel('Pearson Correlation $r$')
ax.set_title('Robustness: Correlation Across Feature-Splitting Thresholds')
ax.set_xticks(tau_vals)
ax.set_xticklabels([f'{t:.2f}' for t in tau_vals])
ax.set_ylim(-0.5, 1.05)
ax.legend(loc='lower right', frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig4_robustness.pdf', format='pdf')
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/writing/figures/fig4_robustness.png', dpi=300)
print('Figure 4 saved.')
