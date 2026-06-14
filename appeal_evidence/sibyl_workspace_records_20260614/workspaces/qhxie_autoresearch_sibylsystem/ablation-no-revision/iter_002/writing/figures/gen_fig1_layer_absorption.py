"""
Generate Figure 1: Absorption Score Distribution by Layer (H1/H3).
Shows mean absorption score and % > 0.5 per layer with inverted-U pattern.
Data source: exp/results/pilots/h3_pilot.json
"""

import matplotlib.pyplot as plt
import numpy as np
import json

# Data from h3_pilot.json layer_results
layers = [0, 2, 4, 6, 8, 10]
mean_l0 = [18.9, 29.1, 37.8, 57.0, 71.9, 56.0]
mean_absorption = [0.2292, 0.4698, 0.5027, 0.4302, 0.3050, 0.2865]
pct_gt_0_5 = [19.48, 45.49, 49.34, 40.98, 20.85, 17.34]

fig, ax1 = plt.subplots(figsize=(9, 5))

x = np.arange(len(layers))
width = 0.35

# Bar chart
bars = ax1.bar(x, pct_gt_0_5, width, color='#4a90d9', alpha=0.8, label='% latents with $A_f > 0.5$')
ax1.set_xlabel('Layer ($\\ell$)', fontsize=12)
ax1.set_ylabel('% Latents with $A_f > 0.5$', fontsize=12, color='#4a90d9')
ax1.set_xticks(x)
ax1.set_xticklabels([str(l) for l in layers])
ax1.tick_params(axis='y', labelcolor='#4a90d9')
ax1.set_ylim(0, 60)

# Line overlay for mean absorption
ax2 = ax1.twinx()
line = ax2.plot(x, mean_absorption, 'o-', color='#e74c3c', linewidth=2, markersize=7,
                label='Mean $A_f$', zorder=5)
ax2.set_ylabel('Mean Absorption Score ($A_f$)', fontsize=12, color='#e74c3c')
ax2.tick_params(axis='y', labelcolor='#e74c3c')
ax2.set_ylim(0, 0.65)

# Annotate the peak
ax1.annotate('Peak: 49.3%\nat layer 4', xy=(2, 49.34), xytext=(2.6, 56),
             fontsize=9, ha='center',
             arrowprops=dict(arrowstyle='->', color='#7f8c8d'),
             color='#4a90d9')

# L0 annotation
for i, (l, l0) in enumerate(zip(layers, mean_l0)):
    ax1.annotate(f'L0={l0}', xy=(i, pct_gt_0_5[i] + 2), ha='center', fontsize=8, color='#7f8c8d')

ax1.set_title('Figure 1. Absorption Is Rare and Peaks at Mid-Layers\n(H1 + H3 Results, Pilot, $d_{\\text{sae}} = 24{,}576$, 100 sequences)', fontsize=11)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/writing/figures/fig1_layer_absorption.pdf', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: fig1_layer_absorption.pdf")