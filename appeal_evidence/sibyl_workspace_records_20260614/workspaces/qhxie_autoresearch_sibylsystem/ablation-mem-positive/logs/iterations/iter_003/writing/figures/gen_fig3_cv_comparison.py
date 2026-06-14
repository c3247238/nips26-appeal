#!/usr/bin/env python3
"""
Generate Figure 3: Cross-Layer CV Comparison
Shows CV difference across layers - absorbed features have HIGHER CV (reversed from prediction).

Data source: exp/results/full/cv_full_analysis.json
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/exp/results/full/cv_full_analysis.json', 'r') as f:
    data = json.load(f)

# Extract CV values per layer
layers = [0, 3, 6, 9, 11]
cv_absorbed_means = []
cv_absorbed_stds = []
cv_non_absorbed_means = []

for layer in layers:
    layer_data = data['layer_results'][str(layer)]
    cv_absorbed_means.append(layer_data['cv_absorbed']['mean'])
    cv_absorbed_stds.append(layer_data['cv_absorbed']['std'])
    cv_non_absorbed_means.append(layer_data['cv_non_absorbed']['mean'])

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart: CV comparison by layer
x = np.arange(len(layers))
width = 0.35

bars1 = ax1.bar(x - width/2, cv_absorbed_means, width, yerr=cv_absorbed_stds,
                label='Absorbed Features', color='crimson', alpha=0.8, capsize=5)
bars2 = ax1.bar(x + width/2, cv_non_absorbed_means, width,
                label='Non-Absorbed Features', color='steelblue', alpha=0.8)

ax1.set_xlabel('Layer', fontsize=12)
ax1.set_ylabel('Coefficient of Variation (CV)', fontsize=12)
ax1.set_title('CV Comparison: Absorbed vs Non-Absorbed Features', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'L{l}' for l in layers])
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3, axis='y')

# Add note about reversal
ax1.text(0.05, 0.85, 'REVERSED from prediction:\nAbsorbed have HIGHER CV',
         transform=ax1.transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Bar chart showing the 733x ratio
ratio_values = [cv_absorbed_means[i] / max(cv_non_absorbed_means[i], 0.001) for i in range(len(layers))]
colors_ratio = ['crimson' if r > 100 else 'steelblue' for r in ratio_values]
bars3 = ax2.bar(layers, ratio_values, color=colors_ratio, alpha=0.8, width=6)
ax2.set_xlabel('Layer', fontsize=12)
ax2.set_ylabel('CV Ratio (Absorbed / Non-Absorbed)', fontsize=12)
ax2.set_title('CV Ratio Across Layers\n(Absorbed features have ~733x higher CV)', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 800])
ax2.axhline(y=733, color='red', linestyle='--', linewidth=2, label='733x ratio')
ax2.legend()

# Add ratio values on bars
for i, (layer, ratio) in enumerate(zip(layers, ratio_values)):
    ax2.text(layer, ratio + 30, f'{ratio:.0f}x', ha='center', fontsize=9, color='darkred')

fig.suptitle('Figure 3: Variance Paradox - Absorbed Features Have Higher CV',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/writing/figures/fig3_cv_comparison.pdf',
            format='pdf', bbox_inches='tight', dpi=300)
print("Figure 3 saved to writing/figures/fig3_cv_comparison.pdf")