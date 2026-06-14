#!/usr/bin/env python3
"""
Generate Figure 4: Decoder Orthogonality vs Steering Effect
Shows H6 falsified - orthogonality does not predict steering
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/exp/results/full/full_decoder_orthogonality.json', 'r') as f:
    data = json.load(f)

# Extract correlation stats
pearson_r = data['correlation_analysis']['pearson_r']
pearson_p = data['correlation_analysis']['pearson_p']
spearman_rho = data['correlation_analysis']['spearman_rho']

# Extract group data
high_orth_data = data['steering_results']['high_orthogonality']
low_orth_data = data['steering_results']['low_orthogonality']

high_orth_effects = [r['abs_effect'] for r in high_orth_data]
low_orth_effects = [r['abs_effect'] for r in low_orth_data]
high_orth_cos = [r['abs_effect'] for r in high_orth_data]  # Just for plotting position
low_orth_cos = [r['abs_effect'] for r in low_orth_data]

# Get mean cosine similarities for x-axis positioning
high_orth_mean_cos = data['feature_groups']['high_orthogonality_mean_cosine']
low_orth_mean_cos = data['feature_groups']['low_orthogonality_mean_cosine']

# Create figure
fig, ax = plt.subplots(figsize=(8, 5.5))

# Box plot for groups
box_data = [high_orth_effects, low_orth_effects]
bp = ax.boxplot(box_data, positions=[0, 1], widths=0.6, patch_artist=True,
                showmeans=True, meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markersize': 8})

colors = ['#2E86AB', '#A23B72']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add scatter points
np.random.seed(42)
jitter_high = np.random.uniform(-0.15, 0.15, len(high_orth_effects))
jitter_low = np.random.uniform(-0.15, 0.15, len(low_orth_effects))
ax.scatter(jitter_high, high_orth_effects, alpha=0.4, color='#2E86AB', s=20, zorder=3)
ax.scatter(1 + jitter_low, low_orth_effects, alpha=0.4, color='#A23B72', s=20, zorder=3)

# Add statistics annotation
stats_text = (f'Pearson r = {pearson_r:.3f}\n'
              f'p = {pearson_p:.3f}\n'
              f'Spearman ρ = {spearman_rho:.3f}\n'
              f'Finding: NOT_SUPPORTED')
ax.annotate(stats_text, xy=(0.5, 0.95), xycoords='axes fraction',
            fontsize=10, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlabel('Decoder Orthogonality Group', fontsize=12)
ax.set_ylabel('Steering Effect (|Δlogit|)', fontsize=12)
ax.set_title('Decoder Orthogonality Does Not Predict Steering Effect\n(H6 - NOT_SUPPORTED)', fontsize=12, fontweight='bold')
ax.set_xticks([0, 1])
ax.set_xticklabels(['High Orthogonality', 'Low Orthogonality'])

# Add group means annotation
ax.annotate(f'μ = 0.131', xy=(0, 0.131), xytext=(0.15, 0.14),
            fontsize=9, color='#2E86AB', fontweight='bold')
ax.annotate(f'μ = 0.107', xy=(1, 0.107), xytext=(1.15, 0.12),
            fontsize=9, color='#A23B72', fontweight='bold')

ax.set_ylim(0, 0.5)
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/writing/figures/fig4_decoder_orthogonality.pdf', dpi=150, bbox_inches='tight')
print("Figure 4 saved: fig4_decoder_orthogonality.pdf")