#!/usr/bin/env python3
"""Generate Figure 5: Cross-Architecture Absorption Comparison"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/current/exp/results/pilots/e7_cross_architecture/e7_cross_architecture.json', 'r') as f:
    data = json.load(f)

# Extract data
architectures = ['GemmaScope\nJumpReLU', 'GPT-2\nReLU']
proj_abs = [data['gemma_data']['absorption_rates']['mean_projection_absorption'],
            data['gpt2_data']['absorption_rates']['mean_projection_absorption']]
proj_std = [data['gemma_data']['absorption_rates']['std_projection_absorption'],
            data['gpt2_data']['absorption_rates']['std_projection_absorption']]
abl_rate = [data['gemma_data']['absorption_rates']['ablation_rate'] * 100,
            data['gpt2_data']['absorption_rates']['ablation_rate'] * 100]

# Create figure with grouped bars
fig, ax = plt.subplots(figsize=(7, 4.5))

x = np.arange(len(architectures))
width = 0.35

# Projection absorption bars
bars1 = ax.bar(x - width/2, proj_abs, width, yerr=proj_std,
               label='Projection absorption', color='#4472C4', alpha=0.8,
               capsize=4, edgecolor='black', linewidth=0.5)

# Ablation rate bars (as percentage)
bars2 = ax.bar(x + width/2, [r/100 for r in abl_rate], width,
               label='Ablation rate', color='#ED7D31', alpha=0.8,
               edgecolor='black', linewidth=0.5)

# Add value labels on bars
for bar, val, std in zip(bars1, proj_abs, proj_std):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar, val in zip(bars2, abl_rate):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Significance marker
ax.annotate('*', xy=(0.5, 0.98), xytext=(0.5, 1.02),
            fontsize=20, ha='center', va='center',
            arrowprops=dict(arrowstyle='-', color='black', lw=1))
ax.text(0.5, 1.04, f'p < 0.001\nCohen\'s d = 1.82', ha='center', va='bottom', fontsize=8)

# Labels and styling
ax.set_ylabel('Rate', fontsize=11)
ax.set_xlabel('Architecture', fontsize=11)
ax.set_title('Cross-Architecture Absorption Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(architectures)
ax.set_ylim(0, 1.15)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Add annotation about difference
ax.annotate(f'Projection absorption\ndiff: 7.7%',
            xy=(0.5, 0.45), fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.4))

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/current/writing/figures/fig5_cross_arch.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/current/writing/figures/fig5_cross_arch.png', dpi=300, bbox_inches='tight')
print("Figure 5 saved.")
