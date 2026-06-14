#!/usr/bin/env python3
"""
Generate Figure 2: Activation Patching Recovery
Shows parent recovery percentages for 9 test words - validates genuine absorption
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# Data from pilot_summary.json - activation patching results
words = ['eight', 'lower', 'liked', 'offer', 'often', 'school', 'turn', 'move', 'play']
recoveries = [75.2, 75.2, 74.8, 63.5, 69.1, 75.2, 73.5, 48.8, 50.4]
top_features = [22545, 22545, 3839, 4356, 18745, 22545, 18836, 20818, 485]

# Also load from full results if available
try:
    with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/exp/results/pilot_summary.json', 'r') as f:
        data = json.load(f)
        # Extract pilot_activation_patching results
        pilot_data = data.get('pilot_results', {}).get('pilot_activation_patching', {})
        key_obs = pilot_data.get('key_observations', [])
except:
    pass

# Create figure
fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(words))
colors = ['#2E86AB' if r >= 60 else '#F18F01' if r >= 50 else '#C73E1D' for r in recoveries]

bars = ax.bar(x, recoveries, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

# Add value labels on bars
for bar, val in zip(bars, recoveries):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Threshold line
ax.axhline(y=10, color='red', linestyle='--', linewidth=2, label='10% threshold')
ax.axhline(y=67.3, color='green', linestyle='-', linewidth=1.5, alpha=0.7, label='Mean: 67.3%')

ax.set_xlabel('Test Words (Persistent Core Words)', fontsize=12)
ax.set_ylabel('Parent Feature Recovery (%)', fontsize=12)
ax.set_title('Activation Patching Validates Genuine Absorption with Causal Structure\n(All 9 words exceed 10% recovery threshold)', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(words, rotation=45, ha='right', fontsize=10)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0, 90)

# Add grid
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# Add annotation
ax.annotate('All 9/9 words pass\n10% threshold', xy=(4, 15), fontsize=9,
            ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/writing/figures/fig2_activation_patching.pdf', dpi=150, bbox_inches='tight')
print("Figure 2 saved: fig2_activation_patching.pdf")