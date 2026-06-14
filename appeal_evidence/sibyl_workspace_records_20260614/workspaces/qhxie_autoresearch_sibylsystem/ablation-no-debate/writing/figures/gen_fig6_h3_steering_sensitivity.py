#!/usr/bin/env python3
"""Generate Figure 6: Steering sensitivity by feature type (absorbed vs. non-absorbed). H3 PASSED."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

feature_types = ['Absorbed\n(n=20)', 'Non-absorbed\n(n=20)']
mean_sensitivity = [0.055, 0.034]
colors_bar = ['#E63946', '#2E86AB']

fig, ax = plt.subplots(figsize=(7, 5.5))

x = np.arange(len(feature_types))
bars = ax.bar(x, mean_sensitivity, color=colors_bar, edgecolor='black', linewidth=1.2, width=0.5)

ax.set_ylabel('Mean Sensitivity', fontsize=13)
ax.set_title('H3: Steering Sensitivity by Feature Type\nAbsorbed features show 1.62x higher sensitivity', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(feature_types, fontsize=12)
ax.set_ylim(0, 0.08)

for bar, val in zip(bars, mean_sensitivity):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.annotate('', xy=(0, 0.065), xytext=(1, 0.065),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(0.5, 0.068, '1.62x ratio', ha='center', va='bottom', fontsize=11, color='green', fontweight='bold')

ax.text(0.5, 0.055, 'H3 PASSED', ha='center', va='top', fontsize=12, color='darkgreen', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.8))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/writing/figures/fig6_h3_steering_sensitivity.pdf', dpi=300, bbox_inches='tight')
print("Figure 6 saved.")
