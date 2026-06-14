#!/usr/bin/env python3
"""
Generate Figure 4: Steering sensitivity by feature type (absorbed vs. non-absorbed).
H3 results: zero improvement for both groups.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from h3_steering_results.json
feature_types = ['Absorbed\n(n=7)', 'Non-absorbed\n(n=1014)']
baseline_means = [37.45, 185.43]
steered_means = [37.45, 185.43]
baseline_stds = [129.66, 3693.98]  # Very large std due to high variance
colors_bar = ['#E63946', '#2E86AB']
improvement = [0.0, 0.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# Left panel: baseline vs steered means (they're identical, so show improvement)
x = np.arange(len(feature_types))
width = 0.35

bars1 = ax1.bar(x - width/2, baseline_means, width, label='Baseline', color=colors_bar, alpha=0.7, edgecolor='black')
bars2 = ax1.bar(x + width/2, steered_means, width, label='Steered', color=colors_bar, alpha=1.0, edgecolor='black', hatch='///')

ax1.set_ylabel('Feature Sensitivity (mean)', fontsize=12)
ax1.set_title('Steering Sensitivity:\nBaseline vs. Steered', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(feature_types, fontsize=11)
ax1.set_yscale('symlog', linthresh=1.0)
ax1.legend(fontsize=10)

# Add value labels
for bar, val in zip(bars1, baseline_means):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f'{val:.1f}', ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars2, steered_means):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f'{val:.1f}', ha='center', va='bottom', fontsize=9)

# Right panel: improvement (0 for both)
colors_imp = ['#E63946', '#2E86AB']
bars3 = ax2.bar(x, improvement, color=colors_imp, edgecolor='black', width=0.5)

ax2.set_ylabel('Sensitivity Improvement ($\\Delta s$)', fontsize=12)
ax2.set_title('Steering Improvement by Feature Type\nZero Improvement for Both Groups', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(feature_types, fontsize=11)
ax2.set_ylim(-0.1, 0.3)
ax2.axhline(y=0, color='black', linewidth=1)

# Add zero labels
for bar in bars3:
    ax2.text(bar.get_x() + bar.get_width() / 2, 0.02,
             '0.0', ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

ax2.text(0.5, 0.22, 'H3 FAILED: Steering produces\nno sensitivity improvement',
         ha='center', va='top', fontsize=10, color='darkred',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='mistyrose', alpha=0.8))

for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('H3: Steering Intervention Results', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current/writing/figures/fig4_h3_steering_sensitivity.pdf', dpi=300, bbox_inches='tight')
print("Figure 4 saved.")
