#!/usr/bin/env python3
"""Generate Figure 7: Safety-critical vs. non-safety feature absorption. H_Safe FAILED."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

feature_types = ['Safety-critical\n(n=20)', 'Non-safety\n(n=20)']
absorption = [0.907, 0.906]
stds = [0.038, 0.048]
colors_bar = ['#E63946', '#2E86AB']

fig, ax = plt.subplots(figsize=(7, 5.5))

x = np.arange(len(feature_types))
bars = ax.bar(x, absorption, color=colors_bar, edgecolor='black', linewidth=1.2, width=0.5)
ax.errorbar(x, absorption, yerr=stds, fmt='none', ecolor='black', capsize=5, capthick=1.5)

ax.set_ylabel('Absorption Rate', fontsize=13)
ax.set_title('H_Safe: Safety-Critical vs. Non-Safety Feature Absorption\nMann-Whitney U = 216.5, p = 0.665', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(feature_types, fontsize=12)
ax.set_ylim(0.8, 1.0)

for bar, val, std in zip(bars, absorption, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.annotate('', xy=(0, 0.975), xytext=(1, 0.975),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax.text(0.5, 0.978, 'Diff: 0.001', ha='center', va='bottom', fontsize=10, color='gray')

ax.text(0.5, 0.955, 'H_Safe FAILED: No significant difference', ha='center', va='top', fontsize=11, color='darkred', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='mistyrose', alpha=0.8))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/writing/figures/fig7_h_safe_comparison.pdf', dpi=300, bbox_inches='tight')
print("Figure 7 saved.")
