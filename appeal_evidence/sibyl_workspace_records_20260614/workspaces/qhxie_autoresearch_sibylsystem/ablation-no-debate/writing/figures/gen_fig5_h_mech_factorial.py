#!/usr/bin/env python3
"""Generate Figure 5: 2x2 factorial decomposition of absorption into geometric vs. learned contributions."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

conditions = ['A\n(Random\nEncoder,\nRandom Decoder)', 'B\n(Trained\nEncoder,\nRandom Decoder)',
              'C\n(Random\nEncoder,\nTrained Decoder)', 'D\n(Trained\nEncoder,\nTrained Decoder)']
absorption = [0.299, 0.490, 0.299, 0.484]
stds = [0.010, 0.030, 0.010, 0.037]
colors = ['#A8DADC', '#457B9D', '#E63946', '#1D3557']

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(conditions))
bars = ax.bar(x, absorption, color=colors, edgecolor='black', linewidth=1.2, width=0.6)
ax.errorbar(x, absorption, yerr=stds, fmt='none', ecolor='black', capsize=5, capthick=1.5)

ax.set_ylabel('Absorption Rate', fontsize=13)
ax.set_title('2x2 Factorial Decomposition: Geometric vs. Learned Contributions', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=10)
ax.set_ylim(0, 0.6)
ax.axhline(y=0.299, color='#E63946', linestyle='--', linewidth=1.5, alpha=0.7, label='Geometric contribution = 0.299')

for bar, val, std in zip(bars, absorption, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.015,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.annotate('', xy=(2, 0.315), xytext=(3, 0.315),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(2.5, 0.33, 'Learned\n+0.185', ha='center', va='bottom', fontsize=10, color='green', fontweight='bold')

ax.annotate('', xy=(0, 0.255), xytext=(2, 0.255),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=2))
ax.text(1, 0.22, 'Decoder geometry\nvs. random', ha='center', va='top', fontsize=9, color='gray')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/writing/figures/fig5_h_mech_factorial.pdf', dpi=300, bbox_inches='tight')
print("Figure 5 saved.")
