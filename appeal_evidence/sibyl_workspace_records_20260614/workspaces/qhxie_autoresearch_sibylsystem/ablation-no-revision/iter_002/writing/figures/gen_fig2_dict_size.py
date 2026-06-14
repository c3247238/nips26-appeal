"""
Generate Figure 2: Absorption vs Dictionary Size (H5).
Shows monotonic decrease in absorption with larger dictionary size.
Data source: exp/results/pilots/h5_pilot.json
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from h5_pilot.json
dict_sizes = [2048, 8192, 24576]
dict_labels = ['2,048', '8,192', '24,576']
mean_absorption = [0.026751, 0.006688, 0.002229]
pct_gt_0_5 = [2.246, 0.562, 0.187]
random_mean = [0.0, 0.0, 0.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: mean absorption score (log scale)
x = np.arange(len(dict_sizes))
ax1.semilogy(x, mean_absorption, 'o-', color='#2c3e50', linewidth=2, markersize=9,
             label='Real SAE', zorder=5)
ax1.semilogy(x, [max(r, 1e-7) for r in random_mean], 's--', color='#e74c3c',
             linewidth=1.5, markersize=7, label='Random control')
ax1.set_xticks(x)
ax1.set_xticklabels(dict_labels)
ax1.set_xlabel('Dictionary Size ($d_{\\text{sae}}$)', fontsize=11)
ax1.set_ylabel('Mean Absorption Score ($A_f$)', fontsize=11)
ax1.set_title('Mean Absorption Score vs Dictionary Size', fontsize=10)
ax1.legend(fontsize=9)
ax1.set_ylim(1e-4, 0.1)
ax1.grid(True, alpha=0.3)

# Right: % > 0.5
ax2.bar(x, pct_gt_0_5, width=0.5, color='#4a90d9', alpha=0.85, label='Real SAE')
ax2.bar(x, [0, 0, 0], width=0.5, color='#e74c3c', alpha=0.5, label='Random control')
ax2.set_xticks(x)
ax2.set_xticklabels(dict_labels)
ax2.set_xlabel('Dictionary Size ($d_{\\text{sae}}$)', fontsize=11)
ax2.set_ylabel('% Latents with $A_f > 0.5$', fontsize=11)
ax2.set_title('Prevalence of Absorption ($A_f > 0.5$)', fontsize=10)
ax2.set_ylim(0, 3)
ax2.legend(fontsize=9)

# Annotate 10x reduction
ax2.annotate('', xy=(2, 0.3), xytext=(0, 2.4),
             arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))
ax2.text(1, 1.2, '10x reduction', ha='center', fontsize=9, color='#7f8c8d')

fig.suptitle('Figure 2. Larger Dictionaries Reduce Absorption\n(H5 Results, Pilot, Layer 8, 100 sequences)', fontsize=11)
plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/writing/figures/fig2_dict_size.pdf', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: fig2_dict_size.pdf")