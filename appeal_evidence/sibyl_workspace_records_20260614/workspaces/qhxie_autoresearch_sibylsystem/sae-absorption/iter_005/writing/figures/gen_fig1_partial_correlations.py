"""Generate Figure 1: Absorption-Quality Partial Correlations Before vs. After L0 Control.

Grouped bar chart showing partial r for 4 quality metrics with/without L0 control.
Highlights the suppression effect on sparse probing.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Data from P1_confound_go_nogo.json
metrics = ['Sparse Probing\nF1', 'SCR', 'RAVEL\nTPP', 'Unlearning']
partial_r_without_l0 = [-0.664, -0.692, -0.488, -0.082]
partial_r_with_l0 = [-0.746, -0.570, -0.331, -0.123]

# CIs from P1_confound_go_nogo.json (with L0)
ci_with_l0 = [
    (-0.853, -0.579),
    (-0.749, -0.315),
    (-0.569, -0.041),
    (-0.458, 0.242),
]
# CIs without L0
ci_without_l0 = [
    (-0.800, -0.463),
    (-0.824, -0.488),
    (-0.682, -0.230),
    (-0.419, 0.275),
]

# Significance markers
p_with_l0 = [1.16e-9, 6.57e-5, 0.022, 0.487]

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(metrics))
width = 0.35

# Error bars
yerr_without = [
    [abs(partial_r_without_l0[i]) - abs(ci_without_l0[i][0]) for i in range(4)],
    [abs(ci_without_l0[i][1]) - abs(partial_r_without_l0[i]) for i in range(4)],
]
yerr_with = [
    [abs(partial_r_with_l0[i]) - abs(ci_with_l0[i][0]) for i in range(4)],
    [abs(ci_with_l0[i][1]) - abs(partial_r_with_l0[i]) for i in range(4)],
]

# Plot absolute values for visual clarity
bars1 = ax.bar(x - width/2, [abs(v) for v in partial_r_without_l0], width,
               label='Without L0 control', color='#4878CF', alpha=0.85,
               edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, [abs(v) for v in partial_r_with_l0], width,
               label='With L0 control', color='#D65F5F', alpha=0.85,
               edgecolor='white', linewidth=0.5)

# Threshold line
ax.axhline(y=0.2, color='gray', linestyle='--', linewidth=1.0, alpha=0.6, label='|r| = 0.2 threshold')

# Suppression effect annotation (sparse probing)
ax.annotate('', xy=(0 + width/2, 0.746), xytext=(0 - width/2, 0.664),
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2.0))
ax.text(0.0, 0.78, 'Suppression\neffect', ha='center', va='bottom',
        fontsize=8, color='#2ca02c', fontweight='bold')

# Significance stars
for i, p in enumerate(p_with_l0):
    if p < 0.001:
        stars = '***'
    elif p < 0.01:
        stars = '**'
    elif p < 0.05:
        stars = '*'
    else:
        stars = 'n.s.'
    ax.text(i + width/2, abs(partial_r_with_l0[i]) + 0.02, stars,
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            color='#D65F5F' if p < 0.05 else 'gray')

ax.set_ylabel('|Partial correlation| with absorption', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylim(0, 0.95)
ax.legend(fontsize=9, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Absorption-Quality Partial Correlations: Effect of L0 Control', fontsize=12, pad=12)

plt.tight_layout()

outdir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(outdir, 'fig1_partial_correlations.pdf'), bbox_inches='tight', dpi=300)
fig.savefig(os.path.join(outdir, 'fig1_partial_correlations.png'), bbox_inches='tight', dpi=300)
print(f"Saved fig1_partial_correlations.pdf and .png to {outdir}")
plt.close()
