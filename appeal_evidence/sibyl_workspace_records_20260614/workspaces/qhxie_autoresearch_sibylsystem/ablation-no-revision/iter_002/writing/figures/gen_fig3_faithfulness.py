"""
Generate Figure 3: Circuit Faithfulness Comparison (H4).
Shows that SAE bottleneck reduces faithfulness but absorption level does not predict which latents matter.
Data source: exp/results/pilots/h4_pilot.json
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from h4_pilot.json
methods = ['Raw\nResidual', 'SAE\nAll Latents', 'SAE\nLow-Absorption', 'SAE\nHigh-Absorption']
faithfulness = [0.400, 0.289, 0.000, 0.000]
colors = ['#27ae60', '#4a90d9', '#e74c3c', '#e74c3c']

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(methods))
bars = ax.bar(x, faithfulness, width=0.55, color=colors, alpha=0.85)

# Dashed line at raw residual level
ax.axhline(y=0.400, color='#27ae60', linestyle='--', linewidth=1.5, alpha=0.7, label='Raw residual level (0.400)')

# Value labels
for bar, val in zip(bars, faithfulness):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f'{val:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Faithfulness Score', fontsize=12)
ax.set_title('Figure 3. SAE Bottleneck Reduces Faithfulness;\nAbsorption Level Does Not Predict Causal Importance\n(H4 Results, Pilot, Layer 8, $d_{\\text{sae}} = 24{,}576$)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, 0.52)
ax.legend(fontsize=9)

# Note
ax.text(0.5, -0.12, 'Low- and high-absorption subsets both yield 0.000, indicating the subset selection\nmethod (top/bottom 10% by corpus-wide score) does not isolate causally relevant latents.',
        ha='center', va='top', fontsize=8.5, color='#555555', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/writing/figures/fig3_faithfulness.pdf', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: fig3_faithfulness.pdf")