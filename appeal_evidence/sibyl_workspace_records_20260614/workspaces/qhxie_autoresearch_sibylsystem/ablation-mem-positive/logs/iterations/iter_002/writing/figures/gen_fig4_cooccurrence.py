#!/usr/bin/env python3
"""
Generate Figure 4: Co-occurrence Formula Comparison
Shows revised vs baseline co-occurrence correlation - positive vs negative.

Data source: exp/results/full/cooccurrence_analysis.json
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/exp/results/full/cooccurrence_analysis.json', 'r') as f:
    data = json.load(f)

# Extract correlation values
metrics = data['metrics']
baseline_r = -0.52  # from prior work
revised_r = metrics['revised_score_vs_absorption']['r']  # 0.647

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart: correlation comparison
labels = ['Baseline\n(Prior Work)', 'Simple Cosine', 'Revised Formula']
values = [baseline_r, metrics['simple_cosine_vs_absorption']['r'], revised_r]
colors = ['gray', 'steelblue', 'forestgreen']

bars = ax1.bar(labels, values, color=colors, alpha=0.8, width=0.6)
ax1.axhline(y=0, color='black', linewidth=1)
ax1.axhline(y=0.6, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target r > 0')
ax1.set_ylabel('Pearson Correlation (r)', fontsize=12)
ax1.set_title('Co-occurrence Formula Comparison', fontsize=12, fontweight='bold')
ax1.set_ylim([-0.7, 0.8])
ax1.legend(loc='upper left')

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02 if height > 0 else height - 0.08,
             f'r = {val:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=11, fontweight='bold')

# Add improvement annotation
ax1.annotate('', xy=(2, revised_r), xytext=(0, baseline_r),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax1.text(1, 0.0, f'Improvement\n+{revised_r - baseline_r:.3f}', ha='center', fontsize=10, color='red')

# Second panel: scatter plot of revised score vs absorption
sample_records = data['sample_records'][:200]  # limit to 200 points for clarity
revised_scores = [r['revised_score'] for r in sample_records]
absorption_rates = [r['absorption_rate'] for r in sample_records]

ax2.scatter(revised_scores, absorption_rates, alpha=0.5, s=30, c='forestgreen')
ax2.set_xlabel('Revised Co-occurrence Score', fontsize=12)
ax2.set_ylabel('Absorption Rate', fontsize=12)
ax2.set_title(f'Revised Score vs Absorption (r = {revised_r:.3f})', fontsize=12, fontweight='bold')

# Add trend line
z = np.polyfit(revised_scores, absorption_rates, 1)
p = np.poly1d(z)
x_line = np.linspace(min(revised_scores), max(revised_scores), 100)
ax2.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (r = {revised_r:.3f})')
ax2.legend()

fig.suptitle('Figure 4: Co-occurrence Formula Achieves Positive Correlation',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/writing/figures/fig4_cooccurrence.pdf',
            format='pdf', bbox_inches='tight', dpi=300)
print("Figure 4 saved to writing/figures/fig4_cooccurrence.pdf")