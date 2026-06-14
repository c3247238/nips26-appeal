"""Figure 2: Hedging Decomposition Across L0 values."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from style_config import *
import numpy as np

l0_values = [22, 41, 82, 176]
l0_labels = ['L0=22', 'L0=41', 'L0=82', 'L0=176']

# From confound_decomposition_multi_l0.json
pct_hedging = [98.6, 98.2, 95.1, 10.0]
pct_hierarchy = [1.4, 1.8, 4.9, 90.0]
pct_recon = [0.0, 0.0, 0.0, 0.0]

# Total FN and absorption rates for secondary axis
total_fn = [657, 489, 185, 10]
absorption_rates = [42.85, 37.49, 14.39, 0.84]

x = np.arange(len(l0_values))
width = 0.5

fig, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

bars_hedge = ax1.bar(x, pct_hedging, width, label='Hedging', color=COLORS['hedging'], alpha=0.85)
bars_hier = ax1.bar(x, pct_hierarchy, width, bottom=pct_hedging, label='Hierarchy-driven', color=COLORS['hierarchy'], alpha=0.85)

ax1.set_ylabel('Fraction of False Negatives (%)')
ax1.set_xlabel('L0 Operating Point')
ax1.set_xticks(x)
ax1.set_xticklabels(l0_labels)
ax1.set_ylim(0, 110)

# Annotate key values
ax1.annotate('98.6%\nhedging', xy=(0, 50), ha='center', fontsize=9, color='white', fontweight='bold')
ax1.annotate('90.0%\nhierarchy', xy=(3, 55), ha='center', fontsize=9, color='white', fontweight='bold')

# Add total FN counts above bars
for i, (fn, ar) in enumerate(zip(total_fn, absorption_rates)):
    ax1.annotate(f'n={fn}\n({ar:.1f}%)', xy=(i, 103), ha='center', fontsize=8, color='black')

ax1.legend(loc='center right', framealpha=0.9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

fig.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), 'hedging_decomposition.pdf')
fig.savefig(outpath)
print(f"Saved: {outpath}")
plt.close(fig)
