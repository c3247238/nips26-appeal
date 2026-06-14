"""Figure 1: Universal Control Failure across 5 hierarchy domains."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from style_config import *
import numpy as np

domains = ['First-letter', 'City→Continent', 'City→Language', 'Animal→Class', 'City→Country']
measured = [15.96, 6.49, 6.56, 1.43, 0.0]
shuffled = [74.6, 45.16, 18.03, 39.29, 10.33]
random_ctrl = [11.8, 12.9, 20.77, 34.29, 19.02]

x = np.arange(len(domains))
width = 0.25

fig, ax = plt.subplots(figsize=(FIG_WIDTH_FULL * 0.85, FIG_HEIGHT))

bars1 = ax.bar(x - width, measured, width, label='Measured', color=COLORS['ours'], edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, shuffled, width, label='Shuffled control', color=COLORS['shuffled'], edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + width, random_ctrl, width, label='Random probe', color=COLORS['random'], edgecolor='white', linewidth=0.5)

ax.axhline(y=2, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Expected random floor (<2%)')

ax.set_ylabel('Absorption Rate (%)')
ax.set_xticks(x)
ax.set_xticklabels(domains, rotation=15, ha='right')
ax.legend(loc='upper right', framealpha=0.9)
ax.set_ylim(0, 85)

# Annotate ratios
for i, (m, s) in enumerate(zip(measured, shuffled)):
    if m > 0:
        ratio = s / m
        ax.annotate(f'{ratio:.1f}×', xy=(i, s + 1), ha='center', fontsize=9, color=COLORS['shuffled'], fontweight='bold')
    else:
        ax.annotate('∞', xy=(i, s + 1), ha='center', fontsize=10, color=COLORS['shuffled'], fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), 'control_failure.pdf')
fig.savefig(outpath)
print(f"Saved: {outpath}")
plt.close(fig)
