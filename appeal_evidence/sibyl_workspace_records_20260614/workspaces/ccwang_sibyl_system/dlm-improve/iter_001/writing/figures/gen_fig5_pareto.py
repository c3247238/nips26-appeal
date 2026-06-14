"""
Generate Figure 5: Pareto frontier — Accuracy vs NFE for LLaDA-8B on GSM8K.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_config import *

workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data from full-scale experiments
methods = {
    'Standard-32':      {'nfe': 32,  'acc': 29.3, 'color': COLORS['standard'], 'marker': 'o'},
    'Standard-64':      {'nfe': 64,  'acc': 30.9, 'color': COLORS['standard'], 'marker': 'o'},
    'DNB-84':           {'nfe': 83,  'acc': 30.9, 'color': COLORS['dnb'],      'marker': 'D'},
    'Standard-128':     {'nfe': 124, 'acc': 32.1, 'color': COLORS['standard'], 'marker': 'o'},
    'Entropy-Revise-64':{'nfe': 68,  'acc': 33.1, 'color': COLORS['entropy_revise'], 'marker': '^'},
    'CARD-84':          {'nfe': 71,  'acc': 34.9, 'color': COLORS['ours'],     'marker': '*'},
}

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
label_offsets = {
    'Standard-32': (8, 6),
    'Standard-64': (8, 8),
    'DNB-84': (10, -14),
    'Standard-128': (-38, 8),
    'Entropy-Revise-64': (10, 12),
    'CARD-84': (-22, 14),
}

for name, m in methods.items():
    ms = MARKER_SIZE + 4 if name == 'CARD-84' else MARKER_SIZE
    ax.plot(m['nfe'], m['acc'], m['marker'], color=m['color'], markersize=ms,
            markeredgecolor='black', markeredgewidth=0.5, zorder=5,
            label=name)
    dx, dy = label_offsets[name]
    ax.annotate(
        name,
        (m['nfe'], m['acc']),
        textcoords='offset points',
        xytext=(dx, dy),
        fontsize=LEGEND_SIZE,
        bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.86),
    )

# Pareto frontier: Standard-32 -> Standard-64 -> CARD-84
pareto_nfe = [32, 64, 71]
pareto_acc = [29.3, 30.9, 34.9]
ax.plot(pareto_nfe, pareto_acc, '--', color=COLORS['ours'], alpha=0.5, lw=1.5,
        label='Pareto frontier')

# Dashed line showing DNB-84 = Standard-64
ax.plot([64, 83], [30.9, 30.9], ':', color=COLORS['highlight'], alpha=0.6, lw=1.2)
ax.annotate(
    'DNB = Std-64\n(+20 steps, +0 pp)',
    xy=(73, 30.5),
    fontsize=7,
    color=COLORS['highlight'],
    ha='center',
    bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.86),
)

ax.set_xlabel('NFE (Number of Forward Evaluations)', fontsize=FONT_SIZE)
ax.set_ylabel('GSM8K Accuracy (%)', fontsize=FONT_SIZE)
ax.set_title('Figure 5: Accuracy vs. Compute (LLaDA-8B / GSM8K)', fontsize=TITLE_SIZE)
ax.set_xlim(20, 140)
ax.set_ylim(28, 37)
ax.legend(fontsize=LEGEND_SIZE, loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig5_pareto.pdf')
fig.savefig(output_path, format='pdf', bbox_inches='tight')
print(f"Saved to {output_path}")
