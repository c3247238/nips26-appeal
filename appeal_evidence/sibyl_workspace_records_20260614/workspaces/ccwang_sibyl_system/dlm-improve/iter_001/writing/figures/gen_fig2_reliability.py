"""
Generate Figure 2: Reliability diagrams for SC vs Oracle calibration.
2x4 panel grid showing SC calibration at 4 stage bands (top row)
and Oracle calibration at 4 masking ratios (bottom row).
"""
import json
import sys
import os

# Add workspace to path for style config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_config import *
import numpy as np

# Load calibration data
workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(workspace, '..', 'exp', 'results', 'calibration_profile.json')) as f:
    data = json.load(f)

sc = data['method_a_self_consistency']['calibration_profile']
oracle = data['method_b_oracle']['calibration_profile']

fig, axes = plt.subplots(2, 4, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT * 2), sharey=True)

# SC panels (top row)
sc_stages = [
    ('early_80_100', '80-100% masked\n(early)'),
    ('mid_50_80', '50-80% masked\n(mid)'),
    ('mid_20_50', '20-50% masked\n(mid-late)'),
    ('late_0_20', '0-20% masked\n(late)'),
]

for i, (key, title) in enumerate(sc_stages):
    ax = axes[0, i]
    rd = sc[key]['reliability_diagram']
    confs = rd['bin_confidences']
    accs = rd['bin_accuracies']
    counts = rd['bin_counts']

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect')
    # SC curve
    ax.plot(confs, accs, 'o-', color=COLORS['sc'], markersize=5, lw=LINE_WIDTH,
            label=f'SC (ECE={sc[key]["ece"]:.3f})')
    ax.set_title(title, fontsize=FONT_SIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if i == 0:
        ax.set_ylabel('Actual Accuracy', fontsize=FONT_SIZE)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

# Oracle panels (bottom row)
oracle_stages = [
    ('mask_90pct', '90% masking'),
    ('mask_70pct', '70% masking'),
    ('mask_50pct', '50% masking'),
    ('mask_10pct', '10% masking'),
]

for i, (key, title) in enumerate(oracle_stages):
    ax = axes[1, i]
    rd = oracle[key]['reliability_diagram']
    confs = rd['bin_confidences']
    accs = rd['bin_accuracies']

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect')
    ax.plot(confs, accs, 's-', color=COLORS['oracle'], markersize=5, lw=LINE_WIDTH,
            label=f'Oracle (ECE={oracle[key]["ece"]:.3f})')
    ax.set_title(title, fontsize=FONT_SIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Predicted Confidence', fontsize=FONT_SIZE)
    if i == 0:
        ax.set_ylabel('Actual Accuracy', fontsize=FONT_SIZE)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

fig.suptitle('Figure 2: Reliability Diagrams — Self-Consistency (top) vs. Oracle (bottom)',
             fontsize=TITLE_SIZE, y=1.02)
plt.tight_layout()

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig2_reliability.pdf')
fig.savefig(output_path, format='pdf', bbox_inches='tight')
print(f"Saved to {output_path}")
