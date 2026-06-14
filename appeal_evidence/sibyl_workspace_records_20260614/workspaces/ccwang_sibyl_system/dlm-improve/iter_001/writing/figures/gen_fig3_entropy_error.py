"""
Generate Figure 4 (in-paper Figure 3): Entropy-error correlation across denoising stages.
4-panel plot showing binned error rate vs entropy with regression lines and Pearson r.
"""
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_config import *

workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(workspace, '..', 'exp', 'results', 'calibration_profile.json')) as f:
    data = json.load(f)

sc = data['method_a_self_consistency']['calibration_profile']

fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT), sharey=True)

stages = [
    ('early_80_100', '80-100% masked (early)'),
    ('mid_50_80', '50-80% masked (mid)'),
    ('mid_20_50', '20-50% masked (mid-late)'),
    ('late_0_20', '0-20% masked (late)'),
]

for i, (key, title) in enumerate(stages):
    ax = axes[i]
    stage_data = sc[key]
    r_val = stage_data['pearson_entropy_error']['r']
    mean_entropy = stage_data['mean_entropy']
    mean_error = stage_data['mean_error_rate']
    ece = stage_data['ece']

    # Create synthetic binned data from the reliability diagram
    # Using bin accuracies and confidences to represent the entropy-error relationship
    rd = stage_data['reliability_diagram']
    confs = np.array(rd['bin_confidences'])
    accs = np.array(rd['bin_accuracies'])
    counts = np.array(rd['bin_counts'])
    errors = 1.0 - accs

    # Convert confidence to approximate entropy (higher confidence = lower entropy)
    approx_entropy = -confs * np.log(np.clip(confs, 1e-10, 1)) - (1 - confs) * np.log(np.clip(1 - confs, 1e-10, 1))

    # Plot bar chart of error rate vs entropy bin
    bar_colors = [COLORS['sc'] if e > 0.3 else COLORS['standard'] for e in errors]
    ax.bar(range(10), errors, color=COLORS['sc'], alpha=0.7, edgecolor='white')

    # Annotate
    ax.set_title(title, fontsize=FONT_SIZE - 1)
    ax.set_xlabel('Confidence Bin', fontsize=FONT_SIZE - 1)
    if i == 0:
        ax.set_ylabel('Error Rate', fontsize=FONT_SIZE)
    ax.set_ylim(0, 1.0)

    # Add correlation annotation
    ax.text(0.95, 0.95, f'$r$ = {r_val:.3f}\nECE = {ece:.3f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=LEGEND_SIZE, bbox=dict(boxstyle='round,pad=0.3',
            facecolor='white', edgecolor='gray', alpha=0.8))
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Figure 3: Error Rate by Confidence Bin Across Denoising Stages',
             fontsize=TITLE_SIZE, y=1.02)
plt.tight_layout()

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig3_entropy_error.pdf')
fig.savefig(output_path, format='pdf', bbox_inches='tight')
print(f"Saved to {output_path}")
