#!/usr/bin/env python3
"""
Figure 6: Hypothesis Summary
Visual summary of all three hypothesis results
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Hypothesis data
hypotheses = ['H1: Arrhenius\nKinetics', 'H2: $E_a$ =\nDifficulty', 'H3: Single-Pass\nThreshold']
metrics = ['$R^2$', 'Spearman $\\rho$', 'Accuracy']
thresholds = [0.85, 0.40, 75.0]
actual_values = [0.936, 0.578, 68.4]
units = ['', '', '%']
statuses = ['CONFIRMED', 'CONFIRMED', 'FALSIFIED']
status_colors = ['#22c55e', '#22c55e', '#ef4444']

# Create figure
fig, ax = plt.subplots(figsize=(9, 4.5))

# Create table-like visualization
ax.axis('off')

# Table headers
headers = ['Hypothesis', 'Metric', 'Threshold', 'Actual', 'Status']
col_widths = [0.28, 0.18, 0.18, 0.18, 0.18]
x_starts = np.cumsum([0.02] + col_widths[:-1])

# Draw header row
y_top = 0.85
y_bottom = 0.65
y_mid = (y_top + y_bottom) / 2

for i, (header, x_start, width) in enumerate(zip(headers, x_starts, col_widths)):
    rect = plt.Rectangle((x_start, y_top), width - 0.01, y_bottom - y_top,
                         facecolor='#374151', edgecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x_start + width/2, y_mid, header, ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

# Draw data rows
for row_idx, (hyp, metric, thresh, actual, status, color) in enumerate(
        zip(hypotheses, metrics, thresholds, actual_values, statuses, status_colors)):

    y_row_top = y_bottom - row_idx * 0.2
    y_row_bottom = y_row_top - 0.18
    y_row_mid = (y_row_top + y_row_bottom) / 2

    # Row background
    bg_color = '#f3f4f6' if row_idx % 2 == 0 else 'white'

    # Format actual value
    if isinstance(actual, float):
        if thresh == 0.85:  # R^2
            actual_str = f'{actual:.3f}'
        elif thresh == 0.40:  # Spearman
            actual_str = f'{actual:.3f}'
        else:  # percentage
            actual_str = f'{actual:.1f}%'
    else:
        actual_str = str(actual)

    # Format threshold
    if thresh == 0.85:
        thresh_str = f'${thresh:.2f}$'
    elif thresh == 0.40:
        thresh_str = f'${thresh:.2f}$'
    else:
        thresh_str = f'${thresh:.0f}$%'

    # Draw cells
    cells = [hyp, metric, thresh_str, actual_str, status]
    for col_idx, (cell, x_start, width) in enumerate(zip(cells, x_starts, col_widths)):
        rect = plt.Rectangle((x_start, y_row_bottom), width - 0.01, y_row_top - y_row_bottom,
                             facecolor=bg_color, edgecolor='#d1d5db', linewidth=0.5)
        ax.add_patch(rect)

        # Cell text
        if col_idx == 4:  # Status column
            text_color = color
            fontweight = 'bold'
        elif col_idx == 3:  # Actual value
            text_color = '#dc2626' if status == 'FALSIFIED' else '#059669'
            fontweight = 'bold'
        else:
            text_color = 'black'
            fontweight = 'normal'

        ax.text(x_start + width/2, y_row_mid, cell, ha='center', va='center',
                fontsize=10, color=text_color, fontweight=fontweight)

# Add title
ax.text(0.5, 0.97, 'Hypothesis Testing Summary', ha='center', va='top',
        fontsize=14, fontweight='bold', transform=ax.transAxes)

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='#22c55e', label='Confirmed'),
    mpatches.Patch(facecolor='#ef4444', label='Falsified')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)

# Add key insight
insight = "Key Finding: Exponential saturation is real ($R^2=0.936$)\nbut doesn't enable routing (68.4% < 75% threshold)"
ax.text(0.5, 0.05, insight, ha='center', va='bottom',
        fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='#fef3c7', alpha=0.8, edgecolor='#f59e0b'))

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/iter_001/writing/figures/hypothesis_summary.pdf', dpi=150, bbox_inches='tight')
print("Saved: hypothesis_summary.pdf")
