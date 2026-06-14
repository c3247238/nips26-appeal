"""
Figure 3: First-Letter Absorption Across Layers
Grouped bar chart: 4 layers x 2 widths, y = absorption rate, error bars = 95% bootstrap CI.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load style
from style_config import COLORS, FONT_SIZE, FIG_WIDTH, FIG_HEIGHT

# Data from phase1_absorption_firstletter (pilot summary)
data = {
    'L6':  {'16k': 2.4, '65k': 2.4},
    'L12': {'16k': 5.7, '65k': 9.2},
    'L18': {'16k': 2.2, '65k': 4.5},
    'L24': {'16k': 34.5, '65k': 25.5},
}

# Bootstrap 95% CI from firstletter summary
ci = {
    'L6':  {'16k': (0.6, 14.4), '65k': (0.6, 14.7)},
    'L12': {'16k': (2.0, 8.1),  '65k': (4.1, 13.4)},
    'L18': {'16k': (0.4, 4.0),  '65k': (0.9, 8.1)},
    'L24': {'16k': (21.3, 49.5),'65k': (16.7, 38.3)},
}

layers = ['L6', 'L12', 'L18', 'L24']
widths = ['16k', '65k']

x = np.arange(len(layers))
width_bar = 0.35

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

for i, w in enumerate(widths):
    vals = [data[l][w] for l in layers]
    lo = [data[l][w] - ci[l][w][0] for l in layers]
    hi = [ci[l][w][1] - data[l][w] for l in layers]
    color = COLORS['l12'] if w == '16k' else COLORS['l18']
    label = f'JumpReLU {w}'
    hatch = '' if w == '16k' else '///'
    bars = ax.bar(x + i * width_bar - width_bar / 2, vals, width_bar,
                  yerr=[lo, hi], capsize=3, color=color, edgecolor='black',
                  linewidth=0.5, label=label, hatch=hatch, alpha=0.85)

ax.set_xlabel('Model Layer')
ax.set_ylabel('Absorption Rate (%)')
ax.set_xticks(x)
ax.set_xticklabels(layers)
ax.legend(frameon=True, framealpha=0.9)
ax.set_ylim(0, 55)

# Add value annotations on L24 bars
for i, w in enumerate(widths):
    val = data['L24'][w]
    ax.annotate(f'{val}%', xy=(3 + i * width_bar - width_bar / 2, val + 3),
                ha='center', va='bottom', fontsize=FONT_SIZE - 2, fontweight='bold')

# Horizontal reference line at Chanin et al. lower bound
ax.axhline(y=15, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.text(0.02, 16, 'Chanin et al. lower bound (15%)', fontsize=FONT_SIZE - 2,
        color='gray', alpha=0.7)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), 'fig3_layer_absorption.pdf')
plt.savefig(outpath)
print(f'Saved: {outpath}')
plt.close()
