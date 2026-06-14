"""
Figure 4: Absorption Heatmap (Hierarchy x Layer x SAE Width)
Shows the layer-hierarchy interaction visually.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from style_config import COLORS, FONT_SIZE, FIG_WIDTH_FULL, FIG_HEIGHT

# ---- Data ----
# First-letter absorption across all 8 configs (probe F1=0.97 everywhere)
firstletter = {
    ('L6', '16k'): 2.4, ('L6', '65k'): 2.4,
    ('L12', '16k'): 5.7, ('L12', '65k'): 9.2,
    ('L18', '16k'): 2.2, ('L18', '65k'): 4.5,
    ('L24', '16k'): 34.5, ('L24', '65k'): 25.5,
}

# Cross-domain at L24 only (best probe layer for RAVEL)
crossdomain_L24 = {
    ('city-continent', '16k'): 35.8, ('city-continent', '65k'): 26.0,
    ('city-country', '16k'): 18.5, ('city-country', '65k'): 12.7,
    ('city-language', '16k'): 13.6, ('city-language', '65k'): 13.6,
}

# Probe F1 at each layer for annotation
probe_f1 = {
    ('first-letter', 'L6'): 0.69, ('first-letter', 'L12'): 0.31,
    ('first-letter', 'L18'): 0.94, ('first-letter', 'L24'): 0.97,
    ('city-continent', 'L24'): 0.84,
    ('city-country', 'L24'): 0.79,
    ('city-language', 'L24'): 0.82,
}

hierarchies = ['first-letter', 'city-continent', 'city-country', 'city-language']
layers = ['L6', 'L12', 'L18', 'L24']
widths = ['16k', '65k']

fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT + 0.5),
                         sharey=True)

for w_idx, w in enumerate(widths):
    ax = axes[w_idx]
    matrix = np.full((len(hierarchies), len(layers)), np.nan)

    for h_idx, h in enumerate(hierarchies):
        for l_idx, l in enumerate(layers):
            if h == 'first-letter':
                key = (l, w)
                if key in firstletter:
                    matrix[h_idx, l_idx] = firstletter[key]
            else:
                if l == 'L24':
                    key = (h, w)
                    if key in crossdomain_L24:
                        matrix[h_idx, l_idx] = crossdomain_L24[key]

    # Use masked array for NaN
    masked = np.ma.masked_invalid(matrix)

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color='#F5F5F5')

    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=40, aspect='auto')

    # Annotate cells
    for h_idx in range(len(hierarchies)):
        for l_idx in range(len(layers)):
            val = matrix[h_idx, l_idx]
            if not np.isnan(val):
                color = 'white' if val > 20 else 'black'
                ax.text(l_idx, h_idx, f'{val:.1f}%', ha='center', va='center',
                        fontsize=FONT_SIZE - 1, fontweight='bold', color=color)
            else:
                ax.text(l_idx, h_idx, '--', ha='center', va='center',
                        fontsize=FONT_SIZE - 2, color='#AAAAAA')

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(hierarchies)))
    hier_labels = ['First-letter', 'City-continent', 'City-country', 'City-language']
    ax.set_yticklabels(hier_labels if w_idx == 0 else [])
    ax.set_xlabel('Model Layer')
    ax.set_title(f'JumpReLU {w}', fontsize=FONT_SIZE + 1)

# Colorbar
cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
cbar.set_label('Absorption Rate (%)', fontsize=FONT_SIZE)

plt.suptitle('', y=1.02)
plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), 'fig4_crossdomain_heatmap.pdf')
plt.savefig(outpath)
print(f'Saved: {outpath}')
plt.close()
