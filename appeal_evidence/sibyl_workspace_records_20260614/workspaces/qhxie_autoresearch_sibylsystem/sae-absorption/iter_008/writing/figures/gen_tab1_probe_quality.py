"""Generate Table 1: Probe Quality Heatmap (hierarchy x layer).

Reads probe training results and produces a heatmap showing F1 scores
across hierarchies and layers, with quality gate annotations.
"""
import sys
import os
import json

# Add project root for style_config import
WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(WORKSPACE, 'writing', 'figures')
sys.path.insert(0, FIGURES_DIR)

import style_config  # noqa: E402 -- applies global matplotlib/seaborn style
import matplotlib.pyplot as plt
import numpy as np

# Load probe training data
probe_file = os.path.join(WORKSPACE, 'exp', 'results', 'phase1', 'probe_training_full.json')
with open(probe_file) as f:
    data = json.load(f)

# Build the matrix
hierarchies = ['first-letter', 'city-continent', 'city-country', 'city-language']
hierarchy_labels = ['First-letter', 'City-continent', 'City-country', 'City-language']
layers = [6, 12, 18, 24]

# Map probe keys to (hierarchy, layer) -> F1
f1_matrix = np.zeros((len(hierarchies), len(layers)))
for i, h in enumerate(hierarchies):
    for j, l in enumerate(layers):
        # Try sklearn key first
        key = f"{h}-sklearn_L{l}" if h == 'first-letter' else f"{h}_L{l}"
        if key in data['probes']:
            f1_matrix[i, j] = data['probes'][key]['f1_weighted_cv']
        else:
            # Try sae-spelling key for first-letter
            key2 = f"{h}-sae-spelling_L{l}"
            if key2 in data['probes']:
                f1_matrix[i, j] = data['probes'][key2]['f1_weighted_cv']

# Create heatmap
fig, ax = plt.subplots(figsize=(style_config.FIG_WIDTH, 3.5))

# Custom colormap: red (<0.85) -> orange (0.85-0.90) -> green (>0.90)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('quality',
    [(0.0, style_config.COLORS['below_gate']),
     (0.7, style_config.COLORS['pass_relaxed']),
     (1.0, style_config.COLORS['pass_strict'])],
    N=256)

im = ax.imshow(f1_matrix, cmap=cmap, aspect='auto', vmin=0.3, vmax=1.0)

# Annotate cells
for i in range(len(hierarchies)):
    for j in range(len(layers)):
        val = f1_matrix[i, j]
        # Choose text color for contrast
        text_color = 'white' if val < 0.65 else 'black'
        # Add gate symbol
        if val >= 0.90:
            symbol = '\u2713'  # checkmark
        elif val >= 0.85:
            symbol = '~'
        else:
            symbol = ''
        text = f'{val:.2f}{symbol}'
        ax.text(j, i, text, ha='center', va='center',
                fontsize=style_config.FONT_SIZE, color=text_color, fontweight='bold')

ax.set_xticks(range(len(layers)))
ax.set_xticklabels([f'Layer {l}' for l in layers])
ax.set_yticks(range(len(hierarchies)))
ax.set_yticklabels(hierarchy_labels)

# Quality gate legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=style_config.COLORS['pass_strict'], label='Strict gate (F1 >= 0.90)'),
    Patch(facecolor=style_config.COLORS['pass_relaxed'], label='Relaxed gate (0.85 <= F1 < 0.90)'),
    Patch(facecolor=style_config.COLORS['below_gate'], label='Below gate (F1 < 0.85)'),
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=3, frameon=False, fontsize=style_config.FONT_SIZE - 2)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Weighted F1')

plt.tight_layout()
out_path = os.path.join(FIGURES_DIR, 'tab1_probe_quality.pdf')
fig.savefig(out_path)
print(f'Saved: {out_path}')
plt.close()
