"""
Generate Figure 1: per-layer mean rho_t trajectories for all 7 methods.
Data source: exp/results/pilots/diagnostic_cifar10/*/trajectories.json
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from style_config import apply_style, COLORS, NAMES, MARKERS, DPI, LINE_WIDTH, MARKER_SIZE

apply_style()

# __file__ is: .../current/writing/figures/gen_rho_trajectories.py
# go up 2 levels to reach .../current/, then into exp/
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../current/writing/figures
_WORKSPACE_CURRENT = os.path.normpath(os.path.join(_CURRENT_DIR, '../..'))  # .../current
BASE = os.path.join(_WORKSPACE_CURRENT, 'exp/results/pilots/diagnostic_cifar10')

METHODS = ['FixedWD', 'CWD', 'SWD', 'CPR', 'DefazioCorrective', 'NoWD', 'UDWDC']

def load_mean_rho(method):
    """Load trajectory file and compute layer-averaged rho_t per epoch."""
    path = os.path.join(BASE, method, f'{method}_seed42_trajectories.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    layers = data.get('layers', {})
    # collect rho_t per layer
    rho_per_layer = []
    for layer_data in layers.values():
        rho = layer_data.get('rho_t', [])
        if rho:
            rho_per_layer.append(rho)
    if not rho_per_layer:
        return None
    arr = np.array(rho_per_layer)  # (n_layers, n_epochs)
    return arr.mean(axis=0)  # (n_epochs,)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.8),
                                gridspec_kw={'width_ratios': [2, 1]})

# ─── Left panel: rho_t trajectories ──────────────────────────────────────────
for method in METHODS:
    rho = load_mean_rho(method)
    if rho is None:
        continue
    epochs = np.arange(1, len(rho) + 1)
    color = COLORS.get(method, '#000000')
    name  = NAMES.get(method, method)
    marker = MARKERS.get(method, 'o')
    ax1.plot(epochs, rho, label=name, color=color,
             marker=marker, markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
             markevery=max(1, len(rho)//8))

ax1.set_xlabel('Epoch', fontsize=10)
ax1.set_ylabel(r'Mean $\rho_t^l$ (across layers)', fontsize=10)
ax1.set_title(r'(a) Per-Layer GW Ratio $\rho_t^l$ Trajectories', fontsize=11)
ax1.legend(fontsize=8.5, ncol=2, loc='upper right')
ax1.set_xlim(0.5, 10.5)

# ─── Right panel: box-plot at final epoch ────────────────────────────────────
# Focus on 4 methods that illustrate the spread
methods_box = ['FixedWD', 'CWD', 'CPR', 'UDWDC']
colors_box   = [COLORS[m] for m in methods_box]
names_box    = [NAMES[m] for m in methods_box]

box_data = []
for method in methods_box:
    path = os.path.join(BASE, method, f'{method}_seed42_trajectories.json')
    if not os.path.exists(path):
        box_data.append([])
        continue
    with open(path) as f:
        data = json.load(f)
    layers = data.get('layers', {})
    final_rho = []
    for layer_data in layers.values():
        rho = layer_data.get('rho_t', [])
        if rho:
            final_rho.append(rho[-1])
    box_data.append(final_rho)

bp = ax2.boxplot(box_data, patch_artist=True, notch=False,
                 widths=0.55, medianprops={'color': 'white', 'lw': 1.8})
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
for whisker in bp['whiskers']:
    whisker.set(color='#455A64', lw=1.1, linestyle='--')
for cap in bp['caps']:
    cap.set(color='#455A64', lw=1.1)
for flier in bp['fliers']:
    flier.set(marker='o', markersize=3, alpha=0.5, color='#455A64')

ax2.set_xticks(range(1, len(methods_box) + 1))
ax2.set_xticklabels(names_box, fontsize=9)
ax2.set_ylabel(r'$\rho_t^l$ at final epoch', fontsize=10)
ax2.set_title(r'(b) Per-Layer $\rho_t^l$ Distribution', fontsize=11)

plt.tight_layout(pad=0.8)

out_dir = os.path.dirname(__file__)
for ext in ('pdf', 'png'):
    plt.savefig(os.path.join(out_dir, f'rho_trajectories.{ext}'),
                dpi=DPI, bbox_inches='tight')
print("Saved rho_trajectories.pdf and .png")
plt.close()
