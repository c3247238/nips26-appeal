"""Generate Figure 7: Precision-Recall Decomposition (H5).

Two subplots:
- Left: Precision vs. absorption rate at k=5 (flat line near 1.0)
- Right: Recall vs. absorption rate at k=5 (scatter with regression line)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_config import COLORS, FONT_SIZE, FIG_WIDTH_FULL

# Load precision-recall data
with open('../../exp/results/full/precision_recall_analysis.json', 'r') as f:
    pr_data = json.load(f)

# Load absorption rates and per-feature precision/recall
with open('../../exp/results/full/full_steering_probing_gpt2_l4_results.json', 'r') as f:
    l4_data = json.load(f)
with open('../../exp/results/full/full_steering_probing_gpt2_l8_results.json', 'r') as f:
    l8_data = json.load(f)

# Load absorption rates from absorption results
with open('../../exp/results/full/full_absorption_gpt2_l4_absorption_rates.json', 'r') as f:
    l4_abs_data = json.load(f)
with open('../../exp/results/full/full_absorption_gpt2_l8_absorption_rates.json', 'r') as f:
    l8_abs_data = json.load(f)

# Extract absorption rates
l4_absorption = {k: v['absorption_rate'] for k, v in l4_abs_data['absorption_rates'].items()}
l8_absorption = {k: v['absorption_rate'] for k, v in l8_abs_data['absorption_rates'].items()}

# Extract precision and recall at k=5 from probing results
# k_results is a list of dicts with 'k', 'f1', 'precision', 'recall'
def get_k5_metrics(probing_results):
    """Extract k=5 precision and recall from probing results dict."""
    metrics = {}
    for letter, data in probing_results.items():
        k_results = data['k_results']
        for kr in k_results:
            if kr['k'] == 5:
                metrics[letter] = {'precision': kr['precision'], 'recall': kr['recall']}
                break
    return metrics

l4_pr = get_k5_metrics(l4_data['probing_results'])
l8_pr = get_k5_metrics(l8_data['probing_results'])

letters = sorted(l4_absorption.keys())

l4_abs = np.array([l4_absorption[l] for l in letters])
l4_prec = np.array([l4_pr[l]['precision'] for l in letters])
l4_rec = np.array([l4_pr[l]['recall'] for l in letters])
l8_abs = np.array([l8_absorption[l] for l in letters])
l8_prec = np.array([l8_pr[l]['precision'] for l in letters])
l8_rec = np.array([l8_pr[l]['recall'] for l in letters])

fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_FULL, 4.5))

# Left subplot: Precision vs Absorption
ax = axes[0]
ax.scatter(l4_abs, l4_prec, c=COLORS['layer4'], alpha=0.7, s=60, label=f'Layer 4 (std={0.054:.3f})', zorder=3)
ax.scatter(l8_abs, l8_prec, c=COLORS['layer8'], alpha=0.7, s=60, marker='s', label=f'Layer 8 (std={0.027:.3f})', zorder=3)

# Add horizontal reference lines at mean precision
ax.axhline(y=l4_prec.mean(), color=COLORS['layer4'], linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=l8_prec.mean(), color=COLORS['layer8'], linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlabel('Absorption Rate $A(f)$', fontsize=FONT_SIZE)
ax.set_ylabel('Precision at $k=5$', fontsize=FONT_SIZE)
ax.set_title('Precision vs. Absorption Rate', fontsize=FONT_SIZE + 1)
ax.set_xlim(-0.01, 0.27)
ax.set_ylim(0.75, 1.05)
ax.legend(loc='lower left', fontsize=FONT_SIZE - 1)
ax.grid(True, alpha=0.3)

# Add annotation
ax.text(0.14, 0.82, 'Precision is stable across\nabsorption levels. Layer 4: 21/26 = 1.0;\nLayer 8: 25/26 = 1.0.', fontsize=FONT_SIZE - 2,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Right subplot: Recall vs Absorption
ax = axes[1]
ax.scatter(l4_abs, l4_rec, c=COLORS['layer4'], alpha=0.7, s=60, label='Layer 4', zorder=3)
ax.scatter(l8_abs, l8_rec, c=COLORS['layer8'], alpha=0.7, s=60, marker='s', label='Layer 8', zorder=3)

# Regression lines
l4_slope, l4_intercept, l4_r, l4_p, _ = stats.linregress(l4_abs, l4_rec)
l8_slope, l8_intercept, l8_r, l8_p, _ = stats.linregress(l8_abs, l8_rec)

x_range = np.array([-0.01, 0.27])
ax.plot(x_range, l4_slope * x_range + l4_intercept, color=COLORS['layer4'], linestyle='-', alpha=0.6, linewidth=1.5)
ax.plot(x_range, l8_slope * x_range + l8_intercept, color=COLORS['layer8'], linestyle='-', alpha=0.6, linewidth=1.5)

ax.set_xlabel('Absorption Rate $A(f)$', fontsize=FONT_SIZE)
ax.set_ylabel('Recall at $k=5$', fontsize=FONT_SIZE)
ax.set_title('Recall vs. Absorption Rate', fontsize=FONT_SIZE + 1)
ax.set_xlim(-0.01, 0.27)
ax.set_ylim(0, 1.1)
ax.legend(loc='upper right', fontsize=FONT_SIZE - 1)
ax.grid(True, alpha=0.3)

# Add annotation
ax.text(0.14, 0.15, f'Recall varies widely and drives\nall F1 variance. Layer 4: $r={l4_r:.3f}$, $p={l4_p:.3f}$;\nLayer 8: $r={l8_r:.3f}$, $p={l8_p:.3f}$.', fontsize=FONT_SIZE - 2,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('fig7_precision_recall.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig7_precision_recall.png', dpi=300, bbox_inches='tight')
print("Saved fig7_precision_recall.pdf and .png")
