#!/usr/bin/env python3
"""Generate Figure 4: Absorption vs. Probing F1 Scatter."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load correlation data
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/exp/results/full/correlation_report_full.json') as f:
    data = json.load(f)

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

layers = [4, 8]
colors = ['#2E86AB', '#A23B72']

for idx, layer in enumerate(layers):
    ax = axes[idx]
    lr = data['layer_results'][f'layer_{layer}']

    abs_rates = []
    probing_f1 = []
    labels = []

    for feat in lr['absorption_rates']:
        abs_rates.append(lr['absorption_rates'][feat])
        probing_f1.append(lr['probing_f1_k5'][feat])
        labels.append(feat)

    abs_rates = np.array(abs_rates)
    probing_f1 = np.array(probing_f1)

    # Scatter plot
    ax.scatter(abs_rates, probing_f1, s=80, alpha=0.7,
               color=colors[idx], edgecolors='white', linewidth=0.5, zorder=3)

    # Add feature labels
    for i, feat in enumerate(labels):
        ax.annotate(feat, (abs_rates[i], probing_f1[i]),
                    textcoords='offset points', xytext=(4, 4),
                    fontsize=7, alpha=0.8)

    # Regression line
    if len(abs_rates) > 1 and np.std(abs_rates) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(abs_rates, probing_f1)
        x_line = np.linspace(0, max(abs_rates) * 1.1, 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, '--', color='gray', alpha=0.7, linewidth=1.2,
                label=f'r = {r_value:.3f}, p = {p_value:.3f}')

    ax.set_xlabel('Absorption Rate')
    ax.set_ylabel('Probing F1 (k = 5)')
    ax.set_title(f'Layer {layer}')
    ax.set_xlim(-0.01, max(abs_rates) * 1.15)
    ax.set_ylim(0.1, 1.05)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)

plt.suptitle('Absorption Rate vs. Sparse Probing F1', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/writing/figures/fig4_absorption_vs_probing.pdf',
            format='pdf')
plt.close()
print('Figure 4 saved.')
