#!/usr/bin/env python3
"""Generate Figure 2: Absorption Rates Across Layers."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/exp/results/full/full_absorption_gpt2_all_layers_combined.json') as f:
    data = json.load(f)

# Style settings
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

features = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
layers = [0, 4, 8, 10]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
layer_labels = ['Layer 0', 'Layer 4', 'Layer 8', 'Layer 10']

fig, ax = plt.subplots(figsize=(14, 4.5))

x = np.arange(len(features))
width = 0.2

for i, layer in enumerate(layers):
    lr = data['layer_results'][f'layer_{layer}']
    rates = [lr['absorption_rates'][f]['absorption_rate'] for f in features]
    bars = ax.bar(x + i * width - 1.5 * width, rates, width,
                  label=layer_labels[i], color=colors[i], alpha=0.85,
                  edgecolor='white', linewidth=0.3)

# Add 10% threshold line
ax.axhline(y=0.10, color='gray', linestyle='--', linewidth=1.0, alpha=0.7,
           label='10% threshold')

ax.set_xlabel('First-Letter Feature')
ax.set_ylabel('Absorption Rate')
ax.set_title('Absorption Rates Across Layers (GPT-2 Small, res-jb SAEs)')
ax.set_xticks(x)
ax.set_xticklabels(features)
ax.set_ylim(0, 0.28)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_axisbelow(True)
ax.yaxis.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/writing/figures/fig2_absorption_rates.pdf',
            format='pdf')
plt.close()
print('Figure 2 saved.')
