#!/usr/bin/env python3
"""Generate Figure 5: Steering Dose-Response Curves by Absorption Level."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/exp/results/full/full_steering_probing_gpt2_l4_results.json') as f:
    l4 = json.load(f)
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/exp/results/full/full_steering_probing_gpt2_l8_results.json') as f:
    l8 = json.load(f)
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/exp/results/full/correlation_report_full.json') as f:
    corr = json.load(f)

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

strengths = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

layers_data = [
    ('Layer 4', l4, corr['layer_results']['layer_4']['absorption_rates']),
    ('Layer 8', l8, corr['layer_results']['layer_8']['absorption_rates']),
]

colors = {'HIGH': '#C73E1D', 'MEDIUM': '#F18F01', 'LOW': '#2E86AB'}

for idx, (layer_name, data, abs_data) in enumerate(layers_data):
    ax = axes[idx]

    # Categorize features
    cats = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
    for feat in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        a = abs_data.get(feat, 0)
        if a >= 0.10:
            cats['HIGH'].append(feat)
        elif a >= 0.05:
            cats['MEDIUM'].append(feat)
        else:
            cats['LOW'].append(feat)

    for cat_name, cat_feats in cats.items():
        if not cat_feats:
            continue
        means = []
        for s in strengths:
            rates = []
            for feat in cat_feats:
                if feat in data['steering_results']:
                    for sr in data['steering_results'][feat]['strength_results']:
                        if sr['strength'] == s:
                            rates.append(sr['success_rate'])
                            break
            means.append(np.mean(rates) if rates else 0)

        label = f"{cat_name} ({'≥10%' if cat_name == 'HIGH' else '5–10%' if cat_name == 'MEDIUM' else '<5%'}), n={len(cat_feats)}"
        ax.plot(strengths, means, 'o-', color=colors[cat_name], linewidth=2.0,
                markersize=6, label=label, alpha=0.85)

    ax.set_xlabel('Steering Strength')
    ax.set_ylabel('Mean Steering Success Rate')
    ax.set_title(layer_name)
    ax.set_xscale('log')
    ax.set_xticks(strengths)
    ax.set_xticklabels([str(int(s)) if s == int(s) else str(s) for s in strengths])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)

plt.suptitle('Steering Dose-Response by Absorption Level', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/writing/figures/fig5_dose_response.pdf',
            format='pdf')
plt.close()
print('Figure 5 saved.')
