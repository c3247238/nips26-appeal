#!/usr/bin/env python3
"""Generate Figure 4: Absorption vs. Delta Steering Effectiveness."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load correlation with baseline data
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/exp/results/full/correlation_with_baseline.json') as f:
    corr_base = json.load(f)

# Load absorption rates
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/exp/results/full/full_absorption_gpt2_l8_absorption_rates.json') as f:
    l8_abs = json.load(f)
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/exp/results/full/full_absorption_gpt2_l4_absorption_rates.json') as f:
    l4_abs = json.load(f)

# Load random baseline
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/exp/results/full/ablation_random_baseline.json') as f:
    rand_data = json.load(f)

# Load feature-specific steering
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/exp/results/full/full_steering_probing_gpt2_l8_results.json') as f:
    l8_steer = json.load(f)
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/exp/results/full/full_steering_probing_gpt2_l4_results.json') as f:
    l4_steer = json.load(f)

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

layers_data = [
    ('Layer 4', l4_abs, l4_steer, rand_data['layer_results']['layer_8']['steering_results'], corr_base['H1b_steering_delta']['layer_4']),
    ('Layer 8', l8_abs, l8_steer, rand_data['layer_results']['layer_8']['steering_results'], corr_base['H1b_steering_delta']['layer_8']),
]

colors = ['#2E86AB', '#A23B72']

for idx, (layer_name, abs_data, steer_data, rand_results, h1b_stats) in enumerate(layers_data):
    ax = axes[idx]

    abs_rates = []
    delta_success = []
    labels = []

    abs_rates_dict = {k: v['absorption_rate'] for k, v in abs_data['absorption_rates'].items()}

    for feat in sorted(abs_rates_dict.keys()):
        abs_rate = abs_rates_dict[feat]

        # Feature-specific success at strength 50
        feat_succ = None
        for sr in steer_data['steering_results'][feat]['strength_results']:
            if sr['strength'] == 50.0:
                feat_succ = sr['success_rate']
                break

        # Random success at strength 50
        rand_succ = None
        if feat in rand_results:
            for sr in rand_results[feat]['strength_results']:
                if sr['strength'] == 50.0:
                    rand_succ = sr['success_rate']
                    break

        if feat_succ is not None and rand_succ is not None:
            delta = feat_succ - rand_succ
            abs_rates.append(abs_rate)
            delta_success.append(delta)
            labels.append(feat)

    abs_rates = np.array(abs_rates)
    delta_success = np.array(delta_success)

    # Scatter plot
    ax.scatter(abs_rates, delta_success, s=80, alpha=0.7,
               color=colors[idx], edgecolors='white', linewidth=0.5, zorder=3)

    # Add feature labels
    for i, feat in enumerate(labels):
        ax.annotate(feat, (abs_rates[i], delta_success[i]),
                    textcoords='offset points', xytext=(4, 4),
                    fontsize=7, alpha=0.8)

    # Regression line
    if len(abs_rates) > 1 and np.std(abs_rates) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(abs_rates, delta_success)
        x_line = np.linspace(0, max(abs_rates) * 1.1, 100)
        y_line = slope * x_line + intercept
        sig_marker = '*' if p_value < 0.05 else ''
        ax.plot(x_line, y_line, '--', color='gray', alpha=0.7, linewidth=1.2,
                label=f'r = {r_value:.3f}, p = {p_value:.3f}{sig_marker}')

    ax.set_xlabel('Absorption Rate')
    ax.set_ylabel('Delta Steering Success (feature - random)')
    ax.set_title(layer_name)
    ax.set_xlim(-0.01, max(abs_rates) * 1.15)
    ax.set_ylim(-0.35, 1.1)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)

    # Add significance annotation for layer 8
    if idx == 1:
        ax.annotate('Significant\n(p < 0.05)', xy=(0.15, 0.85), fontsize=8,
                    color='#C73E1D', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3F3', edgecolor='#C73E1D', alpha=0.8))

plt.suptitle('Absorption Rate vs. Delta Steering Effectiveness', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/writing/figures/fig4_absorption_vs_delta_steering.pdf',
            format='pdf')
plt.close()
print('Figure 4 (delta steering) saved.')
