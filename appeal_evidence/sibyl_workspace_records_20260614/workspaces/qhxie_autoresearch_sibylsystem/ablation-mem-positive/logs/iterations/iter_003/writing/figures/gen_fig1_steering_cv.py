#!/usr/bin/env python3
"""
Generate Figure 1: Steering Effect by CV Group and Strength
Shows main result - high-CV features are more steerable than low-CV at all strengths
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/exp/results/full_steering_cv.json', 'r') as f:
    data = json.load(f)

# Extract statistics
strengths = [3, 5, 7]
high_cv_means = []
high_cv_stds = []
low_cv_means = []
low_cv_stds = []

for s in strengths:
    stats = data['statistics_by_strength'][str(s)]
    high_cv_means.append(stats['high_cv_mean_abs_effect'])
    high_cv_stds.append(stats['high_cv_std'])
    low_cv_means.append(stats['low_cv_mean_abs_effect'])
    low_cv_stds.append(stats['low_cv_std'])

# Create figure
fig, ax = plt.subplots(figsize=(8, 5.5))

x = np.arange(len(strengths))
width = 0.35

bars1 = ax.bar(x - width/2, high_cv_means, width, yerr=high_cv_stds,
               label='High-CV (CV > 1.0)', color='#2E86AB', capsize=4, alpha=0.9)
bars2 = ax.bar(x + width/2, low_cv_means, width, yerr=low_cv_stds,
               label='Low-CV (CV ≤ 1.0)', color='#A23B72', capsize=4, alpha=0.9)

# Add significance markers
for i, s in enumerate(strengths):
    stats = data['statistics_by_strength'][str(s)]
    max_val = max(high_cv_means[i] + high_cv_stds[i], low_cv_means[i] + low_cv_stds[i])
    ax.annotate('**', xy=(i, max_val + 0.08), ha='center', fontsize=14, fontweight='bold')

ax.set_xlabel('Steering Strength (τ)', fontsize=12)
ax.set_ylabel('Mean Absolute Steering Effect (|Δlogit|)', fontsize=12)
ax.set_title('High-CV Absorbed Features Show Larger Steering Effects\nAcross All Steering Strengths', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'+{s}' for s in strengths])
ax.legend(loc='upper left', fontsize=10)
ax.set_ylim(0, 1.1)

# Add ratio annotations
for i, s in enumerate(strengths):
    stats = data['statistics_by_strength'][str(s)]
    ratio = stats['high_cv_mean_abs_effect'] / stats['low_cv_mean_abs_effect']
    ax.annotate(f'{ratio:.2f}x', xy=(i, 0.05), ha='center', fontsize=10,
                color='gray', style='italic')

# Add grid
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/writing/figures/fig1_steering_cv.pdf', dpi=150, bbox_inches='tight')
print("Figure 1 saved: fig1_steering_cv.pdf")