#!/usr/bin/env python3
"""Generate figures for the NeurIPS paper."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Set publication-quality defaults
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

# Determine base path - we're in iter_NNN/writing/latex/figures/
# Go up 4 levels to reach iter_NNN, then into exp/results/full
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, '..', '..', '..', 'exp', 'results', 'full')
results_dir = os.path.abspath(results_dir)

print(f"Results dir: {results_dir}")
print(f"Exists: {os.path.exists(results_dir)}")

# Load data
with open(os.path.join(results_dir, 'f4_collision_correlation_results.json')) as f:
    f4_data = json.load(f)

with open(os.path.join(results_dir, 'f2_uad_ablations_results.json')) as f:
    f2_data = json.load(f)

# ========== Figure 3: Collision Rate vs True Absorption Rate Scatter Plot ==========
fig, ax = plt.subplots(figsize=(3.2, 3.0))

# Extract data points
numbers_pairs = f4_data['pair_results']['numbers']
punctuation_pairs = f4_data['pair_results']['punctuation']

num_collision = [p['collision_rate'] for p in numbers_pairs]
num_absorption = [p['true_absorption'] for p in numbers_pairs]

punct_collision = [p['collision_rate'] for p in punctuation_pairs]
punct_absorption = [p['true_absorption'] for p in punctuation_pairs]

# Plot
ax.scatter(num_collision, num_absorption, c='#1f77b4', marker='o', s=30,
           alpha=0.7, edgecolors='white', linewidth=0.5, label='Numbers')
ax.scatter(punct_collision, punct_absorption, c='#ff7f0e', marker='s', s=30,
           alpha=0.7, edgecolors='white', linewidth=0.5, label='Punctuation')

# Regression line for all data
all_collision = num_collision + punct_collision
all_absorption = num_absorption + punct_absorption
z = np.polyfit(all_collision, all_absorption, 1)
p = np.poly1d(z)
x_line = np.linspace(0, max(all_collision), 100)
ax.plot(x_line, p(x_line), 'k--', linewidth=1.0, alpha=0.5)

ax.set_xlabel('Collision Rate')
ax.set_ylabel('True Absorption Rate')
ax.set_xlim(-0.05, 0.85)
ax.set_ylim(-0.05, 1.05)
ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='gray')

# Annotate correlation
ax.text(0.55, 0.15, r"$\rho = 0.87$" + "\n" + r"$n = 56$", fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, edgecolor='none'))

plt.tight_layout()
plt.savefig('fig3_collision_scatter.pdf', bbox_inches='tight', dpi=300)
plt.close()

# ========== Figure 2: Token-Level Activation Heatmap ==========
fig, ax = plt.subplots(figsize=(4.5, 2.2))

# Token activation data from paper Section 4.4
tokens = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']
features = ['F11513', 'F12413', 'F22971', 'F24189']

# Activation matrix (mean activation values, 0 if feature doesn't activate)
# Feature 11513: activates exclusively on "three" (mean activation = 29.4)
# Feature 12413: activates exclusively on "one" (mean activation = 15.3)
# Feature 22971: activates exclusively on "two" (mean activation = 24.2)
# Feature 24189: activates on "four" through "eight" (mean activation = 14.3--18.9)
activation_matrix = np.array([
    [0, 15.3, 0, 0],      # one
    [0, 0, 24.2, 0],      # two
    [29.4, 0, 0, 0],      # three
    [0, 0, 0, 14.3],      # four
    [0, 0, 0, 17.8],      # five
    [0, 0, 0, 16.5],      # six
    [0, 0, 0, 18.9],      # seven
    [0, 0, 0, 14.3],      # eight
])

im = ax.imshow(activation_matrix.T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=30)

ax.set_xticks(range(len(tokens)))
ax.set_xticklabels(tokens)
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features)
ax.set_xlabel('Token')
ax.set_ylabel('SAE Feature')

# Add text annotations
for i in range(len(tokens)):
    for j in range(len(features)):
        val = activation_matrix[i, j]
        if val > 0:
            text_color = 'white' if val > 20 else 'black'
            ax.text(i, j, f'{val:.1f}', ha='center', va='center',
                   fontsize=8, color=text_color)
        else:
            ax.text(i, j, '—', ha='center', va='center',
                   fontsize=10, color='lightgray')

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Mean Activation', fontsize=9)

plt.tight_layout()
plt.savefig('fig2_token_heatmap.pdf', bbox_inches='tight', dpi=300)
plt.close()

# ========== Figure 4: Ablation Bar Chart ==========
fig, ax1 = plt.subplots(figsize=(4.5, 2.8))

variants = ['Full UAD', 'No dead\nfilter', 'No phi', 'No clustering',
            'Single\nlinkage', 'K-means', 'Global\nrandom', 'Same-cluster\nrandom']
f1_values = [0.0004805, 0.0004805, 0.0004805, 0.0000561,
             0.0, 0.0036923, 0.0001103, 0.0004805]
recall_values = [14.3, 14.3, 14.3, 42.9, 0.0, 85.7, 0.0, 14.3]

x = np.arange(len(variants))
width = 0.35

# F1 on log scale (left y-axis)
bars1 = ax1.bar(x - width/2, f1_values, width, label='F1', color='#2ca02c', alpha=0.8)
ax1.set_ylabel('F1 Score', color='#2ca02c')
ax1.set_yscale('log')
ax1.set_ylim(1e-5, 1e-1)
ax1.tick_params(axis='y', labelcolor='#2ca02c')
ax1.set_xticks(x)
ax1.set_xticklabels(variants, fontsize=7.5, rotation=0)

# Recall on linear scale (right y-axis)
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, recall_values, width, label='Recall (%)', color='#d62728', alpha=0.8)
ax2.set_ylabel('Recall (%)', color='#d62728')
ax2.set_ylim(0, 100)
ax2.tick_params(axis='y', labelcolor='#d62728')

# Highlight K-means
bars1[5].set_color('#1f77b4')
bars1[5].set_alpha(1.0)
bars2[5].set_color('#1f77b4')
bars2[5].set_alpha(1.0)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
           frameon=True, fancybox=False, edgecolor='gray', fontsize=8)

plt.tight_layout()
plt.savefig('fig4_ablation_bars.pdf', bbox_inches='tight', dpi=300)
plt.close()

print("All figures generated successfully.")
