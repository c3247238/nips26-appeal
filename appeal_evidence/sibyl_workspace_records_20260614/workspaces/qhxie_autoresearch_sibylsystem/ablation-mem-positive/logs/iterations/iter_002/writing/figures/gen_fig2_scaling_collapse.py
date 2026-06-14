#!/usr/bin/env python3
"""
Generate Figure 2: Scaling Collapse Plot
Shows finite-size scaling collapse across dictionary sizes with rescaled x-axis.

Data source: exp/results/full/dict_size_sweep.json
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load data
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/exp/results/full/dict_size_sweep.json', 'r') as f:
    data = json.load(f)

dict_sizes = [6144, 12288, 24576]
nu_values = [1, 2, 3]
collapse_results = data['scaling_collapse']['nu_results']
best_nu = data['scaling_collapse']['best_nu']  # 3
best_r2 = data['scaling_collapse']['best_collapse_quality']  # 0.951

# Extract data for each dictionary size
dict_results = data['dict_size_results']
lambda_thresholds_by_size = {}
absorption_rates_by_size = {}

for d_sae in dict_sizes:
    results = dict_results[str(d_sae)]['lambda_results']
    lambdas = [r['lambda_threshold'] for r in results]
    absorption = [r['absorption_rate'] for r in results]
    lambda_thresholds_by_size[d_sae] = lambdas
    absorption_rates_by_size[d_sae] = absorption

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot for each nu value (different line styles)
colors = ['blue', 'green', 'red']
linestyles = ['-', '--', ':']

# We'll show the best nu=3 case with the collapse
# For nu=3, rescale x-axis: lambda * N^(1/nu) = lambda * N^(1/3)
nu = 3
for i, d_sae in enumerate(dict_sizes):
    lambdas = np.array(lambda_thresholds_by_size[d_sae])
    absorption = np.array(absorption_rates_by_size[d_sae])

    # Rescale x-axis: lambda * N^(1/nu)
    x_rescaled = lambdas * (d_sae ** (1.0/nu))

    ax.plot(x_rescaled, absorption, 'o-', color=colors[i],
            linewidth=2, markersize=8, label=f'N = {d_sae}')

ax.set_xlabel(f'Rescaled Sparsity: λ × N¹/ⁿ (ν = {nu})', fontsize=12)
ax.set_ylabel('Absorption Rate m', fontsize=12)
ax.set_title(f'Finite-Size Scaling Collapse (ν = {nu})\nR² = {best_r2:.3f}', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)

# Add annotation for R^2
ax.text(0.05, 0.15, f'Best collapse: ν = {best_nu}\nR² = {best_r2:.3f}',
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/writing/figures/fig2_scaling_collapse.pdf',
            format='pdf', bbox_inches='tight', dpi=300)
print("Figure 2 saved to writing/figures/fig2_scaling_collapse.pdf")

# Also create a small bar chart showing R^2 for different nu values
fig2, ax2 = plt.subplots(figsize=(6, 4))
nus = [1, 2, 3]
r2_values = [collapse_results[str(nu)]['collapse_quality'] for nu in nus]
colors_bar = ['steelblue', 'forestgreen', 'crimson']
bars = ax2.bar(nus, r2_values, color=colors_bar, width=0.6)
ax2.set_xlabel('Critical Exponent ν', fontsize=12)
ax2.set_ylabel('Collapse Quality (R²)', fontsize=12)
ax2.set_title('Finite-Size Scaling: Collapse Quality by ν', fontsize=12, fontweight='bold')
ax2.set_ylim([0.8, 1.0])
ax2.axhline(y=0.9, color='gray', linestyle='--', linewidth=1, label='R² = 0.9 threshold')
ax2.legend()

# Highlight best nu
best_idx = nus.index(best_nu)
ax2.annotate(f'Best: ν = {best_nu}', xy=(nus[best_idx], r2_values[best_idx]),
             xytext=(best_nu + 0.3, r2_values[best_idx] + 0.02),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red')

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/writing/figures/fig2b_scaling_nu_comparison.pdf',
            format='pdf', bbox_inches='tight', dpi=300)
print("Figure 2b saved to writing/figures/fig2b_scaling_nu_comparison.pdf")