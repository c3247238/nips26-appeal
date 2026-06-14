#!/usr/bin/env python3
"""
Generate Figure 1: Phase Transition Diagram
Shows quasi-critical phase transition with m(lambda) curve and susceptibility peak inset.

Data source: exp/results/full/sparsity_sweep_full.json
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/exp/results/full/sparsity_sweep_full.json', 'r') as f:
    data = json.load(f)

lambda_values = data['lambda_values']
absorption_rates = data['absorption_rates']
susceptibility_chi = data['susceptibility_chi']

# Critical point and metrics from data
critical_lambda = data['critical_lambda']  # 5e-05
max_susceptibility = data['max_susceptibility']  # 11.19
chi_ratio = data['phase_transition_analysis']['chi_ratio']  # 1.88

# Create figure with main plot and inset
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Main plot: absorption rate vs lambda
ax1.plot(lambda_values, absorption_rates, 'b-o', linewidth=2, markersize=8, label='Absorption rate m(λ)')
ax1.axvline(x=critical_lambda, color='red', linestyle='--', linewidth=2, label=f'λ_c = {critical_lambda:.0e}')
ax1.axvline(x=0.001, color='gray', linestyle=':', linewidth=1.5, label='λ = 0.001 (saturation)')
ax1.set_xscale('log')
ax1.set_xlabel('Sparsity Penalty λ', fontsize=12)
ax1.set_ylabel('Absorption Rate m(λ)', fontsize=12)
ax1.set_title('Phase Transition: Absorption Rate vs Sparsity', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([5e-6, 0.1])
ax1.set_ylim([0, 0.12])

# Add annotation for susceptibility peak
peak_idx = susceptibility_chi.index(max_susceptibility)
ax1.annotate(f'χ_max = {max_susceptibility:.2f}',
             xy=(lambda_values[peak_idx], absorption_rates[peak_idx]),
             xytext=(1e-4, 0.06),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red')

# Inset plot: susceptibility vs lambda
ax2.plot(lambda_values, susceptibility_chi, 'r-s', linewidth=2, markersize=8, label='Susceptibility χ')
ax2.axvline(x=critical_lambda, color='red', linestyle='--', linewidth=2, label=f'λ_c = {critical_lambda:.0e}')
ax2.scatter([critical_lambda], [max_susceptibility], color='red', s=150, zorder=5, marker='*', label=f'Peak χ = {max_susceptibility:.2f}')
ax2.set_xscale('log')
ax2.set_xlabel('Sparsity Penalty λ', fontsize=12)
ax2.set_ylabel('Susceptibility χ = dm/dλ', fontsize=12)
ax2.set_title('Susceptibility Peak (Inset)', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([5e-6, 0.1])

# Add chi_ratio annotation
ax2.text(0.15, 0.85, f'χ_ratio = {chi_ratio:.2f}\n(quasi-critical)',
         transform=ax2.transAxes, fontsize=11,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Overall title
fig.suptitle('Figure 1: Quasi-Critical Phase Transition in SAE Feature Absorption',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/writing/figures/fig1_phase_transition.pdf',
            format='pdf', bbox_inches='tight', dpi=300)
print("Figure 1 saved to writing/figures/fig1_phase_transition.pdf")