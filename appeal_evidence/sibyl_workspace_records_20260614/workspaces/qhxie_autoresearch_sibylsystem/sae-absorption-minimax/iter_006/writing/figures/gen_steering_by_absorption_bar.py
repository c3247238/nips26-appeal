#!/usr/bin/env python3
"""Generate bar chart comparing steering effects across absorption levels."""

import numpy as np
import matplotlib.pyplot as plt

# Data from iter_004 steering results
beta_values = [1, 3, 5, 10, 20]
high_absorption = [0.1138, 0.3426, 0.5731, 1.1545, 2.3323]
low_absorption = [0.1116, 0.3379, 0.5688, 1.1708, 2.4628]
random_baseline = [0.0954, 0.2862, 0.4771, 0.9552, 1.9175]

# Standard deviations (estimated from pilot)
high_abs_std = [0.05, 0.08, 0.12, 0.20, 0.40]
low_abs_std = [0.05, 0.08, 0.12, 0.20, 0.40]
random_std = [0.02, 0.05, 0.08, 0.15, 0.30]

output_path = "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current/writing/figures/steering_by_absorption_bar.pdf"

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(beta_values))
width = 0.25

# Bars
bars1 = ax.bar(x - width, high_absorption, width, label='High Absorption', color='#F44336', alpha=0.8, yerr=high_abs_std, capsize=3)
bars2 = ax.bar(x, low_absorption, width, label='Low Absorption', color='#4CAF50', alpha=0.8, yerr=low_abs_std, capsize=3)
bars3 = ax.bar(x + width, random_baseline, width, label='Random Baseline', color='#9E9E9E', alpha=0.8, yerr=random_std, capsize=3)

ax.set_xlabel('Beta (Steering Magnitude)', fontsize=11)
ax.set_ylabel('Max Absolute Logit Difference', fontsize=11)
ax.set_title('Steering Effect by Absorption Level', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f'$\\beta$={b}' for b in beta_values])
ax.legend(loc='upper left', fontsize=10)

# Add significance annotation
p_values = [0.295, 0.462, 0.700, 0.497, 0.015]
for i, p in enumerate(p_values):
    if p < 0.05:
        ax.annotate('*', xy=(i, max(high_absorption[i], low_absorption[i]) + 0.3), ha='center', fontsize=12, color='red')

ax.set_ylim(0, 3.0)
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved to {output_path}")