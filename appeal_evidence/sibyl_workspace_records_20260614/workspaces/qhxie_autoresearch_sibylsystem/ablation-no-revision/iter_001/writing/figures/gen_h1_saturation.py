#!/usr/bin/env python3
"""
Figure 3: H1 Saturation Curve with Exponential Fit
Accuracy vs. k with fitted Arrhenius kinetics model
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json

# Data from pilot experiments
k_values = np.array([1, 2, 4, 8, 16])
accuracy_values = np.array([0.667, 0.767, 0.800, 0.833, 0.833])

# Fitted parameters from pilot_summary_round4.json
P_inf = 0.818
k_0 = 0.613
R_squared = 0.936

# Exponential saturation model
def exponential_saturation(k, P_inf, k_0):
    return P_inf * (1 - np.exp(-k / k_0))

# Generate smooth curve for plotting
k_smooth = np.linspace(0, 18, 100)
acc_smooth = exponential_saturation(k_smooth, P_inf, k_0)

# Create figure
fig, ax = plt.subplots(figsize=(7, 5))

# Plot data points with error indication
ax.scatter(k_values, accuracy_values * 100, s=120, c='#2563eb', zorder=5,
           edgecolors='white', linewidths=1.5, label='Observed accuracy')

# Plot fitted curve
ax.plot(k_smooth, acc_smooth * 100, 'b-', linewidth=2.5, alpha=0.8,
        label=f'Fitted: $P_k = P_\\infty(1 - e^{{-k/k_0}})$')

# Add threshold and annotations
ax.axhline(y=75, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
ax.text(17, 76, '75% threshold', fontsize=9, color='gray', ha='right')

# Labels and title
ax.set_xlabel('Number of Samples ($k$)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title(f'H1: Arrhenius Kinetics Validation ($R^2 = {R_squared:.3f}$)', fontsize=13, fontweight='bold')

# Annotate fitted parameters
param_text = f'$P_\\infty = {P_inf:.3f}$\n$k_0 = {k_0:.3f}$'
ax.text(0.05, 0.12, param_text, transform=ax.transAxes, fontsize=10,
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Configure axes
ax.set_xlim(-0.5, 18)
ax.set_ylim(50, 95)
ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=10)

# Add confirmation badge
ax.text(0.95, 0.95, 'CONFIRMED', transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='#22c55e', alpha=0.8),
        color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/iter_001/writing/figures/h1_saturation.pdf', dpi=150, bbox_inches='tight')
print("Saved: h1_saturation.pdf")
