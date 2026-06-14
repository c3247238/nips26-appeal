#!/usr/bin/env python3
"""Generate Figure 1: Teaser plot - Exponential Saturation of LLM Reasoning Accuracy."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data from G1 saturation experiment (n=30, Qwen2.5-Math-7B-Instruct)
k_values = np.array([1, 2, 4, 8, 16])
accuracy_values = np.array([0.667, 0.767, 0.800, 0.833, 0.833])

# Fitted parameters from pilot
P_inf = 0.818
k0 = 0.613

# Exponential saturation model
def exp_saturation(k, P_inf, k0):
    return P_inf * (1 - np.exp(-k / k0))

# Smooth curve for plotting
k_smooth = np.linspace(0, 20, 200)
acc_smooth = exp_saturation(k_smooth, P_inf, k0)

# Create figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot empirical data points
ax.scatter(k_values, accuracy_values, s=100, color='#2E86AB', zorder=5,
           label='Observed accuracy (n=30)')

# Plot fitted curve
ax.plot(k_smooth, acc_smooth, color='#E94F37', linewidth=2.0,
        label=r'Exponential fit ($P_\infty$={:.3f}, $k_0$={:.3f})'.format(P_inf, k0),
        zorder=4)

# Reference lines
ax.axhline(y=P_inf, color='gray', linestyle='--', linewidth=1.2, alpha=0.6,
           label=r'Asymptotic ceiling $P_\infty$={:.3f}'.format(P_inf))
ax.axhline(y=0.75, color='#F5A623', linestyle=':', linewidth=1.5, alpha=0.8,
           label='75% routing threshold')

# Formatting
ax.set_xlabel('Number of samples ($k$)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Multi-sample reasoning exhibits exponential saturation', fontsize=13)
ax.set_xlim(-0.5, 18)
ax.set_ylim(0.55, 0.90)
ax.set_xticks([1, 2, 4, 8, 16])
ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3)

# Annotate diminishing returns
ax.annotate('Diminishing\nreturns', xy=(8, 0.833), xytext=(12, 0.72),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/writing/figures/fig1_teaser.pdf', dpi=150)
print("Saved: fig1_teaser.pdf")
