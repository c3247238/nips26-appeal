#!/usr/bin/env python3
"""Generate Figure 3: H1 validation — accuracy vs. k with exponential fit."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data from G1 saturation v2 (n=30, Qwen2.5-Math-7B-Instruct)
k_values = np.array([1, 2, 4, 8, 16])
accuracy_values = np.array([0.667, 0.767, 0.800, 0.833, 0.833])

# Fitted parameters
P_inf = 0.818
k0 = 0.613

def exp_saturation(k, P_inf, k0):
    return P_inf * (1 - np.exp(-k / k0))

k_smooth = np.linspace(0, 18, 200)
acc_smooth = exp_saturation(k_smooth, P_inf, k0)

fig, ax = plt.subplots(figsize=(6, 4))

# Empirical points
ax.scatter(k_values, accuracy_values, s=100, color='#2E86AB', zorder=5,
           label='Observed accuracy (n=30)')

# Fitted curve
ax.plot(k_smooth, acc_smooth, color='#E94F37', linewidth=2.0, zorder=4,
        label=r'Exponential fit ($R^2 = 0.936$)')

# Asymptotic ceiling
ax.axhline(y=P_inf, color='gray', linestyle='--', linewidth=1.2, alpha=0.6,
           label=r'$P_\infty = 0.818$')

# Routing threshold
ax.axhline(y=0.75, color='#F5A623', linestyle=':', linewidth=1.5, alpha=0.8,
           label='75% routing threshold')

# Diminishing returns annotation
ax.annotate('Diminishing\nreturns', xy=(8, 0.833), xytext=(11, 0.72),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray')

ax.set_xlabel('Number of samples ($k$)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('H1: Arrhenius Kinetics — Exponential Saturation Confirmed', fontsize=12)
ax.set_xlim(-0.5, 18)
ax.set_ylim(0.55, 0.90)
ax.set_xticks([1, 2, 4, 8, 16])
ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3)

outpath = '/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/writing/figures/fig3_h1_saturation.pdf'
plt.tight_layout()
plt.savefig(outpath, dpi=150)
print(f"Saved: {outpath}")
