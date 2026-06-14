"""Generate Figure 6: IGSD T_draft Ablation (QAS and AccRet vs T_draft)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

t_draft = [16, 32, 48]

# From ablation data (tau=0.9, GSM8K pilot 200 samples)
gsm8k_speedup = [2.505, 1.706, 1.222]
gsm8k_accret = [60.3, 67.8, 73.3]
gsm8k_qas = [1.510, 1.157, 0.895]

# MATH500
math500_speedup = [2.311, 1.712, 1.254]
math500_accret = [31.3, 43.8, 50.0]
math500_qas = [0.722, 0.749, 0.627]

fig, ax1 = plt.subplots(figsize=(7, 4.5))

color_qas = '#2166ac'
color_acc = '#d6604d'

# QAS lines
ln1, = ax1.plot(t_draft, gsm8k_qas, 'o-', color=color_qas, linewidth=2, markersize=8,
                label='GSM8K QAS', zorder=3)
ln2, = ax1.plot(t_draft, math500_qas, 's--', color=color_qas, linewidth=1.5, markersize=7,
                alpha=0.6, label='MATH500 QAS', zorder=3)
ax1.set_xlabel('$T_{draft}$ (draft steps out of 64)', fontsize=11)
ax1.set_ylabel('Quality-Adjusted Speedup (QAS)', fontsize=11, color=color_qas)
ax1.tick_params(axis='y', labelcolor=color_qas)
ax1.set_ylim(0.4, 1.8)

# AccRet on right axis
ax2 = ax1.twinx()
ln3, = ax2.plot(t_draft, gsm8k_accret, '^-', color=color_acc, linewidth=2, markersize=8,
                label='GSM8K AccRet (%)', zorder=3)
ln4, = ax2.plot(t_draft, math500_accret, 'v--', color=color_acc, linewidth=1.5, markersize=7,
                alpha=0.6, label='MATH500 AccRet (%)', zorder=3)
ax2.set_ylabel('Accuracy Retention (%)', fontsize=11, color=color_acc)
ax2.tick_params(axis='y', labelcolor=color_acc)
ax2.set_ylim(20, 85)

# Mark Pareto-optimal
ax1.annotate('Pareto-optimal', (32, gsm8k_qas[1]), textcoords="offset points",
             xytext=(12, 12), fontsize=9, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# Legend
lns = [ln1, ln2, ln3, ln4]
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, fontsize=8.5, loc='upper right')

ax1.set_xticks(t_draft)
ax1.grid(alpha=0.2)
ax1.set_title('Figure 6: IGSD $T_{draft}$ Ablation ($\\tau$=0.9)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current/writing/figures/fig6_tdraft_ablation.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Figure 6 saved to writing/figures/fig6_tdraft_ablation.pdf")
