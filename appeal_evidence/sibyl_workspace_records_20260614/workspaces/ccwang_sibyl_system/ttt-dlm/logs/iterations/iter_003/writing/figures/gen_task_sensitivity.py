"""
Generate Figure: Task-Dependent Effectiveness of DTA
Shows DTA's relative improvement (pp over vanilla) across three benchmarks,
highlighting the strong task dependence of parameter-space adaptation.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from pilot experiments (N=16 each)
benchmarks = ['Countdown\n(Constrained\nArithmetic)', 'GSM8K\n(Multi-step\nMath)', 'MBPP\n(Code\nGeneration)']
vanilla_acc = [12.5, 25.0, 25.0]
dta_acc = [6.25, 12.5, 37.5]
remdm_acc = [6.25, 37.5, None]  # MBPP ReMDM not tested in pilot
dta_remdm_acc = [6.25, 18.75, 12.5]

# Compute deltas relative to vanilla
dta_delta = [d - v for d, v in zip(dta_acc, vanilla_acc)]
remdm_delta = [r - v if r is not None else 0 for r, v in zip(remdm_acc, vanilla_acc)]

# Figure setup
fig, ax = plt.subplots(figsize=(7, 4.5))

x = np.arange(len(benchmarks))
width = 0.30

bars_dta = ax.bar(x - width/2, dta_delta, width, label='DTA',
                  color='#4C72B0', edgecolor='white', linewidth=0.8, zorder=3)
bars_remdm = ax.bar(x + width/2, remdm_delta, width, label='ReMDM-conf',
                    color='#DD8452', edgecolor='white', linewidth=0.8, zorder=3)

# Mark MBPP ReMDM as N/A
ax.text(2 + width/2, 0.5, 'N/A', ha='center', va='bottom', fontsize=8,
        fontstyle='italic', color='#999999')

# Annotate bars with values
for bar, val in zip(bars_dta, dta_delta):
    y = bar.get_height()
    offset = 0.5 if y >= 0 else -1.5
    ax.text(bar.get_x() + bar.get_width()/2, y + offset,
            f'{val:+.1f}pp', ha='center', va='bottom' if y >= 0 else 'top',
            fontsize=9, fontweight='bold', color='#4C72B0')

for bar, val, r in zip(bars_remdm, remdm_delta, remdm_acc):
    if r is None:
        continue
    y = bar.get_height()
    offset = 0.5 if y >= 0 else -1.5
    ax.text(bar.get_x() + bar.get_width()/2, y + offset,
            f'{val:+.1f}pp', ha='center', va='bottom' if y >= 0 else 'top',
            fontsize=9, fontweight='bold', color='#DD8452')

# Styling
ax.set_ylabel('Accuracy Change vs. Vanilla (pp)', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=9)
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
ax.set_ylim(-16, 18)
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_title('Task-Dependent Effectiveness (Pilot, N=16)', fontsize=12, pad=10)

# Add note
fig.text(0.5, -0.02, 'Pilot-scale results (N=16 per benchmark). Effect sizes may differ at full scale.',
         ha='center', fontsize=8, fontstyle='italic', color='#666666')

plt.tight_layout()
plt.savefig('/Users/cwan0785/sibyl-system/workspaces/ttt-dlm/writing/figures/task_sensitivity.pdf',
            bbox_inches='tight', dpi=300)
plt.savefig('/Users/cwan0785/sibyl-system/workspaces/ttt-dlm/writing/figures/task_sensitivity.png',
            bbox_inches='tight', dpi=150)
print("Saved task_sensitivity.pdf and task_sensitivity.png")
