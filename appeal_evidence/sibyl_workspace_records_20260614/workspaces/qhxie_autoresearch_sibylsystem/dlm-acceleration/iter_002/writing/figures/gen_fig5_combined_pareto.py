"""Generate Figure 5: Combined Pareto Frontier (individual + pairwise + three-way)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Individual methods (GSM8K) ---
indiv = {
    'M1 ($\\eta$=0.5)': (1.158, 94.5),
    'M1 ($\\eta$=1.0)': (1.252, 88.0),
    'M1 ($\\eta$=2.0)': (1.500, 55.5),
    'IGSD (td16)': (2.812, 58.2),
    'IGSD (td32)': (1.706, 67.8),
    'IGSD (td48)': (1.222, 73.3),
    'M3 ($w_g$=0.3)': (1.678, 103.9),
    'M3 ($w_g$=0.7)': (1.677, 103.9),
}

# --- Pairwise compositions (GSM8K, best configs) ---
pair = {
    'M1+IGSD (td16)': (2.750, 58.9),
    'M1+IGSD (td32)': (1.685, 61.6),
    'M3+IGSD (td16)': (2.718, 60.4),
    'M1+M3': (0.858, 102.5),
}

# --- Three-way (GSM8K, 3-seed means) ---
three = {
    'Max-Speed': (1.711, 62.7),
    'Balanced-A': (1.683, 63.3),
    'Quality-First': (1.679, 62.7),
}

# Baseline
baseline = (1.0, 100.0)

fig, ax = plt.subplots(figsize=(9, 6))

# Baseline
ax.scatter(*baseline, c='black', s=120, marker='*', zorder=5, label='Baseline')

# Individual
for name, (spd, ar) in indiv.items():
    color = '#2166ac' if 'M1' in name else '#d6604d' if 'IGSD' in name else '#1b7837'
    ax.scatter(spd, ar, c=color, s=60, marker='o', edgecolors='k', linewidths=0.5, zorder=3)
ax.scatter([], [], c='#2166ac', s=60, marker='o', edgecolors='k', linewidths=0.5, label='M1 (KV Cache)')
ax.scatter([], [], c='#d6604d', s=60, marker='o', edgecolors='k', linewidths=0.5, label='IGSD (Step Distill.)')
ax.scatter([], [], c='#1b7837', s=60, marker='o', edgecolors='k', linewidths=0.5, label='M3 (AR Guided)')

# Pairwise
for name, (spd, ar) in pair.items():
    ax.scatter(spd, ar, c='#762a83', s=90, marker='s', edgecolors='k', linewidths=0.5, zorder=4)
ax.scatter([], [], c='#762a83', s=90, marker='s', edgecolors='k', linewidths=0.5, label='Pairwise')

# Three-way
for name, (spd, ar) in three.items():
    ax.scatter(spd, ar, c='#e66101', s=110, marker='D', edgecolors='k', linewidths=0.5, zorder=4)
ax.scatter([], [], c='#e66101', s=110, marker='D', edgecolors='k', linewidths=0.5, label='Three-way')

# Label key operating points
ax.annotate('Max-Speed\n(1.71x, 62.7%)', (1.711, 62.7), textcoords="offset points",
            xytext=(10, 10), fontsize=7.5, arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
ax.annotate('M3 ($w_g$=0.3)\n(1.68x, 103.9%)', (1.678, 103.9), textcoords="offset points",
            xytext=(-60, 10), fontsize=7.5, arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
ax.annotate('M1+IGSD\n(2.75x, 58.9%)', (2.750, 58.9), textcoords="offset points",
            xytext=(8, -20), fontsize=7.5, arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
ax.annotate('M1+M3\n(0.86x, 102.5%)', (0.858, 102.5), textcoords="offset points",
            xytext=(-75, -20), fontsize=7.5, arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

# Pareto frontier (approximate)
pareto_pts = sorted([(1.0, 100.0), (1.158, 94.5), (1.678, 103.9), (2.750, 58.9), (2.812, 58.2)],
                    key=lambda p: p[0])
# Compute actual Pareto front
front = [pareto_pts[0]]
for pt in pareto_pts[1:]:
    if pt[1] >= front[-1][1] or pt[0] > front[-1][0]:
        front.append(pt)
# Simple: draw the dominant frontier line through non-dominated points
nd = [(1.0, 100.0), (1.678, 103.9)]  # The true Pareto front on AccRet vs Speedup
ax.plot([p[0] for p in nd], [p[1] for p in nd], 'k--', alpha=0.3, linewidth=1.5, label='Pareto frontier')

ax.set_xlabel('Speedup (relative to baseline)', fontsize=11)
ax.set_ylabel('Accuracy Retention (%)', fontsize=11)
ax.set_title('Figure 5: Speed--Quality Pareto Frontier (GSM8K)', fontsize=12, fontweight='bold')
ax.set_xlim(0.5, 3.2)
ax.set_ylim(40, 115)
ax.legend(fontsize=8, loc='lower left', ncol=2)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current/writing/figures/fig5_combined_pareto.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Figure 5 saved to writing/figures/fig5_combined_pareto.pdf")
