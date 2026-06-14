"""Generate Figure 3: Single-Method Pareto Curves for M1, IGSD, and M3."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# -- M1 data (eta sweep, GSM8K, seed 42) --
m1_eta = [0.5, 1.0, 2.0]
m1_speedup = [1.158, 1.252, 1.500]
m1_accret = [0.945, 0.880, 0.555]

# -- IGSD data (tau x T_draft, GSM8K, seed 42) --
igsd_labels = []
igsd_speedup = []
igsd_accret = []
igsd_marker = []

configs = [
    (0.7, 16, 2.812, 0.582), (0.7, 32, 1.755, 0.664), (0.7, 48, 1.231, 0.733),
    (0.85, 16, 2.612, 0.589), (0.85, 32, 1.732, 0.678), (0.85, 48, 1.232, 0.733),
    (0.9, 16, 2.505, 0.603), (0.9, 32, 1.706, 0.678), (0.9, 48, 1.222, 0.733),
]
for tau, td, spd, ar in configs:
    igsd_labels.append(f"td{td}")
    igsd_speedup.append(spd)
    igsd_accret.append(ar)

# -- M3 data (gw sweep, GSM8K, 2-seed mean) --
m3_gw = [0.3, 0.5, 0.7, 1.0]
m3_speedup = [1.678, 1.676, 1.677, 1.677]
m3_accret = [1.039, 1.032, 1.039, 0.849]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

# --- M1 subplot ---
ax = axes[0]
ax.scatter(m1_speedup, [a * 100 for a in m1_accret], c='#2166ac', s=80, zorder=3, edgecolors='k', linewidths=0.5)
for i, eta in enumerate(m1_eta):
    ax.annotate(f'$\\eta$={eta}', (m1_speedup[i], m1_accret[i]*100),
                textcoords="offset points", xytext=(6, -12), fontsize=8)
ax.plot(m1_speedup, [a * 100 for a in m1_accret], c='#2166ac', alpha=0.4, linewidth=1.2)
ax.set_xlabel('Speedup (relative to baseline)', fontsize=10)
ax.set_ylabel('Accuracy Retention (%)', fontsize=10)
ax.set_title('M1 (Entropy KV Cache)', fontsize=11, fontweight='bold')
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
ax.set_xlim(0.9, 1.7)
ax.set_ylim(40, 115)
ax.grid(alpha=0.2)

# --- IGSD subplot ---
ax = axes[1]
colors_td = {16: '#d6604d', 32: '#f4a582', 48: '#fddbc7'}
markers_td = {16: 'o', 32: 's', 48: '^'}
for tau_val in [0.7, 0.85, 0.9]:
    for td_val in [16, 32, 48]:
        idx = [(i, c) for i, c in enumerate(configs) if c[0] == tau_val and c[1] == td_val][0]
        i, (tau, td, spd, ar) = idx
        ax.scatter(spd, ar * 100, c=colors_td[td], s=80, marker=markers_td[td],
                   zorder=3, edgecolors='k', linewidths=0.5)

# Legend for T_draft
for td_val in [16, 32, 48]:
    ax.scatter([], [], c=colors_td[td_val], marker=markers_td[td_val], s=60,
               edgecolors='k', linewidths=0.5, label=f'$T_{{draft}}$={td_val}')
ax.legend(fontsize=8, loc='upper right')

# Connect T_draft=const lines for tau=0.9 as reference
for tau_val in [0.9]:
    pts = [(c[2], c[3] * 100) for c in configs if c[0] == tau_val]
    pts.sort(key=lambda p: p[0])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], c='gray', alpha=0.3, linewidth=1)

ax.set_xlabel('Speedup (relative to baseline)', fontsize=10)
ax.set_title('IGSD (Step Distillation)', fontsize=11, fontweight='bold')
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
ax.set_xlim(0.9, 3.2)
ax.grid(alpha=0.2)

# --- M3 subplot ---
ax = axes[2]
ax.scatter(m3_speedup, [a * 100 for a in m3_accret], c='#1b7837', s=80, zorder=3, edgecolors='k', linewidths=0.5)
for i, gw in enumerate(m3_gw):
    offset = (6, 6) if i < 3 else (6, -14)
    ax.annotate(f'$w_g$={gw}', (m3_speedup[i], m3_accret[i]*100),
                textcoords="offset points", xytext=offset, fontsize=8)
ax.set_xlabel('Speedup (relative to baseline)', fontsize=10)
ax.set_title('M3 (AR-Guided)', fontsize=11, fontweight='bold')
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
ax.set_xlim(1.4, 1.9)
ax.grid(alpha=0.2)

fig.suptitle('Figure 3: Single-Method Speed--Accuracy Tradeoffs on GSM8K', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current/writing/figures/fig3_single_pareto.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Figure 3 saved to writing/figures/fig3_single_pareto.pdf")
