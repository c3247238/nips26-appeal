"""Generate Figure 4: 2x2 ablation grid for BSD and A-CFG hyperparameters."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
bar_color = '#4C72B0'
baseline_color = '#C44E52'
best_color = '#55A868'

# (a) BSD k-parameter ablation
ax = axes[0, 0]
configs = ['Vanilla', 'k=T/4\n(75% belief)', 'k=T/2\n(50% belief)', 'k=3T/4\n(25% belief)']
accs = [0.0, 0.0, 0.0, 6.2]
dist3 = [0.876, 0.899, 0.947, 0.913]
colors = [baseline_color, bar_color, bar_color, best_color]
bars = ax.bar(configs, accs, color=colors, edgecolor='black', linewidth=0.5)
for b, d in zip(bars, dist3):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
            f'd3={d:.2f}', ha='center', va='bottom', fontsize=7)
ax.set_ylabel('Accuracy (%)')
ax.set_title('(a) BSD k-parameter')
ax.set_ylim(0, 16)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# (b) BSD alpha schedule ablation
ax = axes[0, 1]
configs = ['linear\n(0.1→0.8)', 'cosine\n(0.1→0.8)', 'const\n(0.3)', 'const\n(0.5)']
accs = [6.2, 6.2, 6.2, 6.2]
dist3 = [0.913, 0.913, 0.913, 0.852]
colors = [best_color, bar_color, bar_color, bar_color]
bars = ax.bar(configs, accs, color=colors, edgecolor='black', linewidth=0.5)
for b, d in zip(bars, dist3):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
            f'd3={d:.2f}', ha='center', va='bottom', fontsize=7)
ax.axhline(y=0, color=baseline_color, linestyle='--', linewidth=1.0, alpha=0.7, label='Vanilla (0%)')
ax.set_title('(b) BSD alpha schedule')
ax.set_ylim(0, 16)
ax.legend(loc='upper right', framealpha=0.8)

# (c) A-CFG guidance weight ablation
ax = axes[1, 0]
configs = ['Vanilla', 'w=1.0', 'w=1.5', 'w=2.0']
accs = [0.0, 6.2, 12.5, 12.5]
dist3 = [0.876, 0.900, 0.889, 0.885]
colors = [baseline_color, bar_color, best_color, best_color]
bars = ax.bar(configs, accs, color=colors, edgecolor='black', linewidth=0.5)
for b, d in zip(bars, dist3):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
            f'd3={d:.2f}', ha='center', va='bottom', fontsize=7)
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Configuration')
ax.set_title('(c) A-CFG guidance weight')
ax.set_ylim(0, 16)

# (d) A-CFG temporal schedule ablation
ax = axes[1, 1]
configs = ['fixed', 'linear', 'cosine', 'threshold\n70/30']
accs = [12.5, 0.0, 0.0, 0.0]
dist3 = [0.889, 0.866, 0.875, 0.897]
colors = [best_color, bar_color, bar_color, bar_color]
bars = ax.bar(configs, accs, color=colors, edgecolor='black', linewidth=0.5)
for b, d in zip(bars, dist3):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
            f'd3={d:.2f}', ha='center', va='bottom', fontsize=7)
ax.set_xlabel('Configuration')
ax.set_title('(d) A-CFG temporal schedule (w=1.5)')
ax.set_ylim(0, 16)

plt.tight_layout()
out_path = '/Users/cwan0785/sibyl-system/workspaces/ttt-dlm/current/writing/figures/fig4_ablation_grid.pdf'
plt.savefig(out_path, bbox_inches='tight')
print(f"Saved: {out_path}")
