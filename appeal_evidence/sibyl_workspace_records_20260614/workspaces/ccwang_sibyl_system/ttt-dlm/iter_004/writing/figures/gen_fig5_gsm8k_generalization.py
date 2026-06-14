"""Generate Figure 5: Cross-benchmark generalization (Countdown-16 vs GSM8K-16)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
})

methods = ['Vanilla', 'DMI', 'BSD', 'A-CFG']
countdown = [0.0, 12.5, 6.2, 12.5]
gsm8k = [25.0, 25.0, 18.8, 37.5]

x = np.arange(len(methods))
width = 0.32

fig, ax = plt.subplots(figsize=(5.5, 4.0))

bars1 = ax.bar(x - width/2, countdown, width, label='Countdown-16',
               color='#4C72B0', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, gsm8k, width, label='GSM8K-16',
               color='#DD8452', edgecolor='black', linewidth=0.5)

# Add delta annotations for GSM8K relative to vanilla
vanilla_gsm = 25.0
for i, (c, g) in enumerate(zip(countdown, gsm8k)):
    delta_g = g - vanilla_gsm
    if delta_g != 0:
        sign = '+' if delta_g > 0 else ''
        ax.text(x[i] + width/2, g + 1.0, f'{sign}{delta_g:.1f}pp',
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color='#55A868' if delta_g > 0 else '#C44E52')

# Annotate A-CFG as strongest
ax.annotate('Only method with\ncross-benchmark gains',
            xy=(x[3], 37.5), xytext=(x[2] - 0.3, 42),
            arrowprops=dict(arrowstyle='->', color='#55A868', lw=1.5),
            fontsize=8, color='#55A868', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#55A868', alpha=0.8))

ax.set_ylabel('Accuracy (%)')
ax.set_title('Cross-Benchmark Generalization (n=16 pilot)')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(loc='upper left', framealpha=0.9)
ax.set_ylim(0, 50)
ax.axhline(y=25.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

plt.tight_layout()
out_path = '/Users/cwan0785/sibyl-system/workspaces/ttt-dlm/current/writing/figures/fig5_gsm8k_generalization.pdf'
plt.savefig(out_path, bbox_inches='tight')
print(f"Saved: {out_path}")
