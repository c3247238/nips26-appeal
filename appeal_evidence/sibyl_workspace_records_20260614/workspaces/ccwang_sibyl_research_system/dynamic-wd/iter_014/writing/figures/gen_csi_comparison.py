"""Generate Figure 7: CSI comparison across methods."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_config import apply_style, get_color, get_name, FIG_WIDTH, DPI
import matplotlib.pyplot as plt
import numpy as np

apply_style()

# CSI data from metrics_results.json (refined CSI)
methods = ['FixedWD', 'DefazioCorrective', 'CWD', 'CPR', 'SWD', 'UDWDC']
csi_temporal = [1.000, 1.000, 0.997, 0.957, 0.902, -5.75]
csi_spatial  = [1.000, 1.000, 0.997, 1.000, 0.907,  0.935]
csi_combined = [1.000, 1.000, 0.997, 0.978, 0.904, -2.408]

fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH * 1.8, 3.8), sharey=False)

titles = [r'CSI$_{\rm temporal}$', r'CSI$_{\rm spatial}$', r'CSI$_{\rm combined}$']
data_sets = [csi_temporal, csi_spatial, csi_combined]

for ax, title, data in zip(axes, titles, data_sets):
    colors = [get_color(m) for m in methods]
    display_names = [get_name(m) for m in methods]

    bars = ax.barh(range(len(methods)), data, color=colors, edgecolor='white', height=0.6)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(display_names, fontsize=9)
    ax.set_xlabel(title, fontsize=10)
    ax.axvline(x=0, color='black', linewidth=0.5, linestyle='-')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, data)):
        if val >= 0:
            ax.text(val + 0.05, i, f'{val:.3f}', va='center', fontsize=8)
        else:
            ax.text(val - 0.05, i, f'{val:.2f}', va='center', ha='right', fontsize=8)

    ax.set_xlim(min(min(data) - 1, -6.5), 1.3)

axes[0].set_title('Temporal Stability', fontsize=11)
axes[1].set_title('Spatial Stability', fontsize=11)
axes[2].set_title('Combined', fontsize=11)

plt.tight_layout()
out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, 'csi_comparison.pdf'), dpi=DPI)
fig.savefig(os.path.join(out_dir, 'csi_comparison.png'), dpi=DPI)
plt.close(fig)
print("Generated csi_comparison.pdf and csi_comparison.png")
