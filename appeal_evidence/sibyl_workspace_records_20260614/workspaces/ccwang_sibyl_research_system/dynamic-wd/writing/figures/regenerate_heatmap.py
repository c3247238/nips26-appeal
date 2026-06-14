"""Regenerate Figure 5 (diagnostic heatmap) with corrected BEM for half_lambda."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

methods = ['constant', 'cosine_schedule', 'random_mask',
           'half_lambda', 'no_wd', 'cwd_hard', 'swd']
# CSI, AIS, BEM  -- BEM for half_lambda CORRECTED to 0.500
raw = np.array([
    [0.841, 0.336, 0.000],  # constant
    [0.936, 0.352, 0.503],  # cosine_schedule
    [0.892, 0.359, 0.500],  # random_mask
    [0.853, 0.410, 0.500],  # half_lambda  <-- FIXED from 0.000
    [0.964, 0.343, 1.000],  # no_wd
    [0.851, 0.368, 0.503],  # cwd_hard
    [0.838, 0.360, 0.900],  # swd
])
metrics = ['CSI\n(Coupling Stability)', 'AIS\n(Alignment Info.)', 'BEM\n(Budget Equiv.)']
hm_labels = ['Constant', 'Cosine', 'Random Mask', r'Half-$\lambda$', 'No WD', 'CWD-Hard', 'SWD']

col_min = raw.min(axis=0)
col_max = raw.max(axis=0)
normalized = (raw - col_min) / (col_max - col_min + 1e-9)

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(normalized, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

ax.set_xticks(np.arange(len(metrics)))
ax.set_yticks(np.arange(len(methods)))
ax.set_xticklabels(metrics, fontsize=10)
ax.set_yticklabels(hm_labels, fontsize=10)

for i in range(len(methods)):
    for j in range(len(metrics)):
        val = raw[i, j]
        norm_val = normalized[i, j]
        text_color = 'white' if norm_val > 0.65 else '#222222'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=10, fontweight='bold', color=text_color)

cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
cbar.set_label('Normalized Value [0-1]', fontsize=10)
cbar.ax.tick_params(labelsize=9)

ax.set_xticks(np.arange(len(metrics)) - 0.5, minor=True)
ax.set_yticks(np.arange(len(methods)) - 0.5, minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
ax.tick_params(which='minor', bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.text(0.98, 1.02, 'CIFAR-10 / ResNet-20 / AdamW', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=10, color='#555555', style='italic')

plt.tight_layout(pad=1.0)
outdir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(outdir, 'fig4_diagnostic_heatmap.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(outdir, 'fig4_diagnostic_heatmap.pdf'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved fig4_diagnostic_heatmap.png/pdf with corrected BEM")
