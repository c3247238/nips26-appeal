"""Generate Figure 3: Per-class absorption heatmap for city-continent."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from style_config import *
import numpy as np

# Data from full/phase1_absorption_crossdomain_summary.md
# city-continent per-class at L24_16k and L24_65k
classes = ['Africa', 'Asia', 'Europe', 'N. America', 'Oceania', 'S. America']

# Absorption rates
rates_16k = [0.0390, 0.2438, 0.9022, 0.1909, 0.5294, 0.0386]
rates_65k = [0.0519, 0.2469, 0.9203, 0.1411, 0.5490, 0.0386]

# Entity counts (n for each class)
n_16k = [278, 377, 315, 288, 78, 231]
n_65k = [278, 377, 315, 288, 78, 231]

# Create heatmap matrix: classes x [16k, 65k]
data = np.array([rates_16k, rates_65k]).T * 100  # percent

fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.75, FIG_HEIGHT + 0.5))

im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=100)

# Annotate cells with rate and n
for i in range(len(classes)):
    for j, (rates, ns) in enumerate([(rates_16k, n_16k), (rates_65k, n_65k)]):
        rate = rates[i] * 100
        n = ns[i]
        text_color = 'white' if rate > 50 else 'black'
        ax.text(j, i, f'{rate:.1f}%\n(n={n})',
                ha='center', va='center', fontsize=FONT_SIZE - 1,
                color=text_color, fontweight='bold')

ax.set_xticks([0, 1])
ax.set_xticklabels(['16k SAE', '65k SAE'], fontsize=FONT_SIZE)
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(classes, fontsize=FONT_SIZE)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Absorption Rate (%)', fontsize=FONT_SIZE)

ax.set_title('City-Continent Per-Class Absorption at L24', fontsize=FONT_SIZE + 1)

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), 'fig3_perclass_heatmap.pdf')
plt.savefig(outpath)
plt.close()
print(f'Saved: {outpath}')
