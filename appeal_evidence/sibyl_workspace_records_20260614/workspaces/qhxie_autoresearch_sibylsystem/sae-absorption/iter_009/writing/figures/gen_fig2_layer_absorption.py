"""Generate Figure 2: Layer-dependent absorption profile across hierarchies."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from style_config import *
import numpy as np

# Data from phase1_absorption_firstletter.json (FULL mode, 500 words x 3 prompts)
# and consolidation_summary.json absorption_rate_table
layers = [6, 12, 18, 24]

# First-letter absorption rates by layer (16k JumpReLU SAEs)
fl_16k = [0.0105, 0.0470, 0.0202, 0.2707]
fl_65k = [0.0070, 0.0498, 0.0104, 0.1770]

# RAVEL hierarchies only measured at L24 (probes below gate at other layers)
# We plot first-letter as the full line and indicate RAVEL as L24 data points
cc_L24_16k = 0.3143  # city-continent
cl_L24_16k = 0.1156  # city-language
ck_L24_16k = 0.4510  # city-country

cc_L24_65k = 0.3128
cl_L24_65k = 0.0774
ck_L24_65k = 0.3292

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH_FULL, FIG_HEIGHT + 0.5),
                                gridspec_kw={'width_ratios': [3, 2]})

# Left panel: Layer-dependent absorption for first-letter
ax1.plot(layers, [r * 100 for r in fl_16k], 'o-',
         color=COLORS['first_letter'], label='First-letter (16k)',
         markersize=7, linewidth=LINE_WIDTH + 0.5, zorder=5)
ax1.plot(layers, [r * 100 for r in fl_65k], 's--',
         color=COLORS['first_letter'], alpha=0.6, label='First-letter (65k)',
         markersize=6, linewidth=LINE_WIDTH, zorder=4)

# Add RAVEL L24 points for 16k
ax1.scatter([24], [cc_L24_16k * 100], marker='D', s=80,
            color=COLORS['city_continent'], zorder=6, label='City-continent (16k)')
ax1.scatter([24], [cl_L24_16k * 100], marker='^', s=80,
            color=COLORS['city_language'], zorder=6, label='City-language (16k)')
ax1.scatter([24], [ck_L24_16k * 100], marker='v', s=80,
            color=COLORS['city_country'], zorder=6, label='City-country (16k)')

ax1.set_xlabel('Transformer Layer')
ax1.set_ylabel('Absorption Rate (%)')
ax1.set_xticks(layers)
ax1.set_xlim(4, 26)
ax1.set_ylim(-2, 50)
ax1.legend(fontsize=FONT_SIZE - 2, loc='upper left', framealpha=0.9)
ax1.set_title('(a) Layer-Dependent Absorption', fontsize=FONT_SIZE + 1)

# Annotate the L24 concentration
ax1.annotate(f'{fl_16k[-1]*100:.1f}%', xy=(24, fl_16k[-1]*100),
             xytext=(21, fl_16k[-1]*100 + 4), fontsize=FONT_SIZE - 2,
             color=COLORS['first_letter'],
             arrowprops=dict(arrowstyle='->', color=COLORS['first_letter'], lw=0.8))

# Right panel: Cross-domain comparison at L24 (bar chart, both widths)
hierarchies = ['First-\nletter', 'City-\ncontinent', 'City-\nlanguage', 'City-\ncountry']
rates_16k = [0.2707, 0.3143, 0.1156, 0.4510]
rates_65k = [0.1770, 0.3128, 0.0774, 0.3292]
ci_16k = [(0.2707-0.245, 0.295-0.2707),  # approximate from bootstrap
          (0.3143-0.289, 0.339-0.3143),
          (0.1156-0.097, 0.135-0.1156),
          (0.4510-0.422, 0.479-0.4510)]
ci_65k = [(0.1770-0.155, 0.200-0.1770),  # approximate
          (0.3128-0.287, 0.338-0.3128),
          (0.0774-0.062, 0.094-0.0774),
          (0.3292-0.302, 0.356-0.3292)]

x = np.arange(len(hierarchies))
width = 0.35
colors_h = [COLORS['first_letter'], COLORS['city_continent'],
            COLORS['city_language'], COLORS['city_country']]

bars1 = ax2.bar(x - width/2, [r*100 for r in rates_16k], width,
                color=colors_h, alpha=0.9, edgecolor='white', linewidth=0.5,
                yerr=[[c[0]*100 for c in ci_16k], [c[1]*100 for c in ci_16k]],
                capsize=3, error_kw={'linewidth': 1.0}, label='16k')
bars2 = ax2.bar(x + width/2, [r*100 for r in rates_65k], width,
                color=colors_h, alpha=0.5, edgecolor='white', linewidth=0.5,
                yerr=[[c[0]*100 for c in ci_65k], [c[1]*100 for c in ci_65k]],
                capsize=3, error_kw={'linewidth': 1.0}, label='65k')

ax2.set_ylabel('Absorption Rate (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(hierarchies, fontsize=FONT_SIZE - 2)
ax2.set_ylim(0, 55)
ax2.legend(fontsize=FONT_SIZE - 2, title='SAE Width', title_fontsize=FONT_SIZE - 2)
ax2.set_title('(b) Cross-Domain at L24', fontsize=FONT_SIZE + 1)

# Add rate labels on 16k bars
for i, r in enumerate(rates_16k):
    ax2.text(x[i] - width/2, r*100 + 2.5, f'{r*100:.1f}%',
             ha='center', fontsize=FONT_SIZE - 2, fontweight='bold')

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), 'fig2_layer_absorption.pdf')
plt.savefig(outpath)
plt.close()
print(f'Saved: {outpath}')
