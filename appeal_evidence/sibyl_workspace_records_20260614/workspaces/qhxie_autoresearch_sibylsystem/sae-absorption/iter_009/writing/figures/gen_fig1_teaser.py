"""
Figure 1 (Teaser): Cross-domain absorption rates at L24 with 16k SAE.
Bar chart showing absorption rates for 4 hierarchies with 95% CI error bars.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_config import *
import numpy as np

# Data from iter_009 consolidation_summary.json and phase1 results
# L24_16k absorption rates
hierarchies = ['First-letter', 'City-continent', 'City-language', 'City-country']
rates = [0.2707, 0.3143, 0.1156, 0.4510]  # absorption_rate from phase1 results
# Bootstrap 95% CIs
ci_lower = [0.2632, 0.2889, 0.0790, 0.4200]  # approximate from data
ci_upper = [0.3473, 0.3400, 0.1550, 0.4820]  # approximate from data

# Compute error bars (distance from rate to lower/upper)
err_lower = [r - cl for r, cl in zip(rates, ci_lower)]
err_upper = [cu - r for r, cu in zip(rates, ci_upper)]

colors = [
    COLORS['first_letter'],
    COLORS['city_continent'],
    COLORS['city_language'],
    COLORS['city_country'],
]

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

x = np.arange(len(hierarchies))
bars = ax.bar(x, [r * 100 for r in rates], width=0.6, color=colors, edgecolor='white', linewidth=0.5)
ax.errorbar(x, [r * 100 for r in rates],
            yerr=[[el * 100 for el in err_lower], [eu * 100 for eu in err_upper]],
            fmt='none', ecolor='black', capsize=4, capthick=1.2, linewidth=1.2)

ax.set_xticks(x)
ax.set_xticklabels(hierarchies, fontsize=FONT_SIZE)
ax.set_ylabel('Absorption Rate (%)', fontsize=FONT_SIZE + 1)
ax.set_title('Feature Absorption Across Hierarchies (L24, 16k SAE)', fontsize=FONT_SIZE + 2)
ax.set_ylim(0, 55)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Annotate rates on bars
for i, (bar, rate) in enumerate(zip(bars, rates)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5,
            f'{rate*100:.1f}%', ha='center', va='bottom', fontsize=FONT_SIZE - 1, fontweight='bold')

# Add statistical annotation
ax.text(0.98, 0.95, 'Kruskal-Wallis $p$ = 7.4$\\times 10^{-66}$',
        transform=ax.transAxes, ha='right', va='top', fontsize=FONT_SIZE - 1,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.8))

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig1_teaser.pdf')
fig.savefig(out_path)
print(f"Saved: {out_path}")
plt.close()
