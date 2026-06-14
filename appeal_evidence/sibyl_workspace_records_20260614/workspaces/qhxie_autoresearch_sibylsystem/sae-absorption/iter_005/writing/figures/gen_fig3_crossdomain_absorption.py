"""Generate Figure 3: Cross-Domain Absorption Rates with Shuffled Controls.

Bar chart of mean absorption rates per domain (Country, Language, Continent)
across all layers, with shuffled control line at 100%.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Data from P2_crossdomain_comparison.json domain_summary
domains = ['Country\n(binary)', 'Country\n(top-10)', 'Language\n(binary)', 'Language\n(top-10)', 'Continent']
mean_rates = [0.512, 0.657, 0.848, 0.737, 0.828]
std_rates = [0.316, 0.143, 0.134, 0.144, 0.127]
colors = ['#4878CF', '#4878CF', '#6ACC65', '#6ACC65', '#D65F5F']

# Cosine-calibrated rates (all 0%)
cosine_rates = [0.0, 0.0, 0.0, 0.0, 0.0]

fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(domains))
width = 0.35

bars_dom = ax.bar(x - width/2, mean_rates, width, yerr=std_rates,
                  color=colors, alpha=0.85, edgecolor='white', linewidth=0.5,
                  capsize=4, label='Dominance-based')
bars_cos = ax.bar(x + width/2, cosine_rates, width,
                  color='#999999', alpha=0.5, edgecolor='white', linewidth=0.5,
                  label='Cosine-calibrated')

# Shuffled control line
ax.axhline(y=1.0, color='#B71C1C', linestyle='--', linewidth=2.0, alpha=0.8,
           label='Shuffled control (100%)')

# Value labels on dominance bars
for i, (v, s) in enumerate(zip(mean_rates, std_rates)):
    ax.text(i - width/2, v + s + 0.02, f'{v:.1%}', ha='center', va='bottom',
            fontsize=8, fontweight='bold')

ax.set_ylabel('Absorption rate', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=9)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=9, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Cross-Domain Absorption Rates (GPT-2 Small, Layers 5/8/11 Averaged)', fontsize=11, pad=12)

plt.tight_layout()

outdir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(outdir, 'fig3_crossdomain_absorption.pdf'), bbox_inches='tight', dpi=300)
fig.savefig(os.path.join(outdir, 'fig3_crossdomain_absorption.png'), bbox_inches='tight', dpi=300)
print(f"Saved fig3_crossdomain_absorption.pdf and .png to {outdir}")
plt.close()
