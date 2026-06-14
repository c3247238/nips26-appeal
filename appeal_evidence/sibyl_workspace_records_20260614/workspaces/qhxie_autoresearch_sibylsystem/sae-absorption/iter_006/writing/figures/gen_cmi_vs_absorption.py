"""Figure 4: CMI vs Absorption Rate scatter plot for 25 letters."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from style_config import *
import numpy as np

# Load CMI data at d'=10
results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'exp', 'results', 'full')
with open(os.path.join(results_dir, 'cmi_estimation.json')) as f:
    cmi_data = json.load(f)

letters = []
cmis = []
rates = []
absorbed_status = []

dim10 = cmi_data['cmi_by_subspace_dim']['10']
partition = cmi_data['partition']
absorbed_set = set(partition['absorbed_letters'])

for letter, info in dim10.items():
    if info['status'] == 'ok' and info['is_finite']:
        letters.append(letter)
        cmis.append(info['cmi'])
        rates.append(info['absorption_rate'] * 100)  # to percent
        absorbed_status.append(letter in absorbed_set)

cmis = np.array(cmis)
rates = np.array(rates)
absorbed_status = np.array(absorbed_status)

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT + 0.5))

# Scatter: absorbed vs non-absorbed
ax.scatter(cmis[absorbed_status], rates[absorbed_status],
           c=COLORS['absorbed'], s=60, alpha=0.8, edgecolors='white', linewidth=0.5,
           label=f'Absorbed (n={absorbed_status.sum()})', zorder=5)
ax.scatter(cmis[~absorbed_status], rates[~absorbed_status],
           c=COLORS['non_absorbed'], s=60, alpha=0.8, edgecolors='white', linewidth=0.5,
           label=f'Non-absorbed (n={(~absorbed_status).sum()})', zorder=5)

# Label each point with letter
for i, letter in enumerate(letters):
    offset_y = 1.5 if rates[i] > 5 else 1.0
    ax.annotate(letter, (cmis[i], rates[i]), xytext=(3, offset_y),
                textcoords='offset points', fontsize=8, alpha=0.7)

# Add group means
abs_mean_cmi = cmis[absorbed_status].mean()
nonabs_mean_cmi = cmis[~absorbed_status].mean()
abs_mean_rate = rates[absorbed_status].mean()
nonabs_mean_rate = rates[~absorbed_status].mean()

ax.axvline(x=abs_mean_cmi, color=COLORS['absorbed'], linestyle=':', alpha=0.5, linewidth=1)
ax.axvline(x=nonabs_mean_cmi, color=COLORS['non_absorbed'], linestyle=':', alpha=0.5, linewidth=1)

# Annotation box
textstr = (f'Spearman $\\rho$ = −0.383 (p = 0.059)\n'
           f"Cohen's d = −0.924\n"
           f'Mann-Whitney p = 0.042')
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right', bbox=props)

ax.set_xlabel('Conditional Mutual Information (CMI) at d\'=10')
ax.set_ylabel('Absorption Rate (%)')
ax.legend(loc='upper left', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
outpath = os.path.join(os.path.dirname(__file__), 'cmi_vs_absorption.pdf')
fig.savefig(outpath)
print(f"Saved: {outpath}")
plt.close(fig)
