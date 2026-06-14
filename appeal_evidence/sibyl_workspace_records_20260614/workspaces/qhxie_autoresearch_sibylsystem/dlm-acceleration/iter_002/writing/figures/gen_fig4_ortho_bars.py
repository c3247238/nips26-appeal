"""Generate Figure 4: Pairwise Orthogonality Bar Chart."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

pairs = ['M1+IGSD', 'M3+IGSD', 'M1+M3']

# Best config per pair from experimental data
gsm8k_ortho = [0.989, 0.959, 0.520]
math500_ortho = [0.637, 0.764, 0.318]
combined_ortho = [0.958, 0.841, 0.411]

x = np.arange(len(pairs))
width = 0.22

fig, ax = plt.subplots(figsize=(8, 4.5))

bars1 = ax.bar(x - width, gsm8k_ortho, width, label='GSM8K', color='#2166ac', edgecolor='k', linewidth=0.5)
bars2 = ax.bar(x, math500_ortho, width, label='MATH500', color='#d6604d', edgecolor='k', linewidth=0.5)
bars3 = ax.bar(x + width, combined_ortho, width, label='Combined', color='#4393c3', edgecolor='k', linewidth=0.5)

# Threshold lines
ax.axhline(y=1.0, color='#1b7837', linestyle='-', alpha=0.6, linewidth=1.2, label='Synergy (Ortho=1.0)')
ax.axhline(y=0.8, color='#f4a582', linestyle='--', alpha=0.6, linewidth=1.2, label='Near-orthogonal (Ortho=0.8)')

# Value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(pairs, fontsize=11)
ax.set_ylabel('Orthogonality Score', fontsize=11)
ax.set_title('Figure 4: Pairwise Composition Orthogonality', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.25)
ax.legend(fontsize=8.5, loc='upper right', ncol=2)
ax.grid(axis='y', alpha=0.2)

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current/writing/figures/fig4_ortho_bars.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print("Figure 4 saved to writing/figures/fig4_ortho_bars.pdf")
