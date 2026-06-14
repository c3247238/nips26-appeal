"""
Generate pipeline architecture diagram for the Method section.
This figure shows the experimental pipeline: tokenization -> SAE hook ->
absorption scoring -> patching evaluation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis('off')
ax.set_title("Experimental Pipeline", fontsize=14, fontweight='bold', pad=10)

# Box style
box_style = dict(boxstyle="round,pad=0.3", facecolor="#e8f4f8", edgecolor="#2c3e50", linewidth=1.5)
arrow_style = dict(arrowstyle='->', color='#2c3e50', lw=1.8)

# Stage boxes (y positions)
stages = [
    (1.0, "1. Corpus\n(100 seqs × 128 tokens)\nmonology/pile-uncopyrighted"),
    (3.5, "2. Model Forward Pass\n+ SAE Hook\n(gpt2-small + gpt2-small-res-jb)"),
    (6.0, "3. Absorption Scoring\n$A_f$ per latent\n(co-firer RVE > 80%)"),
    (8.5, "4. Downstream Evaluation\nActivation Patching\n(Faithfulness Test)"),
]

for y, text in stages:
    bbox = FancyBboxPatch((0.3, y - 0.5), 3.4, 1.2,
                          boxstyle="round,pad=0.1",
                          facecolor="#e8f4f8",
                          edgecolor="#2c3e50",
                          linewidth=1.5)
    ax.add_patch(bbox)
    ax.text(2.0, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

# Arrows between stages
for i in range(3):
    y_start = stages[i][0] + 0.6
    y_end = stages[i+1][0] - 0.6
    ax.annotate('', xy=(2.0, y_end), xytext=(2.0, y_start),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.0))

# Side annotations (intermediate outputs)
ax.text(5.0, 2.3, "Cache residual\nactivations\n$x_t \\in \\mathbb{R}^{768}$",
        ha='left', va='center', fontsize=8, color='#555555',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#fffbe8', edgecolor='#d4a017', linewidth=1))

ax.text(5.0, 4.8, "Per-latent scores\n$\\{A_f\\}_{f=1}^{d_{sae}}$\n(top-5 co-firer RVE)",
        ha='left', va='center', fontsize=8, color='#555555',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#fffbe8', edgecolor='#d4a017', linewidth=1))

ax.text(5.0, 7.3, "Subset selection:\nLow-absorption latents\nvs. High-absorption latents",
        ha='left', va='center', fontsize=8, color='#555555',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#fffbe8', edgecolor='#d4a017', linewidth=1))

# H-label annotations
h_labels = [
    (8.2, 3.2, "H1, H3, H5:\nPer-latent scoring"),
    (8.2, 6.7, "H4:\nFaithfulness\ncomparison"),
]
for x, y, text in h_labels:
    ax.text(x, y, text, ha='left', va='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#e8f8e8', edgecolor='#27ae60', linewidth=1))

# Layer annotation
ax.text(0.5, 0.1, "Layers audited: $\\ell \in \\{0, 2, 4, 6, 8, 10\\}$\n$d_{{sae}} \\in \\{{2K, 8K, 24K\\}}$",
        ha='left', va='bottom', fontsize=8, color='#7f8c8d')

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/writing/figures/gen_pipeline.pdf',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: gen_pipeline.pdf")
