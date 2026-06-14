#!/usr/bin/env python3
"""Generate Figure 3: AUROC Distribution Across GemmaScope Layers"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/current/exp/results/pilots/e3v2_semantic_scaled.json', 'r') as f:
    data = json.load(f)

# Extract AUROC values per layer
layers = [5, 10, 15]
layer_names = ['Layer 5', 'Layer 10', 'Layer 15']
auroc_data = []

for sae_id in data['per_sae']:
    sae_data = data['per_sae'][sae_id]
    aurocs = [sae_data['probe_results'][cat]['test_auroc'] for cat in sae_data['probe_results']]
    auroc_data.append(aurocs)

# Create figure
fig, ax = plt.subplots(figsize=(6, 4.5))

# Boxplot
bp = ax.boxplot(auroc_data, labels=layer_names, patch_artist=True,
                widths=0.5, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=6))

# Style boxes
colors = ['#4472C4', '#ED7D31', '#70AD47']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Style whiskers, caps, medians
for whisker in bp['whiskers']:
    whisker.set(color='black', linewidth=1.0)
for cap in bp['caps']:
    cap.set(color='black', linewidth=1.0)
for median in bp['medians']:
    median.set(color='black', linewidth=1.5)

# Validity threshold line
ax.axhline(y=0.7, color='red', linestyle='--', linewidth=1.2, label='Validity threshold (AUROC = 0.7)')

# Labels and styling
ax.set_ylabel('AUROC', fontsize=11)
ax.set_xlabel('GemmaScope Layer', fontsize=11)
ax.set_title('Probe Quality Distribution Across GemmaScope Layers', fontsize=12, fontweight='bold')
ax.set_ylim(0.82, 1.02)
ax.legend(loc='lower left', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Add annotation
ax.annotate('All probes exceed\nvalidity threshold', xy=(2, 0.95), fontsize=9,
            ha='center', style='italic', color='darkgreen')

plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/current/writing/figures/fig3_auroc.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive/current/writing/figures/fig3_auroc.png', dpi=300, bbox_inches='tight')
print("Figure 3 saved.")
