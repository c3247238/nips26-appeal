"""
Generate Figure 4: Layer 4 Absorption Score Distribution (H3/H1 Analysis).
Shows the bimodal distribution of absorption scores at layer 4.
Data source: exp/results/pilots/h3_pilot.json (layer 4 absorption_scores)
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from h3_pilot.json, layer 4 results
# We extract layer 4 scores programmatically
import json
import os

path = 'iter_002/exp/results/pilots/h3_pilot.json'
with open(path) as f:
    data = json.load(f)

layer4_scores = None
for lr in data['layer_results']:
    if lr['layer'] == 4:
        layer4_scores = lr['absorption_scores']
        d_sae = lr.get('d_sae', 24576)
        break

if layer4_scores is None:
    print("ERROR: layer 4 data not found")
    exit(1)

print(f"Layer 4: n={len(layer4_scores)}, d_sae={d_sae}")
print(f"Exact 1.0 count: {sum(1 for s in layer4_scores if abs(s-1.0)<0.001)}")
print(f"Exact 0.0 count: {sum(1 for s in layer4_scores if abs(s-0.0)<0.001)}")

scores = np.array(layer4_scores)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: full histogram with 40 bins
ax1 = axes[0]
n, bins, patches = ax1.hist(scores, bins=40, color='#4a90d9', alpha=0.8, edgecolor='white')
ax1.set_xlabel('Absorption Score ($A_f$)', fontsize=11)
ax1.set_ylabel('Number of Latents', fontsize=11)
ax1.set_title('Layer 4: Full Distribution ($n = 24{,}576$)', fontsize=11)
ax1.set_xlim(0, 1)

# Annotate the peaks
ax1.annotate(f'$A_f = 1.0$\n6,176 latents (25.1%)', xy=(1.0, 6200), xytext=(0.75, 7000),
             fontsize=9, ha='center',
             arrowprops=dict(arrowstyle='->', color='#7f8c8d'),
             color='#e74c3c')
ax1.annotate(f'$A_f = 0.0$\n8,400 latents (34.2%)', xy=(0.0, 8400), xytext=(0.2, 9500),
             fontsize=9, ha='center',
             arrowprops=dict(arrowstyle='->', color='#7f8c8d'),
             color='#27ae60')

# Right: zoom on the middle region (0.1 < A_f < 0.9)
ax2 = axes[1]
middle_scores = scores[(scores > 0.1) & (scores < 0.9)]
n2, bins2, _ = ax2.hist(middle_scores, bins=30, color='#4a90d9', alpha=0.8, edgecolor='white')
ax2.set_xlabel('Absorption Score ($A_f$)', fontsize=11)
ax2.set_ylabel('Number of Latents', fontsize=11)
ax2.set_title('Layer 4: Zoomed View ($0.1 < A_f < 0.9$, $n = {}$)'.format(len(middle_scores)), fontsize=11)
ax2.set_xlim(0.1, 0.9)

# Add bimodality note
fig.text(0.5, -0.02,
         'Bimodal distribution: 6,176 latents (25.1%) at $A_f = 1.0$ and 8,400 (34.2%) at $A_f = 0.0$. '
         'Sharp clustering at boundary values suggests either genuine structural bifurcation or threshold artifact.',
         ha='center', va='top', fontsize=8.5, color='#555555')

fig.suptitle('Figure 4. Layer 4 Absorption Distribution Is Bimodal\n(H3 Results, Pilot, $d_{\\text{sae}} = 24{,}576$, 100 sequences)',
             fontsize=11, y=1.06)
plt.tight_layout()
plt.savefig('/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/writing/figures/fig4_layer4_histogram.pdf',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: fig4_layer4_histogram.pdf")