#!/usr/bin/env python3
"""
Visualize cross-model validation results for Pythia-70m.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
results_dir = Path(__file__).parent
with open(results_dir / "cross_model_pythia_combined.json") as f:
    data = json.load(f)

layers = data['layers']
mean_by_layer = [data['cross_layer_summary']['mean_by_layer'][f'layer_{l}'] for l in layers]
max_by_layer = [data['cross_layer_summary']['max_by_layer'][f'layer_{l}'] for l in layers]
conditions_by_layer = [data['cross_layer_summary']['conditions_met_by_layer'][f'layer_{l}'] for l in layers]

features = list(data['layer_results']['layer_0']['absorption_rates'].keys())

# Create figure with 3 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Cross-Model Validation: Pythia-70m-deduped (19M params, 6 layers)', fontsize=14, fontweight='bold')

# Plot 1: Mean absorption rate by layer
ax1 = axes[0, 0]
colors = ['#3498db' if l != data['cross_layer_summary']['hotspot_layer_idx'] else '#e74c3c' for l in layers]
bars = ax1.bar([f'L{l}' for l in layers], mean_by_layer, color=colors, edgecolor='black', linewidth=1.2)
ax1.set_xlabel('Layer', fontsize=11)
ax1.set_ylabel('Mean Absorption Rate', fontsize=11)
ax1.set_title('Mean Absorption Rate by Layer', fontsize=12, fontweight='bold')
ax1.axhline(y=np.mean(mean_by_layer), color='gray', linestyle='--', alpha=0.7, label=f'Overall mean: {np.mean(mean_by_layer):.3f}')
ax1.legend()
ax1.set_ylim(0, max(mean_by_layer) * 1.3)
# Annotate hotspot
hotspot_idx = data['cross_layer_summary']['hotspot_layer_idx']
ax1.annotate(f'HOTSPOT\nL{hotspot_idx}', xy=(hotspot_idx, mean_by_layer[hotspot_idx]),
             xytext=(hotspot_idx, mean_by_layer[hotspot_idx] + 0.03),
             ha='center', fontsize=9, fontweight='bold', color='#e74c3c',
             arrowprops=dict(arrowstyle='->', color='#e74c3c'))

# Plot 2: Max absorption rate by layer
ax2 = axes[0, 1]
ax2.bar([f'L{l}' for l in layers], max_by_layer, color=colors, edgecolor='black', linewidth=1.2)
ax2.set_xlabel('Layer', fontsize=11)
ax2.set_ylabel('Max Absorption Rate', fontsize=11)
ax2.set_title('Max Absorption Rate by Layer', fontsize=12, fontweight='bold')
ax2.set_ylim(0, max(max_by_layer) * 1.2)

# Plot 3: Average conditions met by layer
ax3 = axes[1, 0]
ax3.bar([f'L{l}' for l in layers], conditions_by_layer, color=colors, edgecolor='black', linewidth=1.2)
ax3.set_xlabel('Layer', fontsize=11)
ax3.set_ylabel('Avg Conditions Met (of 3)', fontsize=11)
ax3.set_title('3-Condition Framework: Avg Conditions Met', fontsize=12, fontweight='bold')
ax3.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='1 condition threshold')
ax3.axhline(y=2.0, color='green', linestyle='--', alpha=0.7, label='2 conditions threshold')
ax3.legend()
ax3.set_ylim(0, 2.5)

# Plot 4: Heatmap of absorption rates (features x layers)
ax4 = axes[1, 1]
absorption_matrix = np.zeros((len(features), len(layers)))
for i, feat in enumerate(features):
    for j, layer in enumerate(layers):
        absorption_matrix[i, j] = data['layer_results'][f'layer_{layer}']['absorption_rates'][feat]['absorption_rate']

im = ax4.imshow(absorption_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.4)
ax4.set_xticks(range(len(layers)))
ax4.set_xticklabels([f'L{l}' for l in layers])
ax4.set_yticks(range(len(features)))
ax4.set_yticklabels(features)
ax4.set_xlabel('Layer', fontsize=11)
ax4.set_ylabel('Feature (First Letter)', fontsize=11)
ax4.set_title('Absorption Rate Heatmap', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax4, label='Absorption Rate')

plt.tight_layout()
plt.savefig(results_dir / 'cross_model_pythia_visualization.png', dpi=150, bbox_inches='tight')
print(f"Saved visualization to {results_dir / 'cross_model_pythia_visualization.png'}")

# Create comparison figure with GPT-2
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Cross-Model Comparison: GPT-2 vs Pythia-70m', fontsize=14, fontweight='bold')

# GPT-2 data (from previous experiments)
gpt2_layers = [0, 3, 6, 9, 11]
gpt2_mean = [0.045, 0.082, 0.156, 0.098, 0.067]  # Approximate from prior results

# Normalize by layer depth for comparison
gpt2_relative = [l / 12 for l in gpt2_layers]
pythia_relative = [l / 6 for l in layers]

ax = axes2[0]
ax.plot(gpt2_relative, gpt2_mean, 'o-', color='#3498db', linewidth=2, markersize=8, label='GPT-2 Small (124M)')
ax.plot(pythia_relative, mean_by_layer, 's-', color='#e74c3c', linewidth=2, markersize=8, label='Pythia-70m (19M)')
ax.set_xlabel('Relative Layer Depth (fraction of total layers)', fontsize=11)
ax.set_ylabel('Mean Absorption Rate', fontsize=11)
ax.set_title('Absorption Rate vs Relative Layer Depth', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Layer type comparison
ax = axes2[1]
layer_types = ['Early\n(L0-L1)', 'Middle\n(L2-L3)', 'Late\n(L4-L5)']
gpt2_early = np.mean([0.045, 0.082])
gpt2_mid = np.mean([0.156, 0.098])
gpt2_late = 0.067
pythia_early = np.mean(mean_by_layer[:2])
pythia_mid = np.mean(mean_by_layer[2:4])
pythia_late = np.mean(mean_by_layer[4:])

x = np.arange(len(layer_types))
width = 0.35
ax.bar(x - width/2, [gpt2_early, gpt2_mid, gpt2_late], width, label='GPT-2 Small', color='#3498db', edgecolor='black')
ax.bar(x + width/2, [pythia_early, pythia_mid, pythia_late], width, label='Pythia-70m', color='#e74c3c', edgecolor='black')
ax.set_ylabel('Mean Absorption Rate', fontsize=11)
ax.set_title('Absorption by Layer Group', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(layer_types)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(results_dir / 'cross_model_comparison.png', dpi=150, bbox_inches='tight')
print(f"Saved comparison to {results_dir / 'cross_model_comparison.png'}")
