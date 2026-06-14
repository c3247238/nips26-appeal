#!/usr/bin/env python3
"""
Generate Figure 2: H_Mech 2x2 Factorial Bar Chart.

2x2 factorial: encoder (random/trained) x decoder (random/trained).
Shows absorption rate by condition with error bars (std).
Highlights Condition C's extreme variance and encoder sufficiency check (B ≈ D).
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

# Project root
PROJECT_ROOT = "/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "current/writing/figures")

# Professional academic color scheme
COLORS = {
    'condition_a': '#94A3B8',   # Slate
    'condition_b': '#3B82F6',   # Blue
    'condition_c': '#F59E0B',   # Amber
    'condition_d': '#10B981',   # Emerald
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def generate_figure2():
    """Generate H_Mech 2x2 factorial bar chart."""
    # Load full H_Mech results
    data_path = os.path.join(PROJECT_ROOT, "current/exp/results/full/h_mech_5seeds.json")
    data = load_json(data_path)

    # Extract condition summary
    summary = data['condition_summary']

    conditions = ['A', 'B', 'C', 'D']
    condition_labels = ['A\n(Random/\nRandom)', 'B\n(Trained/\nRandom)',
                        'C\n(Random/\nTrained)', 'D\n(Trained/\nTrained)']

    means = [summary[c]['mean'] for c in conditions]
    stds = [summary[c]['std'] for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(4)
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[COLORS['condition_a'], COLORS['condition_b'],
                         COLORS['condition_c'], COLORS['condition_d']],
                  width=0.6, alpha=0.85,
                  error_kw={'elinewidth': 1.5, 'capthick': 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, fontsize=10)
    ax.set_ylabel('Feature Absorption Rate', fontsize=12)
    ax.set_title('Figure 2: H_Mech 2x2 Factorial Design\nEncoder vs Decoder Contribution',
                  fontsize=13, fontweight='bold')

    # Set ylimit to accommodate C's extreme variance (max 43.84)
    # But show detail for other conditions
    ax.set_ylim(0, 15)

    # Add value labels
    for bar, val, std in zip(bars, means, stds):
        label = f'{val:.2f}' if val < 10 else f'{val:.1f}'
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.3,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add annotation for B ≈ D check
    max_h = max(means) + max(stds) + 1.5
    ax.plot([1, 1, 3, 3], [max_h, max_h + 0.5, max_h + 0.5, max_h], 'k-', lw=1.2)
    ax.text(2, max_h + 0.7, 'B ≈ D (delta = 0.037)\nEncoder sufficient',
            ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='gray', alpha=0.8))

    # Add annotation for C's extreme variance
    ax.text(2, 35, 'Condition C: extreme variance\n(std = 17.13, range 0-43.84)',
            ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.8))

    # Legend for conditions
    legend_elements = [
        Patch(facecolor=COLORS['condition_a'], alpha=0.85, label='A: Random/Random'),
        Patch(facecolor=COLORS['condition_b'], alpha=0.85, label='B: Trained/Random'),
        Patch(facecolor=COLORS['condition_c'], alpha=0.85, label='C: Random/Trained'),
        Patch(facecolor=COLORS['condition_d'], alpha=0.85, label='D: Trained/Trained'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=8, framealpha=0.9)

    plt.tight_layout()
    return fig


def save_figure(fig, name):
    """Save figure as both PDF and PNG."""
    pdf_path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
    fig.savefig(png_path, format='png', bbox_inches='tight', pad_inches=0.05, dpi=300)
    print(f"Saved: {pdf_path} and {png_path}")
    plt.close(fig)


def main():
    print("Generating Figure 2: H_Mech 2x2 Factorial...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        fig = generate_figure2()
        save_figure(fig, "figure2_h_mech_factorial")
        print("  ✓ figure2_h_mech_factorial saved successfully")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()