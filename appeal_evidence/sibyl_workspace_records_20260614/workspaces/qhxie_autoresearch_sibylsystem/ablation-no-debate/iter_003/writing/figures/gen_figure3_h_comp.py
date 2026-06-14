#!/usr/bin/env python3
"""
Generate Figure 3: Hierarchy Strength vs Absorption Line Plot.

Line plot showing non-monotonic relationship between hierarchy strength (cosine similarity)
and absorption rate. Shows seed-level traces and error bars.
R² = 0.04 indicates no monotonic relationship.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Project root
PROJECT_ROOT = "/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "current/writing/figures")

COLORS = {
    'absorption': '#3B82F6',    # Blue
    'seed_42': '#2563EB',
    'seed_123': '#7C3AED',
    'seed_456': '#059669',
    'error_bar': '#374151',
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


def generate_figure3():
    """Generate H_Comp hierarchy strength line plot."""
    data_path = os.path.join(PROJECT_ROOT, "current/exp/results/full/h_comp_6levels_3seeds.json")
    data = load_json(data_path)

    cosine_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    absorption_by_level = data['absorption_by_level']
    absorption_by_seed = data['absorption_by_seed']

    # Aggregate across seeds
    means = []
    stds = []
    for cos_level in cosine_levels:
        key = f"cos_{cos_level}"
        means.append(absorption_by_level[key]['mean'])
        stds.append(absorption_by_level[key]['std'])

    # Seed-level data for traces
    seeds = ['42', '123', '456']
    seed_colors = [COLORS['seed_42'], COLORS['seed_123'], COLORS['seed_456']]
    seed_traces = []
    for seed in seeds:
        trace = [absorption_by_seed[seed][f"cos_{cos_level}"]['mean'] for cos_level in cosine_levels]
        seed_traces.append(trace)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot seed-level traces (faint)
    for i, (trace, color) in enumerate(zip(seed_traces, seed_colors)):
        ax.plot(cosine_levels, trace, 'o--', markersize=6, color=color, alpha=0.35, lw=1,
                label=f'Seed {seeds[i]}')

    # Plot mean with error bars
    ax.errorbar(cosine_levels, means, yerr=stds,
                fmt='s-', markersize=10, capsize=5, capthick=2, lw=2.5,
                color=COLORS['absorption'], ecolor='gray',
                markerfacecolor=COLORS['absorption'], markeredgecolor='white', markeredgewidth=1.5,
                label='Mean ± Std')

    ax.set_xlabel('Hierarchy Strength (Parent-Child Cosine Similarity)', fontsize=12)
    ax.set_ylabel('Feature Absorption Rate', fontsize=12)
    ax.set_title('Figure 3: Hierarchy Strength vs Absorption\n(Non-Monotonic, R² = 0.04)',
                  fontsize=13, fontweight='bold')
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0, 1.5)

    # Add statistical annotation
    ax.text(0.65, 1.35, 'R² = 0.04 (p = 0.703)\nNo monotonic relationship',
            fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.8))

    # Add legend
    ax.legend(loc='upper right', framealpha=0.9)

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
    print("Generating Figure 3: H_Comp Hierarchy Strength...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        fig = generate_figure3()
        save_figure(fig, "figure3_hierarchy_strength")
        print("  ✓ figure3_hierarchy_strength saved successfully")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()