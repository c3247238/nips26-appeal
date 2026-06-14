#!/usr/bin/env python3
"""
Generate Figure 4: H_Pareto Sensitivity-Absorption Scatter Plot (Degenerate).

Shows the attempted sensitivity-absorption Pareto frontier measurement.
Degenerate result: absorption = 0 across all L0 levels; no frontier detected.
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
    'sensitivity': '#EF4444',   # Red
    'degenerate': '#6B7280',    # Gray for degenerate point
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


def generate_figure4():
    """Generate H_Pareto scatter plot (degenerate case)."""
    data_path = os.path.join(PROJECT_ROOT, "current/exp/results/full/h_pareto_full.json")
    data = load_json(data_path)

    l0_levels = [16, 32, 64, 128]
    summary_by_l0 = data['summary_by_l0']

    # All L0 levels have absorption_mean = 0.0, sensitivity_mean ~ 0.1054
    absorption_means = [summary_by_l0[str(l)]['absorption_mean'] for l in l0_levels]
    absorption_stds = [summary_by_l0[str(l)]['absorption_std'] for l in l0_levels]
    sensitivity_means = [summary_by_l0[str(l)]['sensitivity_mean'] for l in l0_levels]
    sensitivity_stds = [summary_by_l0[str(l)]['sensitivity_std'] for l in l0_levels]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot absorption vs L0 (primary axis)
    ax.scatter(l0_levels, absorption_means, s=150, c=[COLORS['absorption']],
               marker='s', alpha=0.8, edgecolors='white', linewidths=2, zorder=5,
               label='Absorption rate')
    ax.errorbar(l0_levels, absorption_means, yerr=absorption_stds,
                fmt='none', capsize=5, color=COLORS['absorption'], alpha=0.6)

    # Add horizontal line at y=0 to highlight degenerate case
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, lw=1)
    ax.text(130, 0.02, 'Degenerate: absorption = 0\nacross all L0 levels', fontsize=9,
            ha='right', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.8))

    # Since both metrics are degenerate (all same), show sensitivity as inset
    inset_text = (f"Frontier fit: degenerate\n"
                  f"a = 1.0, b = -0.5\n"
                  f"Sensitivity: {sensitivity_means[0]:.4f} ± {sensitivity_stds[0]:.4f}\n"
                  f"(stable across all L0 levels)")
    ax.text(0.05, 0.95, inset_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcyan', edgecolor='gray', alpha=0.9))

    ax.set_xlabel('L0 Sparsity Target', fontsize=12)
    ax.set_ylabel('Feature Absorption Rate', fontsize=12)
    ax.set_title('Figure 4: Sensitivity-Absorption Pareto Frontier\n(Degenerate Result)',
                  fontsize=13, fontweight='bold')
    ax.set_xlim(5, 145)
    ax.set_ylim(-0.05, 0.25)

    # Add annotation for the finding
    ax.annotate('No frontier detected;\ntheoretical prediction\nnot supported',
                xy=(64, 0), xytext=(90, 0.12),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.8))

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
    print("Generating Figure 4: H_Pareto Pareto Frontier (Degenerate)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        fig = generate_figure4()
        save_figure(fig, "figure4_pareto_frontier")
        print("  ✓ figure4_pareto_frontier saved successfully")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()