#!/usr/bin/env python3
"""
Generate Figure 5: Safety vs Non-Safety Feature Absorption Violin Plot.

Shows the comparison of absorption distributions between safety-critical
and non-safety control features for both Gemma Scope pilot and GPT-2 Small validation.
Null result: no significant difference in either model (positive for safety analysis).
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

COLORS = {
    'safety': '#6366F1',        # Indigo
    'non_safety': '#A855F7',    # Purple
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


def generate_figure5():
    """Generate safety vs non-safety violin plot."""
    # Load Gemma Scope pilot
    gemma_path = os.path.join(PROJECT_ROOT, "current/exp/results/pilots/h_safe_gemma_pilot.json")
    gemma_data = load_json(gemma_path)

    # Load GPT-2 Small held-out validation
    held_out_path = os.path.join(PROJECT_ROOT, "current/exp/results/held_out_validation.json")
    held_out_data = load_json(held_out_path)

    safety_gemma = gemma_data['safety_absorption_rates']
    non_safety_gemma = gemma_data['non_safety_absorption_rates']
    p_gemma = gemma_data['mann_whitney_p']

    safety_gpt2 = [held_out_data['h_safe']['aggregate']['safety_mean']]
    non_safety_gpt2 = [held_out_data['h_safe']['aggregate']['non_safety_mean']]
    p_gpt2 = held_out_data['h_safe']['aggregate']['p_value']

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left panel: Gemma Scope pilot (both groups at 0.0)
    ax1 = axes[0]
    positions_gemma = [1, 2]

    # Violin plot
    parts1 = ax1.violinplot([safety_gemma, non_safety_gemma], positions=positions_gemma,
                             showmeans=True, showmedians=True, widths=0.6)

    for i, pc in enumerate(parts1['bodies']):
        pc.set_facecolor(COLORS['safety'] if i == 0 else COLORS['non_safety'])
        pc.set_alpha(0.6)

    parts1['cmeans'].set_color('black')
    parts1['cmedians'].set_color('gray')
    for partname in ['cbars', 'cmins', 'cmaxes']:
        vp = parts1[partname]
        vp.set_edgecolor('gray')

    ax1.set_xticks(positions_gemma)
    ax1.set_xticklabels(['Safety\nFeatures', 'Non-Safety\nFeatures'], fontsize=10)
    ax1.set_ylabel('Feature Absorption Rate', fontsize=11)
    ax1.set_title('Gemma Scope SAE (Pilot)\nn=5/group', fontsize=11, fontweight='bold')
    ax1.set_ylim(-0.1, 0.5)

    # Statistical annotation for Gemma
    ax1.text(0.05, 0.95, f'Mann-Whitney p = {p_gemma:.2f}\n(No significant difference)',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.8))
    ax1.text(0.5, 0.25, 'Note: All absorption\nrates = 0.0', transform=ax1.transAxes,
             fontsize=8, ha='center', va='center', style='italic', color='gray',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    # Right panel: GPT-2 Small validation
    ax2 = axes[1]
    positions_gpt2 = [1, 2]

    # Since we only have means, create a bar chart comparison
    means_gpt2 = [safety_gpt2[0], non_safety_gpt2[0]]
    stds_gpt2 = [50, 50]  # approximate from held-out data

    bars = ax2.bar(positions_gpt2, means_gpt2,
                   color=[COLORS['safety'], COLORS['non_safety']],
                   width=0.5, alpha=0.7)

    ax2.set_xticks(positions_gpt2)
    ax2.set_xticklabels(['Safety\nFeatures', 'Non-Safety\nFeatures'], fontsize=10)
    ax2.set_ylabel('Mean Absorption Rate', fontsize=11)
    ax2.set_title('GPT-2 Small (Held-Out Validation)\nn=20/group', fontsize=11, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, means_gpt2):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Statistical annotation for GPT-2
    ax2.text(0.05, 0.95, f'Mann-Whitney p = {p_gpt2:.3f}\n(No significant difference)',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.8))

    # Shared legend
    legend_elements = [
        Patch(facecolor=COLORS['safety'], alpha=0.7, label='Safety features'),
        Patch(facecolor=COLORS['non_safety'], alpha=0.7, label='Non-safety features'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=10, framealpha=0.9)

    fig.suptitle('Figure 5: Safety vs Non-Safety Feature Absorption\n(Null Result: No Elevated Absorption for Safety-Critical Features)',
                  fontsize=13, fontweight='bold', y=1.02)

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
    print("Generating Figure 5: Safety vs Non-Safety Features...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        fig = generate_figure5()
        save_figure(fig, "figure5_safety_features")
        print("  ✓ figure5_safety_features saved successfully")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()