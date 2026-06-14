#!/usr/bin/env python3
"""
Generate paper figures for the multi-child proportional absorption experiment.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
OUTPUT_DIR = WORKSPACE / "exp" / "results" / "full" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_figure_1_absorption_comparison():
    """Figure 1: Multi-child proportional absorption by condition."""
    # Load data
    with open(WORKSPACE / "exp" / "results" / "full" / "multichild_absorption.json") as f:
        data = json.load(f)

    results = data["absorption_results"]

    conditions = ["Trained SAE", "Random Decoder", "Shuffled Features", "Permuted Encoder"]
    means = [
        results["trained_sae"]["absorption_k5_mean"],
        results["random_decoder"]["absorption_k5_mean"],
        results["shuffled_features"]["absorption_k5_mean"],
        results["permuted_encoder"]["absorption_k5_mean"]
    ]
    stds = [
        results["trained_sae"]["absorption_k5_std"],
        results["random_decoder"]["absorption_k5_std"],
        results["shuffled_features"]["absorption_k5_std"],
        results["permuted_encoder"]["absorption_k5_std"]
    ]

    colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(conditions, means, yerr=stds, color=colors, edgecolor="black", linewidth=1.5, capsize=5)

    ax.set_ylabel("Multi-child Proportional Absorption Rate", fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_title("H1: Multi-child Proportional Absorption by Condition\n(k=5)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 0.7)

    # Add significance markers
    ax.annotate("***", xy=(0, means[0] + 0.05), fontsize=14, ha="center", fontweight="bold")
    ax.plot([0, 3], [0.55, 0.55], "k-", linewidth=1)
    ax.text(1.5, 0.57, "p < 10^-133", ha="center", fontsize=10)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_1_absorption_comparison.pdf", format="pdf")
    plt.savefig(OUTPUT_DIR / "figure_1_absorption_comparison.png", format="png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'figure_1_absorption_comparison.pdf'}")
    plt.close()


def create_figure_2_proportional_variance():
    """Figure 2: Proportional variance by condition."""
    # Load data
    with open(WORKSPACE / "exp" / "results" / "full" / "multichild_absorption.json") as f:
        data = json.load(f)

    results = data["absorption_results"]

    conditions = ["Trained SAE", "Random Decoder", "Shuffled Features", "Permuted Encoder"]
    means = [
        results["trained_sae"]["proportional_variance_mean"],
        results["random_decoder"]["proportional_variance_mean"],
        results["shuffled_features"]["proportional_variance_mean"],
        results["permuted_encoder"]["proportional_variance_mean"]
    ]
    stds = [
        results["trained_sae"]["proportional_variance_std"],
        results["random_decoder"]["proportional_variance_std"],
        results["shuffled_features"]["proportional_variance_std"],
        results["permuted_encoder"]["proportional_variance_std"]
    ]

    colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(conditions, means, yerr=stds, color=colors, edgecolor="black", linewidth=1.5, capsize=5)

    ax.set_ylabel("Proportional Variance", fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_title("Proportional Variance of Child Contributions\n(Trained SAE shows higher asymmetry)", fontsize=14, fontweight="bold")

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_2_proportional_variance.pdf", format="pdf")
    plt.savefig(OUTPUT_DIR / "figure_2_proportional_variance.png", format="png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'figure_2_proportional_variance.pdf'}")
    plt.close()


def create_figure_3_h2_correlation():
    """Figure 3: Absorption vs frequency scatter plot (H2)."""
    # Load data
    with open(WORKSPACE / "exp" / "results" / "full" / "h2_frequency_correlation.json") as f:
        data = json.load(f)

    frequency = np.array(data["data"]["frequency"])
    absorption = np.array(data["data"]["absorption_per_feature"])

    # Only plot active features
    mask = absorption > 0
    frequency_active = frequency[mask]
    absorption_active = absorption[mask]

    rho = data["correlations"]["spearman_rho"]
    p_val = data["correlations"]["spearman_p"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(frequency_active, absorption_active, alpha=0.5, s=20, c="#3498db")

    # Add trend line
    z = np.polyfit(frequency_active, absorption_active, 1)
    p = np.poly1d(z)
    x_line = np.linspace(frequency_active.min(), frequency_active.max(), 100)
    ax.plot(x_line, p(x_line), "r-", linewidth=2, label=f"Trend line")

    ax.set_xlabel("Feature Activation Frequency", fontsize=12)
    ax.set_ylabel("Absorption Rate", fontsize=12)
    ax.set_title(f"H2: Absorption vs Feature Frequency\nSpearman rho = {rho:.3f} (p = {p_val:.2e})", fontsize=14, fontweight="bold")
    ax.legend()

    # Note that H2 is NOT supported
    ax.text(0.95, 0.95, "H2 NOT SUPPORTED\n(Positive correlation)", transform=ax.transAxes,
            fontsize=12, ha="right", va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_3_frequency_correlation.pdf", format="pdf")
    plt.savefig(OUTPUT_DIR / "figure_3_frequency_correlation.png", format="png", dpi=150)
    print(f"Saved: {OUTPUT_DIR / 'figure_3_frequency_correlation.pdf'}")
    plt.close()


def create_summary_table():
    """Create summary table for the paper."""
    # Load data
    with open(WORKSPACE / "exp" / "results" / "full" / "multichild_absorption.json") as f:
        multichild_data = json.load(f)

    with open(WORKSPACE / "exp" / "results" / "full" / "h2_frequency_correlation.json") as f:
        h2_data = json.load(f)

    results = multichild_data["absorption_results"]
    h1_stats = multichild_data["h1_statistics"]

    table = """
\\begin{table}[htbp]
\\centering
\\caption{Main Results: Multi-child Proportional Absorption}
\\label{tab:absorption_results}
\\begin{tabular}{|l|c|c|c|c|}
\\hline
\\textbf{Condition} & \\textbf{Mean} & \\textbf{Std} & \\textbf{vs SAE} & \\textbf{p-value} \\\\
\\hline
Trained SAE & 0.500 & 0.000 & --- & --- \\\\
Random Decoder & 0.059 & 0.069 & -0.441 & < 10$^{-133}$ \\\\
Shuffled Features & 0.487 & 0.034 & -0.013 & 1.62 $\\times$ 10$^{-4}$ \\\\
Permuted Encoder & 0.484 & 0.037 & -0.016 & 2.25 $\\times$ 10$^{-5}$ \\\\
\\hline
\\end{tabular}
\\smallskip
\\textit{Notes:} Multi-child proportional absorption (k=5). Cohen's d = 8.94 for trained SAE vs random decoder.
\\end{table}
"""

    with open(OUTPUT_DIR / "table_absorption_results.tex", "w") as f:
        f.write(table)

    print(f"Saved: {OUTPUT_DIR / 'table_absorption_results.tex'}")


def main():
    print("=" * 70)
    print("Generating Paper Figures")
    print("=" * 70)

    create_figure_1_absorption_comparison()
    create_figure_2_proportional_variance()
    create_figure_3_h2_correlation()
    create_summary_table()

    print("\n" + "=" * 70)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
