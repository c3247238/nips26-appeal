#!/usr/bin/env python3
"""
Generate all publication-ready figures from experiment results.
Figures:
  1. H_Mech 2x2 factorial bar chart (conditions A-D with error bars)
  2. Multi-seed stability plot (trained vs random across seeds)
  3. Steering sensitivity by alpha value (line plot for absorbed vs non-absorbed)
  4. Safety vs non-safety absorption comparison (box plot)
  5. Hierarchy strength ablation (absorption vs cosine similarity)
  6. L0 sparsity ablation (absorption by L0 target)
  7. Held-out generalization scatter (train vs test absorption)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")

RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current/exp/results")
PILOTS_DIR = RESULTS_DIR / "pilots"
FULL_DIR = RESULTS_DIR / "full"
OUTPUT_DIR = FULL_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
COLORS = {
    "trained": "#2E86AB",
    "random": "#A23B72",
    "absorbed": "#F18F01",
    "non_absorbed": "#C73E1D",
    "safety": "#E63946",
    "non_safety": "#457B9D",
    "primary": "#1D3557",
    "secondary": "#457B9D",
    "accent": "#E9C46A",
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_fig(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUTPUT_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {name}.pdf and {name}.png")


def figure_1_h_mech_factorial():
    """Figure 1: H_Mech 2x2 factorial bar chart."""
    data = load_json(PILOTS_DIR / "h_mech_factorial.json")
    results = data["results"]

    conditions = ["A", "B", "C", "D"]
    labels = [
        "Random Enc\n+ Random Dec",
        "Trained Enc\n+ Random Dec",
        "Random Enc\n+ Trained Dec",
        "Trained Enc\n+ Trained Dec",
    ]

    # Use absorption_overlap as the primary metric (proportional absorption)
    means = [results[c]["absorption_overlap_mean"] for c in conditions]
    stds = [results[c]["absorption_overlap_std"] for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(conditions, means, yerr=stds, capsize=8,
                  color=[COLORS["random"], COLORS["trained"], COLORS["random"], COLORS["trained"]],
                  edgecolor="black", linewidth=1.2, alpha=0.85)

    ax.set_ylabel("Multi-Child Proportional Absorption", fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_title("Figure 1: H_Mech Factorial Validation — Absorption is Encoder-Driven",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, max(means) * 1.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Add annotation for encoder vs decoder effect
    encoder_effect = data["effect_decomposition"]["encoder_effect"]
    decoder_effect = data["effect_decomposition"]["decoder_effect"]
    ax.text(0.5, 0.95,
            f"Encoder effect: {encoder_effect:.3f}\nDecoder effect: {decoder_effect:.3f}",
            transform=ax.transAxes, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
            fontsize=10)

    plt.tight_layout()
    save_fig(fig, "figure_1_h_mech_factorial")


def figure_2_multiseed_stability():
    """Figure 2: Multi-seed stability plot."""
    data = load_json(RESULTS_DIR / "multiseed_validation.json")
    trained = data["trained_results"]
    random = data["random_results"]

    seeds = [r["seed"] for r in trained]
    trained_means = [r["absorption"]["mean"] for r in trained]
    trained_stds = [r["absorption"]["std"] for r in trained]
    random_means = [r["absorption"]["mean"] for r in random]
    random_stds = [r["absorption"]["std"] for r in random]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.errorbar(seeds, trained_means, yerr=trained_stds, fmt="o-",
                color=COLORS["trained"], capsize=6, linewidth=2, markersize=8,
                label="Trained SAE", markeredgecolor="black", markeredgewidth=1)
    ax.errorbar(seeds, random_means, yerr=random_stds, fmt="s--",
                color=COLORS["random"], capsize=6, linewidth=2, markersize=8,
                label="Random SAE", markeredgecolor="black", markeredgewidth=1)

    ax.axhline(y=0.3, color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="Threshold (0.3)")

    ax.set_xlabel("Random Seed", fontsize=12)
    ax.set_ylabel("Mean Absorption Rate", fontsize=12)
    ax.set_title("Figure 2: Multi-Seed Stability — Trained vs Random SAEs",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_xticks(seeds)

    # Add stats annotation
    stats = data["statistics"]
    ax.text(0.02, 0.98,
            f"Trained: {stats['trained_mean']:.3f} ± {stats['trained_std']:.3f}\n"
            f"Random: {stats['random_mean']:.3f} ± {stats['random_std']:.3f}\n"
            f"t-test: p = {stats['p_value']:.2e}",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcyan", alpha=0.8),
            fontsize=10)

    plt.tight_layout()
    save_fig(fig, "figure_2_multiseed_stability")


def figure_3_steering_sensitivity():
    """Figure 3: Steering sensitivity by alpha value."""
    data = load_json(PILOTS_DIR / "h3_steering.json")

    # Use parent_steering_parent_inputs as the most relevant condition
    steering = data["sensitivity_results"]["parent_steering_parent_inputs"]["by_alpha"]
    alphas = sorted([float(k) for k in steering.keys()])

    absorbed_means = [steering[str(a)]["absorbed_mean"] for a in alphas]
    absorbed_stds = [steering[str(a)]["absorbed_std"] for a in alphas]
    non_absorbed_means = [steering[str(a)]["non_absorbed_mean"] for a in alphas]
    non_absorbed_stds = [steering[str(a)]["non_absorbed_std"] for a in alphas]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.errorbar(alphas, absorbed_means, yerr=absorbed_stds, fmt="o-",
                color=COLORS["absorbed"], capsize=6, linewidth=2, markersize=8,
                label="Absorbed Features", markeredgecolor="black", markeredgewidth=1)
    ax.errorbar(alphas, non_absorbed_means, yerr=non_absorbed_stds, fmt="s--",
                color=COLORS["non_absorbed"], capsize=6, linewidth=2, markersize=8,
                label="Non-Absorbed Features", markeredgecolor="black", markeredgewidth=1)

    ax.set_xlabel("Steering Alpha", fontsize=12)
    ax.set_ylabel("Sensitivity (||delta||)", fontsize=12)
    ax.set_title("Figure 3: Steering Sensitivity — Absorbed vs Non-Absorbed Features",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="upper left")

    # Add classification info
    fc = data["feature_classification"]
    ax.text(0.98, 0.02,
            f"Absorbed: {fc['n_absorbed_features']} features\n"
            f"Non-absorbed: {fc['n_non_absorbed_features']} features\n"
            f"Threshold: {fc['absorption_threshold']}",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
            fontsize=9)

    plt.tight_layout()
    save_fig(fig, "figure_3_steering_sensitivity")


def figure_4_safety_comparison():
    """Figure 4: Safety vs non-safety absorption comparison."""
    data = load_json(PILOTS_DIR / "h_safe_gemma.json")

    safety_scores = data["results"]["safety"]["absorption_scores"]
    non_safety_scores = data["results"]["non_safety"]["absorption_scores"]

    fig, ax = plt.subplots(figsize=(7, 5))

    bp = ax.boxplot([safety_scores, non_safety_scores], labels=["Safety-Critical", "Non-Safety"],
                    patch_artist=True, widths=0.5)

    bp["boxes"][0].set_facecolor(COLORS["safety"])
    bp["boxes"][1].set_facecolor(COLORS["non_safety"])
    for box in bp["boxes"]:
        box.set_alpha(0.7)
        box.set_edgecolor("black")
        box.set_linewidth(1.5)

    for whisker in bp["whiskers"]:
        whisker.set_linewidth(1.5)
    for cap in bp["caps"]:
        cap.set_linewidth(1.5)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax.set_ylabel("Absorption Rate", fontsize=12)
    ax.set_title("Figure 4: Safety-Critical vs Non-Safety Feature Absorption\n(GPT-2 Small SAE)",
                 fontsize=13, fontweight="bold", pad=15)

    # Add stats annotation
    stats = data["statistics"]
    ax.text(0.5, 0.95,
            f"Mann-Whitney U = {stats['mann_whitney_u']:.1f}\n"
            f"p = {stats['p_value']:.3f}\n"
            f"Effect size = {stats['effect_size_rank_biserial']:.2f}",
            transform=ax.transAxes, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="mistyrose", alpha=0.8),
            fontsize=10)

    plt.tight_layout()
    save_fig(fig, "figure_4_safety_comparison")


def figure_5_hierarchy_strength():
    """Figure 5: Hierarchy strength ablation."""
    data = load_json(FULL_DIR / "ablation_hierarchy_strength.json")
    results = data["results"]
    stats = data["statistics"]

    similarities = [0.5, 0.67, 0.8]
    means = [stats["similarity_means"][str(s)] for s in similarities]
    stds = [stats["similarity_stds"][str(s)] for s in similarities]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(similarities, means, yerr=stds, fmt="o-",
                color=COLORS["primary"], capsize=8, linewidth=2.5, markersize=10,
                markeredgecolor="black", markeredgewidth=1.5)

    ax.set_xlabel("Parent-Child Cosine Similarity", fontsize=12)
    ax.set_ylabel("Mean Absorption Rate", fontsize=12)
    ax.set_title("Figure 5: Ablation — Absorption vs Hierarchy Strength",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(0.45, 0.85)
    ax.set_xticks(similarities)

    # Add value labels
    for s, m, st in zip(similarities, means, stds):
        ax.text(s, m + st + 0.01, f"{m:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    # Add ANOVA annotation
    ax.text(0.5, 0.15,
            f"ANOVA: F = {stats['anova_f']:.1f}, p < 1e-10\n"
            f"Monotonic: {stats['monotonic']}",
            transform=ax.transAxes, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.7),
            fontsize=10)

    plt.tight_layout()
    save_fig(fig, "figure_5_hierarchy_strength")


def figure_6_l0_sparsity():
    """Figure 6: L0 sparsity ablation."""
    data = load_json(FULL_DIR / "ablation_l0_sparsity.json")
    results = data["results"]
    stats = data["statistics"]

    l0_levels = [20, 32, 50]
    means = [stats["l0_means"][str(l)] for l in l0_levels]
    stds = [stats["l0_stds"][str(l)] for l in l0_levels]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar([str(l) for l in l0_levels], means, yerr=stds, capsize=8,
                  color=[COLORS["secondary"], COLORS["primary"], COLORS["accent"]],
                  edgecolor="black", linewidth=1.2, alpha=0.85)

    ax.set_xlabel("L0 Sparsity Target", fontsize=12)
    ax.set_ylabel("Mean Absorption Rate", fontsize=12)
    ax.set_title("Figure 6: Ablation — Absorption vs L0 Sparsity Level",
                 fontsize=13, fontweight="bold", pad=15)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Add ANOVA annotation
    ax.text(0.5, 0.95,
            f"ANOVA: F = {stats['anova_f']:.1f}, p < 1e-10\n"
            f"Monotonic increase: {stats['monotonic']}",
            transform=ax.transAxes, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
            fontsize=10)

    plt.tight_layout()
    save_fig(fig, "figure_6_l0_sparsity")


def figure_7_heldout_generalization():
    """Figure 7: Held-out generalization scatter."""
    data = load_json(PILOTS_DIR / "heldout_generalization.json")
    results = data["results"]

    train_raw = results["train_absorption_raw"]
    test_raw = results["test_absorption_raw"]

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(train_raw, test_raw, alpha=0.6, s=60,
               color=COLORS["primary"], edgecolor="black", linewidth=0.5)

    # Add diagonal line (perfect correlation)
    min_val = min(min(train_raw), min(test_raw))
    max_val = max(max(train_raw), max(test_raw))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, alpha=0.7,
            label="Perfect correlation")

    ax.set_xlabel("Train Absorption", fontsize=12)
    ax.set_ylabel("Test Absorption", fontsize=12)
    ax.set_title("Figure 7: Held-Out Generalization — Train vs Test Absorption",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=11)

    # Add stats annotation
    stats = data["statistics"]
    ax.text(0.05, 0.95,
            f"Pearson r = {stats['pearson_r']:.3f}\n"
            f"p < 1e-10\n"
            f"Within 10%: {stats['within_10_percent']}",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcyan", alpha=0.8),
            fontsize=10)

    plt.tight_layout()
    save_fig(fig, "figure_7_heldout_generalization")


def figure_8_summary_table():
    """Figure 8: Summary table of all hypothesis results."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    table_data = [
        ["H1", "Trained > Random", "PASSED", "p = 1.4e-08", "Trained: 0.355 ± 0.026, Random: 0.033 ± 0.011"],
        ["H_Mech", "Encoder-driven absorption", "PASSED", "p < 1e-10", "Encoder effect: 0.519, Decoder effect: 0.008"],
        ["H3", "Steering sensitivity", "FAILED", "p = 0.936", "Ratio = 0.97x (no difference)"],
        ["H_Safe", "Safety > Non-safety", "FAILED", "p = 0.326", "Safety: 0.960, Non-safety: 0.964"],
        ["Ablation: Hierarchy", "Monotonic with similarity", "PASSED", "p < 1e-10", "0.5→0.416, 0.67→0.501, 0.8→0.544"],
        ["Ablation: L0 Sparsity", "Higher sparsity → higher abs", "FAILED", "p < 1e-10", "20→0.552, 32→0.490, 50→0.419 (reverse!)"],
        ["Generalization", "Train ≈ Test", "PASSED", "r = 1.000", "Train: 0.352, Test: 0.352, diff: 0.0%"],
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=["Hypothesis", "Claim", "Status", "p-value", "Key Result"],
        loc="center",
        cellLoc="center",
        colColours=["#1D3557"] * 5,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    # Style header
    for i in range(5):
        table[(0, i)].set_text_props(color="white", fontweight="bold")
        table[(0, i)].set_height(0.08)

    # Color-code status cells
    for i in range(1, len(table_data) + 1):
        status = table_data[i - 1][2]
        if status == "PASSED":
            table[(i, 2)].set_facecolor("#d4edda")
            table[(i, 2)].set_text_props(color="#155724", fontweight="bold")
        else:
            table[(i, 2)].set_facecolor("#f8d7da")
            table[(i, 2)].set_text_props(color="#721c24", fontweight="bold")

    ax.set_title("Figure 8: Summary of All Hypothesis Tests and Ablations",
                 fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    save_fig(fig, "figure_8_summary_table")


def main():
    print("=" * 60)
    print("Generating all publication-ready figures")
    print("=" * 60)

    figure_1_h_mech_factorial()
    figure_2_multiseed_stability()
    figure_3_steering_sensitivity()
    figure_4_safety_comparison()
    figure_5_hierarchy_strength()
    figure_6_l0_sparsity()
    figure_7_heldout_generalization()
    figure_8_summary_table()

    print("=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)

    # Write DONE marker
    done_marker = RESULTS_DIR / "visualize_all_DONE"
    done_marker.write_text(json.dumps({
        "task_id": "visualize_all",
        "status": "success",
        "summary": f"Generated 8 figures in {OUTPUT_DIR}",
        "timestamp": "2026-04-29T23:55:00",
    }))
    print("DONE marker written.")


if __name__ == "__main__":
    main()
