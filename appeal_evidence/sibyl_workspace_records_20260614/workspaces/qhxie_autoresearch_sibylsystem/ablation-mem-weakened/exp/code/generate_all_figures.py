#!/usr/bin/env python3
"""
Generate all publication-ready figures for the paper.
Includes H1-H6 (completed) + H9-H10 (new pilot results).
All figures use matplotlib with publication-quality styling.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Publication styling
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/exp/results/full")
PILOT_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/exp/results/pilots")
FIGURES_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/exp/results/figures")
FIGURES_DIR.mkdir(exist_ok=True)

def load_all_data():
    with open(RESULTS_DIR / "correlation_report_full.json") as f:
        corr = json.load(f)
    with open(RESULTS_DIR / "correlation_with_baseline.json") as f:
        corr_baseline = json.load(f)
    with open(RESULTS_DIR / "full_steering_probing_gpt2_l4_results.json") as f:
        steering_l4 = json.load(f)
    with open(RESULTS_DIR / "full_steering_probing_gpt2_l8_results.json") as f:
        steering_l8 = json.load(f)
    with open(RESULTS_DIR / "ec50_analysis.json") as f:
        ec50 = json.load(f)
    with open(RESULTS_DIR / "precision_recall_analysis.json") as f:
        pr = json.load(f)
    with open(RESULTS_DIR / "h6_inhibition_graph.json") as f:
        h6 = json.load(f)
    with open(PILOT_DIR / "h9_cooccurrence_analysis.json") as f:
        h9 = json.load(f)
    with open(PILOT_DIR / "h10_random_sae_baseline.json") as f:
        h10 = json.load(f)
    return corr, corr_baseline, steering_l4, steering_l8, ec50, pr, h6, h9, h10

# --- Figure 1: Absorption Rate Distribution Across Layers ---
def fig1_absorption_distribution(corr):
    """Figure 1: Bar chart of absorption rates across layers."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    layers = [("Layer 0", "layer_0"), ("Layer 4", "layer_4"),
              ("Layer 8", "layer_8"), ("Layer 10", "layer_10")]

    for idx, (layer_name, layer_key) in enumerate(layers):
        ax = axes[idx // 2, idx % 2]
        if layer_key not in corr["layer_results"]:
            ax.text(0.5, 0.5, f'{layer_name}\nData not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(layer_name)
            continue

        absorption = corr["layer_results"][layer_key]["absorption_rates"]
        letters = sorted(absorption.keys())
        values = [absorption[l] for l in letters]

        colors = ['#D32F2F' if v > 0.1 else '#1976D2' if v > 0.05 else '#388E3C' for v in values]
        bars = ax.bar(range(len(letters)), values, color=colors, edgecolor='black', linewidth=0.3)
        ax.set_xticks(range(len(letters)))
        ax.set_xticklabels(letters, fontsize=8)
        ax.set_ylabel('Absorption Rate')
        ax.set_title(layer_name)
        ax.set_ylim(0, max(values) * 1.2 if values else 0.3)

        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(len(letters) - 1, mean_val + 0.01, f'mean={mean_val:.3f}',
               ha='right', va='bottom', fontsize=7, color='gray')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D32F2F', label='HIGH (>10%)'),
        Patch(facecolor='#1976D2', label='MEDIUM (5-10%)'),
        Patch(facecolor='#388E3C', label='LOW (<5%)'),
    ]
    axes[0, 0].legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.suptitle('Figure 1: Absorption Rate Distribution Across Layers (26 First-Letter Features)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig1_absorption_distribution.png')
    plt.close()
    print("  Figure 1: Absorption distribution saved")

# --- Figure 2: Steering Success vs Absorption Rate ---
def fig2_steering_scatter(corr, corr_baseline, steering_l4, steering_l8):
    """Figure 2: Scatter plot of absorption vs steering success (L4 and L8)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for idx, (layer_name, steering_data, ax) in enumerate([
        ("Layer 4", steering_l4, axes[0]),
        ("Layer 8", steering_l8, axes[1]),
    ]):
        layer_key = f"layer_{layer_name.split()[-1]}"
        absorption_rates = corr["layer_results"][layer_key]["absorption_rates"]
        steering_success = corr["layer_results"][layer_key]["steering_success_at_50"]

        letters = sorted(absorption_rates.keys())
        x = [absorption_rates[l] for l in letters]
        y = [steering_success[l] for l in letters]

        colors = ['#D32F2F' if a > 0.1 else '#1976D2' if a > 0.05 else '#388E3C' for a in x]
        ax.scatter(x, y, c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=3)

        # Regression line
        if len(x) > 2:
            slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
            x_line = np.linspace(min(x), max(max(x), 0.01), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'k--', alpha=0.5, linewidth=1, zorder=2)

        # Labels for high-absorption points
        for l, xi, yi in zip(letters, x, y):
            if xi > 0.05:
                ax.annotate(l, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=7)

        h1_data = corr_baseline["H1_feature_specific"][layer_key]
        ax.set_title(f'{layer_name}\nr={h1_data["pearson_r"]:.3f}, p={h1_data["pearson_p"]:.3f}, R²={h1_data["R2"]:.3f}')
        ax.set_xlabel('Absorption Rate')
        ax.set_ylabel('Steering Success Rate (strength=50)')
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D32F2F', label='HIGH (>10%)'),
        Patch(facecolor='#1976D2', label='MEDIUM (5-10%)'),
        Patch(facecolor='#388E3C', label='LOW (<5%)'),
    ]
    axes[0].legend(handles=legend_elements, loc='lower left', fontsize=8)

    plt.suptitle('Figure 2: H1 - Absorption Rate vs Raw Steering Effectiveness',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig2_steering_scatter.png')
    plt.close()
    print("  Figure 2: Steering scatter saved")

# --- Figure 3: Precision-Recall Decomposition ---
def fig3_precision_recall(pr, corr):
    """Figure 3: Precision and recall vs absorption rate."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    layers = [("Layer 4", "layer_4", pr["layer_4"]), ("Layer 8", "layer_8", pr["layer_8"])]

    for layer_idx, (layer_name, layer_key, layer_data) in enumerate(layers):
        absorption_rates = corr["layer_results"][layer_key]["absorption_rates"]
        layer_num = layer_name.split()[-1]
        with open(RESULTS_DIR / f"full_steering_probing_gpt2_l{layer_num}_results.json") as f:
            probing = json.load(f)

        letters = sorted(absorption_rates.keys())
        x = [absorption_rates[l] for l in letters]

        for k_idx, k in enumerate([5, 20]):
            ax = axes[layer_idx, k_idx]

            precisions = []
            recalls = []
            for l in letters:
                for kr in probing["probing_results"][l]["k_results"]:
                    if kr["k"] == k:
                        precisions.append(kr["precision"])
                        recalls.append(kr["recall"])
                        break

            colors = ['#D32F2F' if a > 0.1 else '#1976D2' if a > 0.05 else '#388E3C' for a in x]

            ax.scatter(x, precisions, c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                      marker='o', label='Precision', zorder=3)
            ax.scatter(x, recalls, c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                      marker='s', label='Recall', zorder=3)

            if len(x) > 2:
                slope_p, intercept_p, _, _, _ = stats.linregress(x, precisions)
                slope_r, intercept_r, _, _, _ = stats.linregress(x, recalls)
                x_line = np.linspace(min(x), max(max(x), 0.01), 100)
                ax.plot(x_line, slope_p * x_line + intercept_p, 'k--', alpha=0.3, linewidth=1)
                ax.plot(x_line, slope_r * x_line + intercept_r, 'k:', alpha=0.3, linewidth=1)

            ax.set_title(f'{layer_name}, k={k}')
            ax.set_xlabel('Absorption Rate')
            ax.set_ylabel('Score')
            ax.set_ylim(-0.05, 1.1)
            ax.legend(loc='lower left', fontsize=8)
            ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.2)

    plt.suptitle('Figure 3: H5 - Precision-Recall vs Absorption Rate',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig3_precision_recall.png')
    plt.close()
    print("  Figure 3: Precision-recall decomposition saved")

# --- Figure 4: EC50 vs Absorption Rate ---
def fig4_ec50_scatter(ec50, corr):
    """Figure 4: EC50 vs absorption rate scatter plot."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for idx, (layer_name, ax) in enumerate([("Layer 4", axes[0]), ("Layer 8", axes[1])]):
        layer_num = layer_name.split()[-1]
        layer_key = f"layer_{layer_num}"
        ec50_data = ec50[f"layer_{layer_num}"]
        absorption_rates = corr["layer_results"][layer_key]["absorption_rates"]

        letters = []
        x_vals = []
        y_vals = []
        for letter, feat_data in ec50_data["feature_results"].items():
            if feat_data.get("ec50") is not None:
                letters.append(letter)
                x_vals.append(absorption_rates.get(letter, 0))
                y_vals.append(feat_data["ec50"])

        colors = ['#D32F2F' if a > 0.1 else '#1976D2' if a > 0.05 else '#388E3C' for a in x_vals]
        ax.scatter(x_vals, y_vals, c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=3)

        for l, xi, yi in zip(letters, x_vals, y_vals):
            if xi > 0.05:
                ax.annotate(l, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=7)

        if len(x_vals) > 2:
            slope, intercept, r_val, p_val, _ = stats.linregress(x_vals, y_vals)
            x_line = np.linspace(min(x_vals), max(max(x_vals), 0.01), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'k--', alpha=0.5, linewidth=1, zorder=2)

        corr_data = ec50_data["correlation"]
        ax.set_title(f'{layer_name}\nr={corr_data["pearson_r"]:.3f}, p={corr_data["pearson_p"]:.3f}')
        ax.set_xlabel('Absorption Rate')
        ax.set_ylabel('EC50 (Steering Strength)')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D32F2F', label='HIGH (>10%)'),
        Patch(facecolor='#1976D2', label='MEDIUM (5-10%)'),
        Patch(facecolor='#388E3C', label='LOW (<5%)'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.suptitle('Figure 4: H4 - EC50 vs Absorption Rate',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig4_ec50_scatter.png')
    plt.close()
    print("  Figure 4: EC50 scatter saved")

# --- Figure 5: Inhibition Graph Precision@K ---
def fig5_inhibition_graph(h6):
    """Figure 5: Inhibition graph precision@k (flat at 0)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    k_values = list(range(1, 21))
    precision_values = [0.0] * 20
    chance_level = h6["summary"]["chance_precision"]

    ax.plot(k_values, precision_values, 'o-', color='#D32F2F', linewidth=2, markersize=6,
            label='Observed Precision@k', zorder=3)
    ax.axhline(y=chance_level, color='gray', linestyle='--', linewidth=1.5,
              label=f'Chance Level ({chance_level:.3f})', zorder=2)
    ax.axhline(y=0.1, color='green', linestyle=':', linewidth=1.5,
              label='Predicted Minimum (0.10)', zorder=2)

    ax.fill_between(k_values, 0, chance_level, alpha=0.1, color='gray')

    ax.set_xlabel('k (Top-k Neighbors)')
    ax.set_ylabel('Precision@k')
    ax.set_title(f'H6: Inhibition Graph Precision@k\n'
                f'0/{h6["summary"]["total_predictions"]} predictions correct | '
                f'Fisher p = {h6["summary"]["fisher_p_value"]:.3f}',
                fontweight='bold')
    ax.set_ylim(-0.02, 0.25)
    ax.legend(loc='upper right', fontsize=9)

    # Add annotation
    ax.annotate('HYPOTHESIS\nFALSIFIED', xy=(15, 0.15), fontsize=14, fontweight='bold',
               color='#D32F2F', ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#D32F2F'))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig5_inhibition_graph.png')
    plt.close()
    print("  Figure 5: Inhibition graph precision saved")

# --- Figure 6: Cross-Layer Comparison ---
def fig6_cross_layer(corr_baseline):
    """Figure 6: Cross-layer comparison of degradation coefficients."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    layers = [4, 8]

    h1_slopes = [corr_baseline["H1_feature_specific"][f"layer_{l}"]["slope"] for l in layers]
    h1_r2 = [corr_baseline["H1_feature_specific"][f"layer_{l}"]["R2"] for l in layers]

    h1b_slopes = [corr_baseline["H1b_steering_delta"][f"layer_{l}"]["slope"] for l in layers]
    h1b_r2 = [corr_baseline["H1b_steering_delta"][f"layer_{l}"]["R2"] for l in layers]

    h2_slopes = [corr_baseline["H2_probing"][f"layer_{l}"]["slope"] for l in layers]
    h2_r2 = [corr_baseline["H2_probing"][f"layer_{l}"]["R2"] for l in layers]

    x = np.arange(len(layers))
    width = 0.25

    bars1 = ax.bar(x - width, h1_slopes, width, label='H1: Raw Steering', color='#1976D2', alpha=0.8)
    bars2 = ax.bar(x, h1b_slopes, width, label='H1b: Delta-Corrected', color='#D32F2F', alpha=0.8)
    bars3 = ax.bar(x + width, h2_slopes, width, label='H2: Probing F1', color='#388E3C', alpha=0.8)

    for bars, r2_vals in [(bars1, h1_r2), (bars2, h1b_r2), (bars3, h2_r2)]:
        for bar, r2 in zip(bars, r2_vals):
            height = bar.get_height()
            ax.annotate(f'R²={r2:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Slope (Degradation Coefficient)')
    ax.set_title('H3: Cross-Layer Comparison of Degradation Coefficients', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.legend(loc='upper right', fontsize=9)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig6_cross_layer.png')
    plt.close()
    print("  Figure 6: Cross-layer comparison saved")

# --- Figure 7: H9 - Co-occurrence vs Absorption Rate ---
def fig7_cooccurrence(h9):
    """Figure 7: H9 - Co-occurrence strength (p_11) vs absorption rate."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    features = [r["feature"] for r in h9["feature_results"]]
    p11 = [r["p_11"] for r in h9["feature_results"]]
    absorption = [r["absorption_rate"] for r in h9["feature_results"]]

    # Color by whether the relationship is tautological (p11 + absorption = 1)
    tautological = [abs(p + a - 1.0) < 0.01 for p, a in zip(p11, absorption)]
    colors = ['#D32F2F' if t else '#1976D2' for t in tautological]

    ax.scatter(p11, absorption, c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=3)

    # Regression line
    if len(p11) > 2:
        slope, intercept, r_val, p_val, _ = stats.linregress(p11, absorption)
        x_line = np.linspace(min(p11), max(p11), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', alpha=0.5, linewidth=1, zorder=2)

    # Label points
    for f, px, ay in zip(features, p11, absorption):
        if px > 0.3 or ay > 0.5:
            ax.annotate(f, (px, ay), textcoords="offset points", xytext=(5, 5), fontsize=7)

    # Tautological line p11 + absorption = 1
    x_tauto = np.linspace(0, 1, 100)
    y_tauto = 1 - x_tauto
    ax.plot(x_tauto, y_tauto, 'r:', alpha=0.4, linewidth=1.5, label='p₁₁ + A = 1 (tautological)', zorder=1)

    corr_data = h9["correlation"]
    ax.set_title(f'H9: Co-occurrence (p₁₁) vs Absorption Rate (Layer 8)\n'
                f'r={corr_data["pearson_r"]:.3f}, p={corr_data["pearson_p"]:.2e}\n'
                f'Note: Perfect negative correlation is tautological by construction',
                fontweight='bold', fontsize=10)
    ax.set_xlabel('p₁₁ (Parent-Child Co-occurrence Probability)')
    ax.set_ylabel('Absorption Rate')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=8)

    # Warning annotation
    ax.annotate('TAUTOLOGICAL\nOPERATIONALIZATION', xy=(0.7, 0.15), fontsize=10, fontweight='bold',
               color='#D32F2F', ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#D32F2F'))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig7_cooccurrence.png')
    plt.close()
    print("  Figure 7: Co-occurrence analysis saved")

# --- Figure 8: H10 - Trained vs Random SAE Absorption ---
def fig10_random_sae(h10):
    """Figure 8: H10 - Trained vs Random SAE absorption comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    trained = h10["trained_sae"]["absorption_by_feature"]
    random = h10["random_sae"]["absorption_by_feature"]
    letters = sorted(trained.keys())

    # Left: Side-by-side bar chart
    ax = axes[0]
    x = np.arange(len(letters))
    width = 0.35

    trained_vals = [trained[l]["absorption_rate"] for l in letters]
    random_vals = [random[l]["absorption_rate"] for l in letters]

    bars1 = ax.bar(x - width/2, trained_vals, width, label='Trained SAE', color='#1976D2', alpha=0.8, edgecolor='black', linewidth=0.3)
    bars2 = ax.bar(x + width/2, random_vals, width, label='Random SAE', color='#D32F2F', alpha=0.8, edgecolor='black', linewidth=0.3)

    ax.set_xlabel('First-Letter Feature')
    ax.set_ylabel('Absorption Rate')
    ax.set_title('Absorption Rate: Trained vs Random SAE (Layer 8)')
    ax.set_xticks(x)
    ax.set_xticklabels(letters, fontsize=8)
    ax.legend(loc='upper right', fontsize=9)

    # Mean lines
    ax.axhline(y=h10["trained_sae"]["mean"], color='#1976D2', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=h10["random_sae"]["mean"], color='#D32F2F', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(len(letters)-0.5, h10["trained_sae"]["mean"] + 0.02,
           f'mean={h10["trained_sae"]["mean"]:.3f}', ha='right', fontsize=7, color='#1976D2')
    ax.text(len(letters)-0.5, h10["random_sae"]["mean"] + 0.02,
           f'mean={h10["random_sae"]["mean"]:.3f}', ha='right', fontsize=7, color='#D32F2F')

    # Right: Box plot comparison
    ax = axes[1]
    data_to_plot = [trained_vals, random_vals]
    bp = ax.boxplot(data_to_plot, labels=['Trained SAE', 'Random SAE'], patch_artist=True)
    colors_box = ['#1976D2', '#D32F2F']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Individual points
    for i, data in enumerate(data_to_plot):
        jitter = np.random.normal(i + 1, 0.04, len(data))
        ax.scatter(jitter, data, alpha=0.5, s=30, color='black', zorder=3)

    comp = h10["comparison"]
    ax.set_title(f'Distribution Comparison\n'
                f'Paired t={comp["paired_t_stat"]:.2f}, p={comp["paired_t_p"]:.2e}\n'
                f'Random mean = {h10["random_sae"]["mean"]:.3f} vs Trained mean = {h10["trained_sae"]["mean"]:.3f}',
                fontsize=10)
    ax.set_ylabel('Absorption Rate')

    plt.suptitle('Figure 8: H10 - Random SAE Baseline (Chanin Metric Not Specific to Learned Structure)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig8_random_sae.png')
    plt.close()
    print("  Figure 8: Random SAE comparison saved")

# --- Figure 9: Summary Table Visualization ---
def fig9_summary_table(corr_baseline, ec50, pr, h6, h9, h10):
    """Figure 9: Summary table of all hypotheses as a visual table."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Table data
    headers = ['Hypothesis', 'Statement', 'Layer 4 Result', 'Layer 8 Result', 'Verdict']
    rows = []

    # H1
    h1_l4 = corr_baseline["H1_feature_specific"]["layer_4"]
    h1_l8 = corr_baseline["H1_feature_specific"]["layer_8"]
    rows.append(['H1', 'Higher absorption -> lower raw steering',
                f'r={h1_l4["pearson_r"]:.3f}, p={h1_l4["pearson_p"]:.3f}',
                f'r={h1_l8["pearson_r"]:.3f}, p={h1_l8["pearson_p"]:.3f}',
                'FALSIFIED'])

    # H1b
    h1b_l4 = corr_baseline["H1b_steering_delta"]["layer_4"]
    h1b_l8 = corr_baseline["H1b_steering_delta"]["layer_8"]
    l8_str = f'r={h1b_l8["pearson_r"]:.3f}, p={h1b_l8["pearson_p"]:.3f}'
    rows.append(['H1b', 'Higher absorption -> lower delta-corrected steering',
                f'r={h1b_l4["pearson_r"]:.3f}, p={h1b_l4["pearson_p"]:.3f}',
                l8_str,
                'NOT SIGNIFICANT\n(after correction)'])

    # H2
    h2_l4 = corr_baseline["H2_probing"]["layer_4"]
    h2_l8 = corr_baseline["H2_probing"]["layer_8"]
    rows.append(['H2', 'Higher absorption -> lower probing F1',
                f'r={h2_l4["pearson_r"]:.3f}, p={h2_l4["pearson_p"]:.3f}',
                f'r={h2_l8["pearson_r"]:.3f}, p={h2_l8["pearson_p"]:.3f}',
                'FALSIFIED'])

    # H3
    h3 = corr_baseline["H3_consistency"]
    rows.append(['H3', 'Degradation consistent across layers',
                f'CV={h3["cv_h1"]:.2f} (H1)',
                f'CV={h3["cv_h2"]:.2f} (H2)',
                'FALSIFIED'])

    # H4
    ec50_l4 = ec50["layer_4"]["correlation"]
    ec50_l8 = ec50["layer_8"]["correlation"]
    rows.append(['H4', 'Absorption affects steering efficiency (EC50)',
                f'r={ec50_l4["pearson_r"]:.3f}, p={ec50_l4["pearson_p"]:.3f}',
                f'r={ec50_l8["pearson_r"]:.3f}, p={ec50_l8["pearson_p"]:.3f}',
                'FALSIFIED'])

    # H5
    rows.append(['H5', 'Absorption affects recall, not precision',
                'Precision invariant,\nrecall varies',
                'Precision invariant,\nrecall varies',
                'SUPPORTED'])

    # H6
    rows.append(['H6', 'Decoder correlation predicts absorption',
                f'Precision@20 = {h6["summary"]["overall_precision"]:.1f}',
                f'Fisher p = {h6["summary"]["fisher_p_value"]:.3f}',
                'FALSIFIED'])

    # H9
    rows.append(['H9', 'Co-occurrence correlates with absorption',
                'N/A',
                f'r={h9["correlation"]["pearson_r"]:.3f}\n(tautological)',
                'INVALID\n(tautological)'])

    # H10
    rows.append(['H10', 'Random SAE shows structural absorption',
                'N/A',
                f'Random={h10["random_sae"]["mean"]:.3f}\nTrained={h10["trained_sae"]["mean"]:.3f}',
                'SUPPORTED\n(methodological finding)'])

    # Create table
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Style header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#1976D2')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Color rows by verdict
    for i, row in enumerate(rows):
        verdict = row[-1]
        if 'SUPPORTED' in verdict and 'NOT' not in verdict:
            color = '#E8F5E9'
        elif 'FALSIFIED' in verdict:
            color = '#FFEBEE'
        elif 'INVALID' in verdict:
            color = '#FFF8E1'
        else:
            color = '#FFFFFF'

        for j in range(len(headers)):
            table[(i + 1, j)].set_facecolor(color)

    ax.set_title('Table 1: Summary of All Hypotheses and Results',
                fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig9_summary_table.png')
    plt.close()
    print("  Figure 9: Summary table saved")

# --- Figure 10: Rate-Distortion Schematic ---
def fig10_rate_distortion():
    """Figure 10: Conceptual rate-distortion schematic."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Generate rate-distortion curve
    rates = np.linspace(0.01, 1, 100)
    # Distortion decreases with rate, with a knee point
    distortion = 1.0 / (1 + np.exp(5 * (rates - 0.4))) * 0.8 + 0.1

    ax.plot(rates, distortion, 'b-', linewidth=2.5, label='Rate-Distortion Curve', zorder=3)

    # Mark operating points
    # High sparsity (low rate) - high distortion
    ax.plot(0.15, 0.75, 'ro', markersize=12, zorder=4)
    ax.annotate('High Sparsity\n(Low Rate, High Distortion)',
               xy=(0.15, 0.75), xytext=(0.35, 0.85),
               arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
               fontsize=9, color='red', fontweight='bold')

    # Absorption as optimal point
    ax.plot(0.4, 0.35, 'go', markersize=12, zorder=4)
    ax.annotate('Absorption\n(Optimal Compression)',
               xy=(0.4, 0.35), xytext=(0.55, 0.5),
               arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
               fontsize=9, color='green', fontweight='bold')

    # Low sparsity (high rate) - low distortion
    ax.plot(0.8, 0.15, 'mo', markersize=12, zorder=4)
    ax.annotate('Low Sparsity\n(High Rate, Low Distortion)',
               xy=(0.8, 0.15), xytext=(0.6, 0.05),
               arrowprops=dict(arrowstyle='->', color='purple', lw=1.5),
               fontsize=9, color='purple', fontweight='bold')

    ax.set_xlabel('Rate (Sparsity Level / Information Preserved)')
    ax.set_ylabel('Distortion (Reconstruction Error)')
    ax.set_title('Figure 10: Rate-Distortion Interpretation of Absorption\n'
                'Absorption as Optimal Compression Under Hierarchical Co-occurrence',
                fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add shaded region for "optimal" tradeoff
    ax.fill_between([0.3, 0.5], 0, 1, alpha=0.1, color='green', label='Optimal Region')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig10_rate_distortion.png')
    plt.close()
    print("  Figure 10: Rate-distortion schematic saved")

# --- Figure 11: Dose-Response Curves ---
def fig11_dose_response(steering_l4, steering_l8, corr):
    """Figure 11: Dose-response curves by absorption level."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for idx, (layer_name, steering_data, ax) in enumerate([
        ("Layer 4", steering_l4, axes[0]),
        ("Layer 8", steering_l8, axes[1]),
    ]):
        layer_key = f"layer_{layer_name.split()[-1]}"
        absorption_rates = corr["layer_results"][layer_key]["absorption_rates"]

        high_abs = []
        med_abs = []
        low_abs = []

        for letter, rate in absorption_rates.items():
            if letter not in steering_data["steering_results"]:
                continue
            sr = steering_data["steering_results"][letter]["strength_results"]
            strengths = [r["strength"] for r in sr]
            success = [r["success_rate"] for r in sr]
            if rate > 0.1:
                high_abs.append((strengths, success, letter))
            elif rate > 0.05:
                med_abs.append((strengths, success, letter))
            else:
                low_abs.append((strengths, success, letter))

        def plot_group(data, color, label, linestyle='-'):
            if not data:
                return
            all_success = [s for _, s, _ in data]
            strengths = data[0][0]
            mean_success = np.mean(all_success, axis=0)
            std_success = np.std(all_success, axis=0)
            ax.plot(strengths, mean_success, color=color, linestyle=linestyle, linewidth=2, label=label, zorder=3)
            ax.fill_between(strengths,
                           np.maximum(0, mean_success - std_success),
                           np.minimum(1, mean_success + std_success),
                           alpha=0.2, color=color)

        plot_group(high_abs, '#D32F2F', f'HIGH (n={len(high_abs)})')
        plot_group(med_abs, '#1976D2', f'MEDIUM (n={len(med_abs)})')
        plot_group(low_abs, '#388E3C', f'LOW (n={len(low_abs)})')

        ax.set_title(layer_name)
        ax.set_xlabel('Steering Strength')
        ax.set_ylabel('Steering Success Rate')
        ax.set_xscale('log')
        ax.set_ylim(-0.05, 1.1)
        ax.legend(loc='lower right', fontsize=8)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    plt.suptitle('Figure 11: H4 - Dose-Response Curves by Absorption Level',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig11_dose_response.png')
    plt.close()
    print("  Figure 11: Dose-response curves saved")

def generate_markdown_summary(corr_baseline, ec50, pr, h6, h9, h10):
    """Generate a markdown summary of all figures."""
    lines = [
        "# Figure Generation Summary",
        "",
        "All figures generated for the paper:",
        "**Feature Absorption as Optimal Compression: Evidence that SAEs Correctly Handle Hierarchical Features**",
        "",
        "## Figure List",
        "",
        "| Figure | Description | Key Finding |",
        "|--------|-------------|-------------|",
        "| Fig 1 | Absorption rate distribution across layers (L0, L4, L8, L10) | Mean 2.1-3.9%, max 24.2% (Feature U at L8) |",
        "| Fig 2 | Steering success vs absorption rate (L4, L8) | No significant correlation (H1 falsified) |",
        "| Fig 3 | Precision-recall decomposition (k=5, 20) | Precision invariant, recall varies (H5 supported) |",
        "| Fig 4 | EC50 vs absorption rate | No significant correlation (H4 falsified) |",
        "| Fig 5 | Inhibition graph precision@k | 0/520 correct, precision@20 = 0.0 (H6 falsified) |",
        "| Fig 6 | Cross-layer degradation coefficient comparison | Opposite signs, CV > 1.0 (H3 falsified) |",
        "| Fig 7 | H9: Co-occurrence (p_11) vs absorption | Tautological by construction (r = -1.0) |",
        "| Fig 8 | H10: Trained vs Random SAE absorption | Random > Trained (0.278 vs 0.034), methodological finding |",
        "| Fig 9 | Summary table of all hypotheses | 1 supported (H5), 5 falsified, 1 invalid, 1 methodological |",
        "| Fig 10 | Rate-distortion schematic | Conceptual framework for absorption as optimal compression |",
        "| Fig 11 | Dose-response curves by absorption level | No systematic difference across absorption levels |",
        "",
        "## Statistical Summary",
        "",
        "### Multiple Comparison Correction (12 tests)",
        f"- Bonferroni alpha = {corr_baseline.get('multiple_comparisons_correction', {}).get('bonferroni_alpha', 0.00417):.4f}",
        "- **0 tests survive Bonferroni correction**",
        "- **0 tests survive BH-FDR correction**",
        "",
        "### Key Results",
        f"- H1 (Raw Steering, L8): r = {corr_baseline['H1_feature_specific']['layer_8']['pearson_r']:.3f}, p = {corr_baseline['H1_feature_specific']['layer_8']['pearson_p']:.3f}",
        f"- H1b (Delta-Corrected, L8): r = {corr_baseline['H1b_steering_delta']['layer_8']['pearson_r']:.3f}, p = {corr_baseline['H1b_steering_delta']['layer_8']['pearson_p']:.3f} (uncorrected)",
        f"- H2 (Probing, L8): r = {corr_baseline['H2_probing']['layer_8']['pearson_r']:.3f}, p = {corr_baseline['H2_probing']['layer_8']['pearson_p']:.3f}",
        f"- H4 (EC50, L8): r = {ec50['layer_8']['correlation']['pearson_r']:.3f}, p = {ec50['layer_8']['correlation']['pearson_p']:.3f}",
        f"- H5 (Precision-Recall): Precision = 1.0 at k>=5 universally; recall varies 0.05-1.0",
        f"- H6 (Inhibition Graph): Precision@20 = {h6['summary']['overall_precision']:.1f}, Fisher p = {h6['summary']['fisher_p_value']:.3f}",
        f"- H9 (Co-occurrence): r = {h9['correlation']['pearson_r']:.1f} (tautological)",
        f"- H10 (Random SAE): Random mean = {h10['random_sae']['mean']:.3f}, Trained mean = {h10['trained_sae']['mean']:.3f}, paired t p = {h10['comparison']['paired_t_p']:.2e}",
        "",
        "## Paper Framing",
        "",
        "The paper's central contribution is **honest null-result reporting** with rigorous controls:",
        "1. **H1-H4**: Absorption does not significantly degrade downstream tasks in GPT-2 Small SAEs",
        "2. **H5**: The one robust finding - absorption affects recall (coverage) but not precision (selectivity)",
        "3. **H6**: A falsified mechanistic hypothesis (decoder correlations do not predict absorption)",
        "4. **H10**: A methodological finding (Chanin metric is not specific to learned structure)",
        "5. **Rate-distortion framing**: Provides theoretical grounding for why absorption may be optimal",
        "",
    ]
    return "\n".join(lines)

def main():
    print("=" * 70)
    print("GENERATING ALL PUBLICATION-READY FIGURES")
    print("=" * 70)

    corr, corr_baseline, steering_l4, steering_l8, ec50, pr, h6, h9, h10 = load_all_data()

    print(f"\nFigures will be saved to: {FIGURES_DIR}")
    print()

    fig1_absorption_distribution(corr)
    fig2_steering_scatter(corr, corr_baseline, steering_l4, steering_l8)
    fig3_precision_recall(pr, corr)
    fig4_ec50_scatter(ec50, corr)
    fig5_inhibition_graph(h6)
    fig6_cross_layer(corr_baseline)
    fig7_cooccurrence(h9)
    fig10_random_sae(h10)
    fig9_summary_table(corr_baseline, ec50, pr, h6, h9, h10)
    fig10_rate_distortion()
    fig11_dose_response(steering_l4, steering_l8, corr)

    # Generate markdown summary
    md = generate_markdown_summary(corr_baseline, ec50, pr, h6, h9, h10)
    md_path = FIGURES_DIR / "figure_summary.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\n  Figure summary saved to {md_path}")

    # List all generated files
    print("\n" + "=" * 70)
    print("ALL FIGURES GENERATED:")
    print("=" * 70)
    for f in sorted(FIGURES_DIR.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:45s} ({size_kb:.1f} KB)")
    print(f"\n  {md_path.name}")

    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
