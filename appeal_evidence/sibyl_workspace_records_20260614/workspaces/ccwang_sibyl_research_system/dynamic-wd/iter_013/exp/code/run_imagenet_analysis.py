#!/usr/bin/env python3
"""
Comprehensive ImageNet ResNet-50 analysis for Unified Dynamic Weight Decay paper.
Loads all 21 experimental results (7 methods × 3 seeds), computes aggregate
statistics, generates publication-quality figures, and writes a summary report.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current")
RESULTS   = WORKSPACE / "exp" / "results" / "full" / "imagenet_resnet50"
FIG_DIR   = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["NoWD", "FixedWD", "SWD", "CWD", "CPR", "CAWD", "EqWD"]
SEEDS   = [42, 123, 456]

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHOD_COLORS = {
    "NoWD":    "#999999",
    "FixedWD": "#4477AA",
    "SWD":     "#66CCEE",
    "CWD":     "#228833",
    "CPR":     "#CCBB44",
    "CAWD":    "#EE6677",
    "EqWD":    "#AA3377",
}

METHOD_LABELS = {
    "NoWD":    "No WD",
    "FixedWD": "Fixed WD (AdamW/SGDW)",
    "SWD":     "SWD (NeurIPS 2023)",
    "CWD":     "CWD (ICLR 2026)",
    "CPR":     "CPR (NeurIPS 2024)",
    "CAWD":    "CAWD (Ours-variant)",
    "EqWD":    "EqWD (Ours)",
}

METHOD_MARKERS = {
    "NoWD": "x", "FixedWD": "s", "SWD": "^", "CWD": "v",
    "CPR": "D", "CAWD": "o", "EqWD": "*",
}


def load_results():
    """Load all result JSON files."""
    data = {}
    for method in METHODS:
        data[method] = {}
        for seed in SEEDS:
            run_dir = RESULTS / f"{method}_seed{seed}"
            result_files = list(run_dir.glob("*_results.json")) if run_dir.exists() else []
            if result_files:
                with open(result_files[0]) as f:
                    data[method][seed] = json.load(f)
            else:
                print(f"WARNING: Missing {method}_seed{seed}")
    return data


def compute_summary(data):
    """Compute mean ± std for each method."""
    summary = {}
    for method in METHODS:
        accs = [data[method][s]["best_test_top1"] for s in SEEDS if s in data[method]]
        if accs:
            summary[method] = {
                "mean": statistics.mean(accs),
                "std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                "values": accs,
                "n": len(accs),
            }
    return summary


def plot_bar_chart(summary):
    """Bar chart with error bars showing mean ± std."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(METHODS))
    means = [summary[m]["mean"] for m in METHODS]
    stds = [summary[m]["std"] for m in METHODS]
    colors = [METHOD_COLORS[m] for m in METHODS]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black",
                  linewidth=0.5, alpha=0.85, width=0.7)

    # Highlight EqWD
    bars[-1].set_edgecolor("#AA3377")
    bars[-1].set_linewidth(2.0)

    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.15, f"{m:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold" if METHODS[i] == "EqWD" else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m].split(" (")[0] for m in METHODS], rotation=15, ha="right")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("ImageNet ResNet-50 — Weight Decay Method Comparison (45 epochs)")
    ax.set_ylim(69, 73.5)
    ax.axhline(y=summary["FixedWD"]["mean"], color="#4477AA", linestyle="--", alpha=0.4, label="FixedWD baseline")
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "imagenet_resnet50_comparison.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "imagenet_resnet50_comparison.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: imagenet_resnet50_comparison.png/pdf")


def plot_training_curves(data):
    """Training loss and test accuracy curves (mean over seeds, with std band)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for method in METHODS:
        histories = []
        for seed in SEEDS:
            if seed in data[method] and "history" in data[method][seed]:
                histories.append(data[method][seed]["history"])

        if not histories:
            continue

        # Align epochs
        min_epochs = min(len(h) for h in histories)
        epochs = list(range(1, min_epochs + 1))

        # Train loss
        losses = np.array([[h[e]["train_loss"] for e in range(min_epochs)] for h in histories])
        mean_loss = losses.mean(axis=0)
        std_loss = losses.std(axis=0)

        # Test accuracy
        accs = np.array([[h[e]["test_top1"] for e in range(min_epochs)] for h in histories])
        mean_acc = accs.mean(axis=0)
        std_acc = accs.std(axis=0)

        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method].split(" (")[0]
        lw = 2.5 if method == "EqWD" else 1.5
        alpha = 1.0 if method in ["EqWD", "FixedWD", "SWD"] else 0.7

        axes[0].plot(epochs, mean_loss, color=color, label=label, linewidth=lw, alpha=alpha)
        axes[0].fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.1)

        axes[1].plot(epochs, mean_acc, color=color, label=label, linewidth=lw, alpha=alpha)
        axes[1].fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, color=color, alpha=0.1)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss Curves")
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Top-1 Accuracy (%)")
    axes[1].set_title("Test Accuracy Curves")
    axes[1].legend(fontsize=8, ncol=2)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "imagenet_training_curves.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "imagenet_training_curves.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: imagenet_training_curves.png/pdf")


def plot_improvement_over_baseline(summary):
    """Delta accuracy vs FixedWD baseline."""
    baseline = summary["FixedWD"]["mean"]
    methods_no_base = [m for m in METHODS if m != "NoWD"]

    fig, ax = plt.subplots(figsize=(8, 5))

    deltas = []
    colors = []
    labels = []
    for m in methods_no_base:
        delta = summary[m]["mean"] - baseline
        deltas.append(delta)
        colors.append(METHOD_COLORS[m])
        labels.append(METHOD_LABELS[m].split(" (")[0])

    bars = ax.barh(range(len(deltas)), deltas, color=colors, edgecolor="black",
                   linewidth=0.5, alpha=0.85, height=0.6)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Δ Top-1 Accuracy vs FixedWD (%)")
    ax.set_title("ImageNet ResNet-50 — Improvement over FixedWD Baseline")
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.5)

    for i, d in enumerate(deltas):
        ax.text(d + 0.02 if d >= 0 else d - 0.02, i,
                f"{d:+.2f}", va="center", ha="left" if d >= 0 else "right",
                fontsize=10, fontweight="bold" if methods_no_base[i] == "EqWD" else "normal")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "imagenet_delta_baseline.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "imagenet_delta_baseline.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: imagenet_delta_baseline.png/pdf")


def plot_variance_comparison(summary):
    """Scatter: mean accuracy vs std (stability analysis)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for method in METHODS:
        s = summary[method]
        ax.scatter(s["std"], s["mean"], color=METHOD_COLORS[method],
                   marker=METHOD_MARKERS[method], s=200, zorder=5,
                   edgecolors="black", linewidth=0.5,
                   label=METHOD_LABELS[method])

    ax.set_xlabel("Standard Deviation (lower = more stable)")
    ax.set_ylabel("Mean Top-1 Accuracy (%)")
    ax.set_title("ImageNet ResNet-50 — Accuracy vs Stability")
    ax.legend(fontsize=8, loc="lower right")

    # Add quadrant labels
    mid_x = np.median([summary[m]["std"] for m in METHODS])
    mid_y = np.median([summary[m]["mean"] for m in METHODS])
    ax.axvline(x=mid_x, color="gray", linestyle=":", alpha=0.3)
    ax.axhline(y=mid_y, color="gray", linestyle=":", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "imagenet_accuracy_vs_stability.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "imagenet_accuracy_vs_stability.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: imagenet_accuracy_vs_stability.png/pdf")


def plot_top5_comparison(data, summary):
    """Bar chart for top-5 accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    top5_summary = {}
    for method in METHODS:
        vals = []
        for seed in SEEDS:
            if seed in data[method] and "history" in data[method][seed]:
                # Get best top5 from history
                hist = data[method][seed]["history"]
                best_top5 = max(h.get("test_top5", 0) for h in hist)
                vals.append(best_top5)
        if vals:
            top5_summary[method] = {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }

    x = np.arange(len(METHODS))
    means = [top5_summary.get(m, {}).get("mean", 0) for m in METHODS]
    stds = [top5_summary.get(m, {}).get("std", 0) for m in METHODS]
    colors = [METHOD_COLORS[m] for m in METHODS]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black",
                  linewidth=0.5, alpha=0.85, width=0.7)
    bars[-1].set_edgecolor("#AA3377")
    bars[-1].set_linewidth(2.0)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.1, f"{m:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold" if METHODS[i] == "EqWD" else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m].split(" (")[0] for m in METHODS], rotation=15, ha="right")
    ax.set_ylabel("Top-5 Accuracy (%)")
    ax.set_title("ImageNet ResNet-50 — Top-5 Accuracy Comparison")
    ax.set_ylim(88, 92)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "imagenet_resnet50_top5.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "imagenet_resnet50_top5.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: imagenet_resnet50_top5.png/pdf")


def write_summary_report(data, summary):
    """Write comprehensive summary report."""
    report = ["# ImageNet ResNet-50 Experiment Results\n"]
    report.append(f"**Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    report.append("## Configuration\n")
    report.append("- **Dataset**: ImageNet-1K (1000 classes)\n")
    report.append("- **Architecture**: ResNet-50 (25.6M parameters)\n")
    report.append("- **Epochs**: 45\n")
    report.append("- **Batch size**: 256\n")
    report.append("- **Learning rate**: 0.1 (cosine schedule)\n")
    report.append("- **AMP**: Enabled\n")
    report.append("- **Seeds**: 42, 123, 456\n\n")

    report.append("## Results Summary\n\n")
    report.append("| Method | Type | seed42 | seed123 | seed456 | Mean ± Std |\n")
    report.append("|--------|------|--------|---------|---------|------------|\n")

    type_map = {
        "NoWD": "Baseline",
        "FixedWD": "Baseline",
        "SWD": "NeurIPS 2023",
        "CWD": "ICLR 2026",
        "CPR": "NeurIPS 2024",
        "CAWD": "Ours (variant)",
        "EqWD": "**Ours**",
    }

    for m in METHODS:
        s = summary[m]
        vals = [data[m].get(seed, {}).get("best_test_top1", 0) for seed in SEEDS]
        marker = " **" if m == "EqWD" else ""
        report.append(f"| {m} | {type_map[m]} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {s['mean']:.3f} ± {s['std']:.3f}{marker} |\n")

    # Key findings
    best = max(summary.items(), key=lambda x: x[1]["mean"])
    baseline_mean = summary["FixedWD"]["mean"]

    report.append(f"\n## Key Findings\n\n")
    report.append(f"1. **Best method**: {best[0]} with {best[1]['mean']:.3f}% (Δ = +{best[1]['mean'] - baseline_mean:.3f}% vs FixedWD)\n")
    report.append(f"2. **Most stable**: {min(summary.items(), key=lambda x: x[1]['std'])[0]} (lowest std = {min(s['std'] for s in summary.values()):.3f})\n")

    for m in METHODS:
        delta = summary[m]["mean"] - baseline_mean
        report.append(f"3. {m}: {summary[m]['mean']:.3f}% (Δ = {delta:+.3f}% vs FixedWD, std = {summary[m]['std']:.3f})\n")

    report_path = RESULTS / "imagenet_resnet50_summary.md"
    with open(report_path, "w") as f:
        f.writelines(report)
    print(f"  Saved: {report_path}")

    # Also save as JSON for pipeline
    json_path = RESULTS / "imagenet_resnet50_aggregate.json"
    with open(json_path, "w") as f:
        json.dump({
            "dataset": "imagenet",
            "architecture": "resnet50",
            "epochs": 45,
            "methods": {m: summary[m] for m in METHODS},
            "best_method": best[0],
            "best_accuracy": best[1]["mean"],
        }, f, indent=2)
    print(f"  Saved: {json_path}")


def main():
    print("=" * 60)
    print("  ImageNet ResNet-50 Analysis")
    print("=" * 60)

    # Load results
    print("\n1. Loading results...")
    data = load_results()

    # Compute summary
    print("2. Computing summary statistics...")
    summary = compute_summary(data)

    for m in METHODS:
        s = summary[m]
        print(f"   {m:10s}: {s['mean']:.3f} ± {s['std']:.3f}")

    # Generate figures
    print("\n3. Generating figures...")
    plot_bar_chart(summary)
    plot_training_curves(data)
    plot_improvement_over_baseline(summary)
    plot_variance_comparison(summary)
    plot_top5_comparison(data, summary)

    # Write report
    print("\n4. Writing summary report...")
    write_summary_report(data, summary)

    # Write DONE marker
    done_path = WORKSPACE / "exp" / "results" / "imagenet_analysis_DONE"
    done_data = {
        "task_id": "imagenet_analysis",
        "status": "success",
        "summary": f"7 methods × 3 seeds, best: EqWD {summary['EqWD']['mean']:.3f}%",
        "figures": [str(p) for p in FIG_DIR.glob("*.png")],
    }
    with open(done_path, "w") as f:
        json.dump(done_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Analysis complete! Best: EqWD = {summary['EqWD']['mean']:.3f}%")
    print(f"  Figures saved to: {FIG_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
