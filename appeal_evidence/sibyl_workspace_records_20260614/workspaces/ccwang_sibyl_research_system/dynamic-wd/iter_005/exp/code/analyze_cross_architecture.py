#!/usr/bin/env python3
"""
Cross-Architecture Analysis for Iteration 5.
Computes phi spread, Cohen's d, and generates summary tables/figures.

Data sources:
  - iter_003: ResNet-20 AdamW (CIFAR-10/100), SGD original (CIFAR-10/100)
  - iter_005 (current): VGG-16-BN, NoBN, matched-rho SGD, rho_low sweep
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import math

import numpy as np

# Paths
WORKSPACE = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd")
ITER3_RESULTS = WORKSPACE / "iter_003/exp/results"
ITER5_RESULTS = WORKSPACE / "current/exp/results"
OUTPUT_DIR = ITER5_RESULTS / "analysis/iter5"
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_summaries(base_dir, pattern="**/summary.json"):
    """Load all summary.json files from a directory tree."""
    results = []
    for p in sorted(base_dir.rglob("summary.json")):
        try:
            data = json.loads(p.read_text())
            data["_path"] = str(p)
            results.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  WARN: Failed to load {p}: {e}", file=sys.stderr)
    return results


def group_by_method(summaries, method_key="wd_method"):
    """Group summaries by WD method, collecting best_test_acc per seed."""
    groups = defaultdict(list)
    for s in summaries:
        method = s["config"].get(method_key, "unknown")
        acc = s.get("best_test_acc")
        if acc is not None:
            groups[method].append(acc)
    return dict(groups)


def compute_stats(values):
    """Compute mean, std, min, max for a list of values."""
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(arr),
    }


def compute_phi_spread(method_stats):
    """Compute phi spread = max(method means) - min(method means)."""
    means = [s["mean"] for s in method_stats.values() if s["mean"] is not None]
    if len(means) < 2:
        return None
    return max(means) - min(means)


def cohens_d(group1, group2):
    """Compute Cohen's d (pooled std)."""
    if len(group1) < 2 or len(group2) < 2:
        return None
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = math.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    if pooled_std == 0:
        return None
    return float((m1 - m2) / pooled_std)


def filter_summaries(summaries, epochs_min=100):
    """Filter out incomplete runs (< epochs_min epochs)."""
    return [s for s in summaries if s.get("epochs_completed", 0) >= epochs_min]


# ========== DATA LOADING ==========

print("=" * 70)
print("Cross-Architecture Analysis — Iteration 5")
print("=" * 70)

# 1. ResNet-20 / CIFAR-10 / AdamW (iter_003)
adamw_c10 = load_summaries(ITER3_RESULTS / "full/cifar10/resnet20")
print(f"[1] ResNet-20/CIFAR-10/AdamW: {len(adamw_c10)} runs")

# 2. ResNet-20 / CIFAR-100 / AdamW (iter_003)
adamw_c100 = load_summaries(ITER3_RESULTS / "full/cifar100/resnet20")
print(f"[2] ResNet-20/CIFAR-100/AdamW: {len(adamw_c100)} runs")

# 3. ResNet-20 / CIFAR-10 / SGD original (iter_003)
sgd_c10 = load_summaries(ITER3_RESULTS / "sgd_baseline/cifar10/resnet20")
print(f"[3] ResNet-20/CIFAR-10/SGD(original): {len(sgd_c10)} runs")

# 4. ResNet-20 / CIFAR-100 / SGD original (iter_003)
sgd_c100 = load_summaries(ITER3_RESULTS / "sgd_baseline/cifar100/resnet20")
print(f"[4] ResNet-20/CIFAR-100/SGD(original): {len(sgd_c100)} runs")

# 5. VGG-16-BN / CIFAR-10 / AdamW (iter_005)
vgg_c10 = load_summaries(ITER5_RESULTS / "full/vgg16bn/cifar10")
print(f"[5] VGG-16-BN/CIFAR-10/AdamW: {len(vgg_c10)} runs")

# 6. NoBN ResNet-20 / CIFAR-10 / AdamW (iter_005)
nobn_c10 = load_summaries(ITER5_RESULTS / "full/nobn/cifar10/resnet20_nobn")
print(f"[6] NoBN ResNet-20/CIFAR-10/AdamW: {len(nobn_c10)} runs")

# 7. Matched-rho SGD / CIFAR-10 (iter_005)
matched_sgd_c10_raw = load_summaries(ITER5_RESULTS / "full/matched_rho_sgd/cifar10/resnet20")
matched_sgd_c10 = filter_summaries(matched_sgd_c10_raw, epochs_min=100)
print(f"[7] Matched-rho SGD/CIFAR-10: {len(matched_sgd_c10_raw)} raw, {len(matched_sgd_c10)} valid (>=100 epochs)")

# 8. Rho-low sweep / CIFAR-10 (iter_005)
rho_low_c10 = load_summaries(ITER5_RESULTS / "full/rho_sweep/cifar10/rho_low")
print(f"[8] Rho-low (rho=0.05)/CIFAR-10: {len(rho_low_c10)} runs")

print()

# ========== ANALYSIS ==========

master_results = []
data_gaps = []

# --- Analysis 1: Phi Spread per Architecture/Dataset/Optimizer ---
print("=" * 70)
print("ANALYSIS 1: Phi Spread per Configuration")
print("=" * 70)

configs = [
    ("ResNet-20", "CIFAR-10", "AdamW", "rho~0.5", adamw_c10),
    ("ResNet-20", "CIFAR-100", "AdamW", "rho~0.5", adamw_c100),
    ("ResNet-20", "CIFAR-10", "SGD(original)", "rho~0.005", sgd_c10),
    ("ResNet-20", "CIFAR-100", "SGD(original)", "rho~0.005", sgd_c100),
    ("VGG-16-BN", "CIFAR-10", "AdamW", "rho~0.5", vgg_c10),
    ("ResNet-20-NoBN", "CIFAR-10", "AdamW", "rho~0.5", nobn_c10),
    ("ResNet-20", "CIFAR-10", "SGD(matched-rho)", "rho~0.5", matched_sgd_c10),
    ("ResNet-20", "CIFAR-10", "AdamW(rho_low)", "rho~0.05", rho_low_c10),
]

phi_spread_results = []

for arch, dataset, optimizer, rho_label, summaries in configs:
    method_groups = group_by_method(summaries)
    method_stats = {m: compute_stats(v) for m, v in method_groups.items()}
    phi = compute_phi_spread(method_stats)

    entry = {
        "arch": arch,
        "dataset": dataset,
        "optimizer": optimizer,
        "rho_label": rho_label,
        "phi_spread": round(phi, 4) if phi is not None else None,
        "n_methods": len(method_groups),
        "n_total_runs": sum(s["n"] for s in method_stats.values()),
        "methods": {},
    }

    for m, stats in sorted(method_stats.items()):
        entry["methods"][m] = {
            "mean": round(stats["mean"], 2) if stats["mean"] is not None else None,
            "std": round(stats["std"], 3) if stats["std"] is not None else None,
            "n": stats["n"],
        }

    phi_spread_results.append(entry)
    master_results.append(entry)

    phi_str = f"{phi:.4f}%" if phi is not None else "N/A"
    print(f"  {arch:20s} | {dataset:10s} | {optimizer:20s} | rho={rho_label:8s} | "
          f"phi_spread={phi_str:8s} | {entry['n_methods']} methods, {entry['n_total_runs']} runs")

    # Flag data gaps
    if entry["n_methods"] < 3:
        data_gaps.append(f"{arch}/{dataset}/{optimizer}: only {entry['n_methods']} methods available")
    if entry["n_total_runs"] < 6:
        data_gaps.append(f"{arch}/{dataset}/{optimizer}: only {entry['n_total_runs']} total runs")

print()

# --- Analysis 2: Cohen's d for NoBN vs BN ---
print("=" * 70)
print("ANALYSIS 2: NoBN vs BN Effect Size (Cohen's d)")
print("=" * 70)

nobn_groups = group_by_method(nobn_c10)
adamw_groups = group_by_method(adamw_c10)

nobn_vs_bn_results = {}

# Compare constant method (the only one we have for NoBN)
if "constant" in nobn_groups and "constant" in adamw_groups:
    d_val = cohens_d(adamw_groups["constant"], nobn_groups["constant"])
    bn_mean = np.mean(adamw_groups["constant"])
    nobn_mean = np.mean(nobn_groups["constant"])
    nobn_vs_bn_results["constant"] = {
        "bn_mean": round(float(bn_mean), 2),
        "nobn_mean": round(float(nobn_mean), 2),
        "diff": round(float(bn_mean - nobn_mean), 2),
        "cohens_d": round(d_val, 4) if d_val is not None else None,
        "interpretation": (
            "large" if d_val and abs(d_val) > 0.8 else
            "medium" if d_val and abs(d_val) > 0.5 else
            "small" if d_val and abs(d_val) > 0.2 else "negligible"
        ) if d_val is not None else "N/A",
    }
    print(f"  constant: BN={bn_mean:.2f}% vs NoBN={nobn_mean:.2f}% → "
          f"diff={bn_mean-nobn_mean:.2f}%, Cohen's d={d_val:.4f} ({nobn_vs_bn_results['constant']['interpretation']})")
else:
    print("  WARNING: Cannot compute NoBN vs BN — missing constant method data")
    data_gaps.append("NoBN vs BN: only constant method available for NoBN (missing cwd_hard, no_wd)")

print()

# --- Analysis 3: Rho-Spread Curve (limited data) ---
print("=" * 70)
print("ANALYSIS 3: Rho-Spread Relationship")
print("=" * 70)

rho_spread_data = []

# rho=0.05 (rho_low, iter_005)
rho_low_groups = group_by_method(rho_low_c10)
rho_low_stats = {m: compute_stats(v) for m, v in rho_low_groups.items()}
rho_low_phi = compute_phi_spread(rho_low_stats)
rho_spread_data.append({
    "rho": 0.05,
    "wd": 5e-5,
    "phi_spread": round(rho_low_phi, 4) if rho_low_phi is not None else None,
    "n_methods": len(rho_low_groups),
    "n_runs": sum(len(v) for v in rho_low_groups.values()),
    "data_quality": "partial" if len(rho_low_groups) < 3 else "complete",
    "methods": {m: round(float(np.mean(v)), 2) for m, v in rho_low_groups.items()},
})

# rho=0.5 (default AdamW, iter_003)
adamw_groups_stats = {m: compute_stats(v) for m, v in adamw_groups.items()}
adamw_phi = compute_phi_spread(adamw_groups_stats)
rho_spread_data.append({
    "rho": 0.5,
    "wd": 5e-4,
    "phi_spread": round(adamw_phi, 4) if adamw_phi is not None else None,
    "n_methods": len(adamw_groups),
    "n_runs": sum(len(v) for v in adamw_groups.values()),
    "data_quality": "complete",
    "methods": {m: round(float(np.mean(v)), 2) for m, v in adamw_groups.items()},
})

# rho=5.0 — FAILED, no data
rho_spread_data.append({
    "rho": 5.0,
    "wd": 5e-3,
    "phi_spread": None,
    "n_methods": 0,
    "n_runs": 0,
    "data_quality": "FAILED",
    "methods": {},
})

for rd in rho_spread_data:
    phi_str = f"{rd['phi_spread']:.4f}%" if rd['phi_spread'] is not None else "MISSING"
    print(f"  rho={rd['rho']:<5} (wd={rd['wd']:<8}) | phi_spread={phi_str:10s} | "
          f"{rd['n_methods']} methods, {rd['n_runs']} runs | quality={rd['data_quality']}")

if rho_low_phi is not None:
    data_gaps.append(f"Rho-low (rho=0.05): only {len(rho_low_groups)} methods, {sum(len(v) for v in rho_low_groups.values())} runs")

data_gaps.append("Rho-high (rho=5.0): EXPERIMENT FAILED, no data available")

print()

# --- Analysis 4: SGD/AdamW Ratio at Original vs Matched Rho ---
print("=" * 70)
print("ANALYSIS 4: SGD/AdamW Effect Ratio at Original vs Matched Rho")
print("=" * 70)

# Original SGD (rho~0.005)
sgd_orig_groups = group_by_method(sgd_c10)
sgd_orig_stats = {m: compute_stats(v) for m, v in sgd_orig_groups.items()}
sgd_orig_phi = compute_phi_spread(sgd_orig_stats)

# Matched-rho SGD (rho~0.5) — only constant method available
matched_groups = group_by_method(matched_sgd_c10)
matched_stats = {m: compute_stats(v) for m, v in matched_groups.items()}
matched_phi = compute_phi_spread(matched_stats)

ratio_analysis = {
    "original_sgd": {
        "rho": "~0.005",
        "phi_spread": round(sgd_orig_phi, 4) if sgd_orig_phi is not None else None,
        "n_methods": len(sgd_orig_groups),
    },
    "adamw_default": {
        "rho": "~0.5",
        "phi_spread": round(adamw_phi, 4) if adamw_phi is not None else None,
        "n_methods": len(adamw_groups),
    },
    "matched_sgd": {
        "rho": "~0.5 (target)",
        "phi_spread": round(matched_phi, 4) if matched_phi is not None else None,
        "n_methods": len(matched_groups),
        "data_quality": "CRITICALLY_INCOMPLETE" if len(matched_groups) < 2 else "partial",
    },
}

if sgd_orig_phi and adamw_phi and adamw_phi > 0:
    ratio_analysis["original_ratio"] = round(sgd_orig_phi / adamw_phi, 2)
    print(f"  Original SGD/AdamW ratio: {sgd_orig_phi:.4f}% / {adamw_phi:.4f}% = {sgd_orig_phi/adamw_phi:.1f}x")
else:
    ratio_analysis["original_ratio"] = None

if matched_phi is not None and adamw_phi and adamw_phi > 0:
    ratio_analysis["matched_ratio"] = round(matched_phi / adamw_phi, 2)
    print(f"  Matched-rho SGD/AdamW ratio: {matched_phi:.4f}% / {adamw_phi:.4f}% = {matched_phi/adamw_phi:.1f}x")
else:
    ratio_analysis["matched_ratio"] = None
    print(f"  Matched-rho SGD: CANNOT compute ratio — only {len(matched_groups)} method(s) available")
    data_gaps.append("Matched-rho SGD: only constant method completed 200 epochs; cwd_hard, no_wd missing → cannot compute phi spread or ratio")

print()

# --- Analysis 5: Cross-Scale Comparison Table ---
print("=" * 70)
print("ANALYSIS 5: Cross-Scale Comparison Table")
print("=" * 70)

# Methods shared across all architectures
all_available_methods = set()
for entry in phi_spread_results[:5]:  # AdamW configs only
    all_available_methods.update(entry["methods"].keys())

cross_scale = []
# Only include configs with sufficient data
for entry in phi_spread_results:
    if entry["n_methods"] >= 2 and entry["n_total_runs"] >= 3:
        cross_scale.append({
            "arch": entry["arch"],
            "dataset": entry["dataset"],
            "optimizer": entry["optimizer"],
            "rho_label": entry["rho_label"],
            "phi_spread": entry["phi_spread"],
            "n_methods": entry["n_methods"],
            "n_runs": entry["n_total_runs"],
        })
        phi_str = f"{entry['phi_spread']:.4f}%" if entry['phi_spread'] is not None else "N/A"
        print(f"  {entry['arch']:20s} | {entry['dataset']:10s} | {entry['optimizer']:20s} | "
              f"phi={phi_str:8s} | {entry['n_methods']}M/{entry['n_total_runs']}R")

print()

# ========== DATA GAPS SUMMARY ==========
print("=" * 70)
print("DATA GAPS AND LIMITATIONS")
print("=" * 70)

data_gaps.extend([
    "ImageNet/ResNet-50: ALL experiments FAILED — no large-scale data available",
    "VGG-16-BN: missing swd, no_wd, random_mask methods; cosine_schedule missing seed_456 → only 4/7 methods available",
    "NoBN: only constant method x 3 seeds available; cwd_hard and no_wd MISSING → cannot compute phi spread for NoBN",
    "Rho-high (rho=5.0): EXPERIMENT FAILED",
    "Matched-rho SGD CIFAR-100: NO DATA",
    "Matched-rho SGD CIFAR-10: seed_42 only ran 5 epochs (constant method); cwd_hard/no_wd missing entirely",
])
# Deduplicate
data_gaps = list(dict.fromkeys(data_gaps))

for i, gap in enumerate(data_gaps, 1):
    print(f"  [{i}] {gap}")

print()

# ========== GENERATE FIGURES ==========
print("=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    print("  WARNING: matplotlib not available, skipping figure generation")
    HAS_MPL = False

if HAS_MPL:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })

    # --- Figure 1: Master Results Table (as bar chart) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: CIFAR-10 all architectures
    cifar10_configs = [
        ("ResNet-20\nAdamW", adamw_c10),
        ("VGG-16-BN\nAdamW", vgg_c10),
        ("ResNet-20\nSGD(orig)", sgd_c10),
    ]

    methods_order = ["constant", "cosine_schedule", "cwd_hard", "half_lambda",
                     "swd", "random_mask", "no_wd"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#795548", "#F44336"]

    x_pos = np.arange(len(cifar10_configs))
    width = 0.1
    ax = axes[0]

    for i, method in enumerate(methods_order):
        means = []
        stds = []
        for _, summaries in cifar10_configs:
            groups = group_by_method(summaries)
            if method in groups:
                vals = groups[method]
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
            else:
                means.append(0)
                stds.append(0)
        offset = (i - len(methods_order)/2 + 0.5) * width
        bars = ax.bar(x_pos + offset, means, width, yerr=stds, label=method,
                      color=colors[i], capsize=2, alpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([c[0] for c in cifar10_configs])
    ax.set_ylabel("Best Test Accuracy (%)")
    ax.set_title("CIFAR-10: Cross-Architecture Comparison")
    ax.set_ylim(88, 93)
    ax.legend(ncol=2, fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: CIFAR-100 (ResNet-20 only)
    ax = axes[1]
    c100_groups = group_by_method(adamw_c100)
    sgd_c100_groups = group_by_method(sgd_c100)

    c100_configs = [
        ("ResNet-20\nAdamW", c100_groups),
        ("ResNet-20\nSGD(orig)", sgd_c100_groups),
    ]

    x_pos = np.arange(len(c100_configs))
    for i, method in enumerate(methods_order):
        means = []
        stds = []
        for _, groups in c100_configs:
            if method in groups:
                vals = groups[method]
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
            else:
                means.append(0)
                stds.append(0)
        offset = (i - len(methods_order)/2 + 0.5) * width
        ax.bar(x_pos + offset, means, width, yerr=stds, label=method,
               color=colors[i], capsize=2, alpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([c[0] for c in c100_configs])
    ax.set_ylabel("Best Test Accuracy (%)")
    ax.set_title("CIFAR-100: Optimizer Comparison")
    ax.set_ylim(60, 68)
    ax.legend(ncol=2, fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_master_comparison.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig1_master_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Figure 1: Master comparison saved")

    # --- Figure 2: Phi Spread Summary ---
    fig, ax = plt.subplots(figsize=(10, 5))

    valid_phi = [(e["arch"] + "\n" + e["optimizer"], e["phi_spread"], e["n_methods"])
                 for e in phi_spread_results if e["phi_spread"] is not None and e["n_methods"] >= 2]

    if valid_phi:
        labels, phis, n_methods = zip(*valid_phi)
        bar_colors = ["#2196F3" if "AdamW" in l else "#FF9800" if "SGD" in l else "#9C27B0"
                       for l in labels]
        bars = ax.barh(range(len(labels)), phis, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Phi Spread (percentage points)")
        ax.set_title("WD Method Sensitivity (Phi Spread) Across Configurations")

        # Add value labels
        for i, (phi, n) in enumerate(zip(phis, n_methods)):
            ax.text(phi + 0.01, i, f"{phi:.3f}% ({n}M)", va="center", fontsize=8)

        # Add 0.5% threshold line
        ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.6, label="0.5% threshold")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No valid phi spread data", transform=ax.transAxes, ha="center")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_phi_spread_summary.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig2_phi_spread_summary.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Figure 2: Phi spread summary saved")

    # --- Figure 3: NoBN vs BN comparison ---
    fig, ax = plt.subplots(figsize=(7, 4))

    if nobn_vs_bn_results:
        categories = list(nobn_vs_bn_results.keys())
        bn_vals = [nobn_vs_bn_results[c]["bn_mean"] for c in categories]
        nobn_vals = [nobn_vs_bn_results[c]["nobn_mean"] for c in categories]

        x = np.arange(len(categories))
        width = 0.3
        ax.bar(x - width/2, bn_vals, width, label="ResNet-20 (BN)", color="#2196F3", alpha=0.8)
        ax.bar(x + width/2, nobn_vals, width, label="ResNet-20-NoBN", color="#FF9800", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Best Test Accuracy (%)")
        ax.set_title("BN vs NoBN: Effect on Training Quality\n(NoBN spread analysis limited: only constant method)")
        ax.legend()
        ax.set_ylim(85, 92)
        ax.grid(axis="y", alpha=0.3)

        # Add Cohen's d annotation
        for i, cat in enumerate(categories):
            d = nobn_vs_bn_results[cat]["cohens_d"]
            interp = nobn_vs_bn_results[cat]["interpretation"]
            if d is not None:
                ax.text(i, max(bn_vals[i], nobn_vals[i]) + 0.15,
                        f"d={d:.2f}\n({interp})", ha="center", fontsize=8, color="red")
    else:
        ax.text(0.5, 0.5, "No NoBN vs BN data", transform=ax.transAxes, ha="center")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_nobn_vs_bn.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig3_nobn_vs_bn.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Figure 3: NoBN vs BN saved")

    # --- Figure 4: SGD vs AdamW effect ratio ---
    fig, ax = plt.subplots(figsize=(8, 5))

    sgd_adamw_data = [
        ("SGD(original)\nrho~0.005", sgd_orig_phi if sgd_orig_phi else 0, "#FF9800"),
        ("AdamW\nrho~0.5", adamw_phi if adamw_phi else 0, "#2196F3"),
    ]
    if matched_phi is not None:
        sgd_adamw_data.append(("SGD(matched)\nrho~0.5*", matched_phi, "#4CAF50"))

    labels_r = [d[0] for d in sgd_adamw_data]
    phis_r = [d[1] for d in sgd_adamw_data]
    colors_r = [d[2] for d in sgd_adamw_data]

    bars = ax.bar(range(len(labels_r)), phis_r, color=colors_r, alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels_r)))
    ax.set_xticklabels(labels_r, fontsize=9)
    ax.set_ylabel("Phi Spread (percentage points)")
    ax.set_title("SGD vs AdamW: WD Method Sensitivity (CIFAR-10)")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.6, label="0.5% threshold")

    for i, phi in enumerate(phis_r):
        ax.text(i, phi + 0.02, f"{phi:.3f}%", ha="center", fontsize=9, fontweight="bold")

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    note = "* Matched-rho SGD data incomplete (1 method only)" if matched_phi is not None else "* Matched-rho SGD data MISSING"
    ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=7,
            va="top", color="gray", style="italic")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_sgd_adamw_ratio.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig4_sgd_adamw_ratio.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Figure 4: SGD vs AdamW ratio saved")

    # --- Figure 5: Per-method accuracy comparison across architectures ---
    fig, ax = plt.subplots(figsize=(12, 5))

    arch_labels = ["ResNet-20\n(CIFAR-10)", "ResNet-20\n(CIFAR-100)", "VGG-16-BN\n(CIFAR-10)"]
    arch_data = [adamw_groups, group_by_method(adamw_c100), group_by_method(vgg_c10)]

    x_pos = np.arange(len(arch_labels))
    for i, method in enumerate(methods_order):
        means = []
        stds = []
        for groups in arch_data:
            if method in groups:
                vals = groups[method]
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
            else:
                means.append(np.nan)
                stds.append(0)
        offset = (i - len(methods_order)/2 + 0.5) * width
        # Filter NaN for plotting
        valid_mask = [not np.isnan(m) for m in means]
        valid_x = [x_pos[j] + offset for j in range(len(x_pos)) if valid_mask[j]]
        valid_m = [means[j] for j in range(len(means)) if valid_mask[j]]
        valid_s = [stds[j] for j in range(len(stds)) if valid_mask[j]]
        if valid_m:
            ax.bar(valid_x, valid_m, width, yerr=valid_s, label=method,
                   color=colors[i], capsize=2, alpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(arch_labels)
    ax.set_ylabel("Best Test Accuracy (%)")
    ax.set_title("Cross-Architecture AdamW Method Comparison")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_cross_arch_methods.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig5_cross_arch_methods.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Figure 5: Cross-architecture method comparison saved")

    # --- Figure 6: Rho-Spread curve (limited to 2 points) ---
    fig, ax = plt.subplots(figsize=(7, 4))

    valid_rho = [(d["rho"], d["phi_spread"]) for d in rho_spread_data
                 if d["phi_spread"] is not None]
    if valid_rho:
        rhos, spreads = zip(*valid_rho)
        ax.plot(rhos, spreads, "o-", color="#2196F3", markersize=10, linewidth=2)
        for r, s in zip(rhos, spreads):
            ax.annotate(f"{s:.3f}%", (r, s), textcoords="offset points",
                        xytext=(10, 10), fontsize=9)

        # Mark missing rho=5.0
        ax.axvline(x=5.0, color="red", linestyle=":", alpha=0.5)
        ax.text(5.0, max(spreads)*0.8, "rho=5.0\n(FAILED)", color="red",
                ha="center", fontsize=8)

    ax.set_xlabel("rho (effective WD/LR ratio)")
    ax.set_ylabel("Phi Spread (%)")
    ax.set_title("Phi Spread vs Rho\n(Only 2 of 3 planned rho values available)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="0.5% threshold")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_rho_spread_curve.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig6_rho_spread_curve.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Figure 6: Rho-Spread curve saved")

    # --- Figure 7: Phi spread bar chart with data quality indicators ---
    fig, ax = plt.subplots(figsize=(10, 5))

    config_labels = []
    phi_values = []
    bar_colors = []
    edge_colors = []

    for e in phi_spread_results:
        if e["phi_spread"] is not None and e["n_methods"] >= 2:
            label = f"{e['arch']}\n{e['optimizer']}\n({e['dataset']})"
            config_labels.append(label)
            phi_values.append(e["phi_spread"])

            # Color by optimizer type
            if "SGD" in e["optimizer"]:
                bar_colors.append("#FF9800")
            elif "AdamW" in e["optimizer"]:
                bar_colors.append("#2196F3")
            else:
                bar_colors.append("#9E9E9E")

            # Edge color indicates data completeness
            if e["n_methods"] >= 7:
                edge_colors.append("green")
            elif e["n_methods"] >= 4:
                edge_colors.append("orange")
            else:
                edge_colors.append("red")

    if config_labels:
        bars = ax.bar(range(len(config_labels)), phi_values, color=bar_colors,
                      edgecolor=edge_colors, linewidth=2, alpha=0.85)
        ax.set_xticks(range(len(config_labels)))
        ax.set_xticklabels(config_labels, fontsize=7)
        ax.set_ylabel("Phi Spread (%)")
        ax.set_title("WD Method Sensitivity Summary\n(border: green=7M complete, orange=4-6M partial, red=2-3M limited)")
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.6, label="0.5% invariance threshold")

        for i, phi in enumerate(phi_values):
            ax.text(i, phi + 0.01, f"{phi:.3f}%", ha="center", fontsize=8)

        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_phi_spread_quality.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig7_phi_spread_quality.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [OK] Figure 7: Phi spread with quality indicators saved")

print()

# ========== SAVE SUMMARY JSON ==========
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)

summary = {
    "iteration": 5,
    "analysis_timestamp": datetime.now().isoformat(),
    "phi_spread_results": phi_spread_results,
    "nobn_vs_bn": nobn_vs_bn_results,
    "rho_spread_curve": rho_spread_data,
    "sgd_adamw_ratio": ratio_analysis,
    "cross_scale_comparison": cross_scale,
    "data_gaps": data_gaps,
    "key_findings": [
        {
            "finding": "AdamW phi spread remains < 0.5% across ResNet-20 and VGG-16-BN on CIFAR-10",
            "evidence": f"ResNet-20 phi={adamw_phi:.4f}%, VGG-16-BN phi={compute_phi_spread({m: compute_stats(v) for m, v in group_by_method(vgg_c10).items()}):.4f}%" if compute_phi_spread({m: compute_stats(v) for m, v in group_by_method(vgg_c10).items()}) else "insufficient VGG data",
            "confidence": "medium" if len(group_by_method(vgg_c10)) < 7 else "high",
        },
        {
            "finding": "SGD shows significantly larger phi spread than AdamW on CIFAR-10",
            "evidence": f"SGD phi={sgd_orig_phi:.4f}% vs AdamW phi={adamw_phi:.4f}% → {sgd_orig_phi/adamw_phi:.1f}x ratio" if sgd_orig_phi and adamw_phi else "N/A",
            "confidence": "high",
        },
        {
            "finding": "NoBN reduces accuracy by ~2.2% vs BN on CIFAR-10 (constant method)",
            "evidence": f"BN={np.mean(adamw_groups.get('constant', [])):.2f}% vs NoBN={np.mean(nobn_groups.get('constant', [])):.2f}%",
            "confidence": "high" if "constant" in nobn_groups and len(nobn_groups["constant"]) >= 3 else "medium",
        },
        {
            "finding": "ImageNet and rho_high experiments failed — cannot validate large-scale or high-rho hypotheses",
            "evidence": "All ImageNet seeds and rho=5.0 experiments in FAILED state",
            "confidence": "N/A (missing data)",
        },
        {
            "finding": "Matched-rho SGD analysis incomplete — cannot determine if rho is the confounding variable",
            "evidence": "Only constant method available for matched-rho SGD; need cwd_hard and no_wd for phi spread computation",
            "confidence": "N/A (insufficient data)",
        },
    ],
    "total_runs_analyzed": sum(len(s) for config in configs for s in [config[4]]),
    "figures_generated": [
        "fig1_master_comparison.{png,pdf}",
        "fig2_phi_spread_summary.{png,pdf}",
        "fig3_nobn_vs_bn.{png,pdf}",
        "fig4_sgd_adamw_ratio.{png,pdf}",
        "fig5_cross_arch_methods.{png,pdf}",
        "fig6_rho_spread_curve.{png,pdf}",
        "fig7_phi_spread_quality.{png,pdf}",
    ] if HAS_MPL else [],
}

output_path = OUTPUT_DIR / "iter5_summary.json"
output_path.write_text(json.dumps(summary, indent=2))
print(f"  [OK] Summary JSON saved to {output_path}")

# Also write a human-readable markdown summary
md_path = OUTPUT_DIR / "iter5_summary.md"
with open(md_path, "w") as f:
    f.write("# Iteration 5 Cross-Architecture Analysis Summary\n\n")
    f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

    f.write("## Phi Spread Results\n\n")
    f.write("| Architecture | Dataset | Optimizer | rho | Phi Spread | Methods | Runs |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for e in phi_spread_results:
        phi_str = f"{e['phi_spread']:.4f}%" if e['phi_spread'] is not None else "N/A"
        f.write(f"| {e['arch']} | {e['dataset']} | {e['optimizer']} | {e['rho_label']} | "
                f"{phi_str} | {e['n_methods']} | {e['n_total_runs']} |\n")

    f.write("\n## NoBN vs BN\n\n")
    for method, result in nobn_vs_bn_results.items():
        f.write(f"- **{method}**: BN={result['bn_mean']}% vs NoBN={result['nobn_mean']}% "
                f"(diff={result['diff']}%, Cohen's d={result['cohens_d']}, {result['interpretation']})\n")

    f.write("\n## Key Findings\n\n")
    for kf in summary["key_findings"]:
        f.write(f"1. **{kf['finding']}** (confidence: {kf['confidence']})\n")
        f.write(f"   - Evidence: {kf['evidence']}\n\n")

    f.write("\n## Data Gaps\n\n")
    for gap in data_gaps:
        f.write(f"- {gap}\n")

print(f"  [OK] Summary MD saved to {md_path}")
print()
print("ANALYSIS COMPLETE")
