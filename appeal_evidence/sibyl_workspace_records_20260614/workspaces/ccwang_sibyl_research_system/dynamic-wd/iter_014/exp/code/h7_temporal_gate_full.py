#!/usr/bin/env python3
"""
H7 Temporal Predictability Gate — FULL analysis using 200-epoch data.

Fits degree-4 polynomials to per-layer alignment trajectories alpha_t(epoch)
from Phase 1 (diagnostic_cifar10) and Phase 2 (ablation_cifar100).
Computes R-squared. Gate PASSES if <70% of layer-method combinations have R^2 >= 0.85,
meaning alignment is NOT trivially predictable from time alone.

If gate passes: alignment signal carries information beyond temporal trend.
If gate fails: alignment is mostly a function of training progress (time proxy).
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ── Paths ──────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current")
PHASE1_DIR = WORKSPACE / "exp/results/full/phase1_diagnostic"
PHASE2_DIR = WORKSPACE / "exp/results/full/phase2_ablation"
OUTPUT_DIR = WORKSPACE / "exp/results/full/phase3_alignment"
FIGURES_DIR = WORKSPACE / "exp/results/full/figures"
TASK_ID = "h7_temporal_gate"

# ── Config ─────────────────────────────────────────────────────────────
POLY_DEGREE = 4
R2_THRESHOLD = 0.85
GATE_THRESHOLD_PCT = 70.0  # Gate passes if pct_above < this


def load_trajectories(base_dir, source_name):
    """Load all alpha_t trajectories from a phase directory."""
    records = []
    base = Path(base_dir)
    if not base.exists():
        print(f"[WARN] {base} does not exist, skipping.")
        return records

    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        traj_files = list(subdir.glob("*_trajectories.json"))
        if not traj_files:
            continue
        traj_file = traj_files[0]
        try:
            with open(traj_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARN] Failed to load {traj_file}: {e}")
            continue

        method = data.get("method", subdir.name.rsplit("_seed", 1)[0])
        seed = data.get("seed", "?")
        model = data.get("model", "unknown")
        dataset = data.get("dataset", "unknown")
        layers = data.get("layers", {})

        for layer_name, layer_data in layers.items():
            alpha_t = layer_data.get("alpha_t", [])
            if len(alpha_t) < 10:  # Need enough data points
                continue

            # Classify layer type
            if "conv" in layer_name.lower() or "features" in layer_name.lower():
                if "weight" in layer_name:
                    layer_type = "conv/weight"
                elif "bias" in layer_name:
                    layer_type = "conv/bias"
                else:
                    layer_type = "conv/other"
            elif "bn" in layer_name.lower() or "batch" in layer_name.lower():
                if "weight" in layer_name:
                    layer_type = "bn/weight"
                elif "bias" in layer_name:
                    layer_type = "bn/bias"
                else:
                    layer_type = "bn/other"
            elif "fc" in layer_name.lower() or "linear" in layer_name.lower() or "classifier" in layer_name.lower():
                if "weight" in layer_name:
                    layer_type = "fc/weight"
                elif "bias" in layer_name:
                    layer_type = "fc/bias"
                else:
                    layer_type = "fc/other"
            elif "shortcut" in layer_name.lower() or "downsample" in layer_name.lower():
                layer_type = "shortcut"
            else:
                layer_type = "other"

            records.append({
                "source": source_name,
                "method": method,
                "seed": seed,
                "model": model,
                "dataset": dataset,
                "layer_name": layer_name,
                "layer_type": layer_type,
                "alpha_t": np.array(alpha_t, dtype=np.float64),
                "n_epochs": len(alpha_t),
            })

    return records


def fit_polynomial_r2(alpha_t, degree=4):
    """Fit degree-k polynomial to alpha_t vs epoch index, return R^2."""
    n = len(alpha_t)
    epochs = np.arange(n, dtype=np.float64)
    y = alpha_t.copy()

    # Handle NaN/Inf
    mask = np.isfinite(y)
    if mask.sum() < degree + 2:
        return np.nan

    epochs_clean = epochs[mask]
    y_clean = y[mask]

    # Normalize x for numerical stability
    x_mean = epochs_clean.mean()
    x_std = epochs_clean.std()
    if x_std < 1e-12:
        return np.nan
    x_norm = (epochs_clean - x_mean) / x_std

    try:
        coeffs = np.polyfit(x_norm, y_clean, degree)
        y_pred = np.polyval(coeffs, x_norm)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan

    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)

    if ss_tot < 1e-20:
        # Constant signal → perfectly predictable
        return 1.0

    r2 = 1.0 - ss_res / ss_tot
    return r2


def compute_residual_alignment(alpha_t, degree=4):
    """Compute residual alignment after removing polynomial trend."""
    n = len(alpha_t)
    epochs = np.arange(n, dtype=np.float64)
    y = alpha_t.copy()

    mask = np.isfinite(y)
    if mask.sum() < degree + 2:
        return None

    x_mean = epochs[mask].mean()
    x_std = epochs[mask].std()
    if x_std < 1e-12:
        return None

    x_norm = (epochs - x_mean) / x_std

    try:
        coeffs = np.polyfit(x_norm[mask], y[mask], degree)
        y_pred = np.polyval(coeffs, x_norm)
    except (np.linalg.LinAlgError, ValueError):
        return None

    residual = y - y_pred
    return residual


def main():
    print("=" * 70)
    print("H7 Temporal Predictability Gate — FULL 200-epoch Analysis")
    print("=" * 70)
    print(f"Polynomial degree: {POLY_DEGREE}")
    print(f"R² threshold: {R2_THRESHOLD}")
    print(f"Gate threshold: {GATE_THRESHOLD_PCT}%")
    print()

    # ── Load data ──────────────────────────────────────────────────────
    print("[1/5] Loading alignment trajectories...")
    records = []
    records.extend(load_trajectories(PHASE1_DIR, "diagnostic_cifar10"))
    records.extend(load_trajectories(PHASE2_DIR, "ablation_cifar100"))

    print(f"  Total layer-method-seed combinations: {len(records)}")
    if not records:
        print("[ERROR] No data loaded!")
        sys.exit(1)

    # Summary
    sources = set(r["source"] for r in records)
    methods = set(r["method"] for r in records)
    print(f"  Sources: {sorted(sources)}")
    print(f"  Methods ({len(methods)}): {sorted(methods)}")
    epoch_counts = [r["n_epochs"] for r in records]
    print(f"  Epoch range: {min(epoch_counts)}-{max(epoch_counts)}")
    print()

    # ── Fit polynomials ────────────────────────────────────────────────
    print("[2/5] Fitting degree-4 polynomials to alpha_t trajectories...")
    r2_values = []
    detailed_results = []
    residual_stats = []

    for i, rec in enumerate(records):
        r2 = fit_polynomial_r2(rec["alpha_t"], POLY_DEGREE)
        r2_values.append(r2)

        # Compute residual stats
        residual = compute_residual_alignment(rec["alpha_t"], POLY_DEGREE)
        res_std = float(np.nanstd(residual)) if residual is not None else None
        res_autocorr = None
        if residual is not None and len(residual) > 1:
            # Lag-1 autocorrelation of residual
            r_clean = residual[np.isfinite(residual)]
            if len(r_clean) > 2:
                r_centered = r_clean - r_clean.mean()
                denom = np.sum(r_centered**2)
                if denom > 1e-20:
                    res_autocorr = float(np.sum(r_centered[:-1] * r_centered[1:]) / denom)

        detailed_results.append({
            "source": rec["source"],
            "method": rec["method"],
            "seed": rec["seed"],
            "layer_name": rec["layer_name"],
            "layer_type": rec["layer_type"],
            "model": rec["model"],
            "dataset": rec["dataset"],
            "n_epochs": rec["n_epochs"],
            "r2": float(r2) if np.isfinite(r2) else None,
            "residual_std": res_std,
            "residual_autocorr_lag1": res_autocorr,
        })

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(records)}...")

    r2_arr = np.array(r2_values)
    valid_mask = np.isfinite(r2_arr)
    r2_valid = r2_arr[valid_mask]

    print(f"  Valid R² values: {len(r2_valid)}/{len(r2_arr)}")
    print()

    # ── Aggregate statistics ───────────────────────────────────────────
    print("[3/5] Computing aggregate statistics...")

    # Overall stats
    pct_above = float(100.0 * np.sum(r2_valid >= R2_THRESHOLD) / len(r2_valid))
    count_above = int(np.sum(r2_valid >= R2_THRESHOLD))
    gate_passes = pct_above < GATE_THRESHOLD_PCT
    h7_falsified = not gate_passes  # H7 falsified if gate fails

    stats = {
        "total_layer_method_combinations": len(r2_arr),
        "valid_r2_count": int(len(r2_valid)),
        "mean_r2": float(np.mean(r2_valid)),
        "median_r2": float(np.median(r2_valid)),
        "std_r2": float(np.std(r2_valid)),
        "min_r2": float(np.min(r2_valid)),
        "max_r2": float(np.max(r2_valid)),
        "q25_r2": float(np.percentile(r2_valid, 25)),
        "q75_r2": float(np.percentile(r2_valid, 75)),
        "q10_r2": float(np.percentile(r2_valid, 10)),
        "q90_r2": float(np.percentile(r2_valid, 90)),
        "pct_r2_above_085": pct_above,
        "count_r2_above_085": count_above,
        "pct_r2_above_070": float(100.0 * np.sum(r2_valid >= 0.70) / len(r2_valid)),
        "pct_r2_above_050": float(100.0 * np.sum(r2_valid >= 0.50) / len(r2_valid)),
        "pct_r2_below_030": float(100.0 * np.sum(r2_valid < 0.30) / len(r2_valid)),
    }

    print(f"  Mean R²: {stats['mean_r2']:.4f}")
    print(f"  Median R²: {stats['median_r2']:.4f}")
    print(f"  R² ≥ 0.85: {pct_above:.1f}% ({count_above}/{len(r2_valid)})")
    print(f"  R² ≥ 0.70: {stats['pct_r2_above_070']:.1f}%")
    print(f"  R² ≥ 0.50: {stats['pct_r2_above_050']:.1f}%")
    print(f"  R² < 0.30: {stats['pct_r2_below_030']:.1f}%")
    print(f"  Gate decision: {'PASS' if gate_passes else 'FAIL'}")
    print(f"  H7 falsified: {h7_falsified}")
    print()

    # Per-method stats
    per_method = {}
    for m in sorted(methods):
        m_r2 = [r2_arr[i] for i, r in enumerate(records) if r["method"] == m and np.isfinite(r2_arr[i])]
        if not m_r2:
            continue
        m_r2 = np.array(m_r2)
        per_method[m] = {
            "mean_r2": float(np.mean(m_r2)),
            "median_r2": float(np.median(m_r2)),
            "std_r2": float(np.std(m_r2)),
            "pct_high_r2": float(100.0 * np.sum(m_r2 >= R2_THRESHOLD) / len(m_r2)),
            "pct_above_070": float(100.0 * np.sum(m_r2 >= 0.70) / len(m_r2)),
            "n_layers": len(m_r2),
        }

    # Per-source stats
    per_source = {}
    for s in sorted(sources):
        s_r2 = [r2_arr[i] for i, r in enumerate(records) if r["source"] == s and np.isfinite(r2_arr[i])]
        if not s_r2:
            continue
        s_r2 = np.array(s_r2)
        per_source[s] = {
            "mean_r2": float(np.mean(s_r2)),
            "median_r2": float(np.median(s_r2)),
            "pct_high_r2": float(100.0 * np.sum(s_r2 >= R2_THRESHOLD) / len(s_r2)),
            "n_layers": len(s_r2),
        }

    # Per-layer-type stats
    layer_types = set(r["layer_type"] for r in records)
    per_layer_type = {}
    for lt in sorted(layer_types):
        lt_r2 = [r2_arr[i] for i, r in enumerate(records) if r["layer_type"] == lt and np.isfinite(r2_arr[i])]
        if not lt_r2:
            continue
        lt_r2 = np.array(lt_r2)
        per_layer_type[lt] = {
            "mean_r2": float(np.mean(lt_r2)),
            "median_r2": float(np.median(lt_r2)),
            "std_r2": float(np.std(lt_r2)),
            "pct_high_r2": float(100.0 * np.sum(lt_r2 >= R2_THRESHOLD) / len(lt_r2)),
            "n_layers": len(lt_r2),
        }

    # Per-seed stats (cross-seed consistency)
    seeds = sorted(set(r["seed"] for r in records))
    per_seed = {}
    for seed in seeds:
        s_r2 = [r2_arr[i] for i, r in enumerate(records) if r["seed"] == seed and np.isfinite(r2_arr[i])]
        if not s_r2:
            continue
        s_r2 = np.array(s_r2)
        per_seed[str(seed)] = {
            "mean_r2": float(np.mean(s_r2)),
            "pct_high_r2": float(100.0 * np.sum(s_r2 >= R2_THRESHOLD) / len(s_r2)),
            "n_layers": len(s_r2),
        }

    # Residual analysis summary
    valid_residuals = [d for d in detailed_results if d["residual_std"] is not None]
    residual_summary = {}
    if valid_residuals:
        res_stds = np.array([d["residual_std"] for d in valid_residuals])
        res_acorr = np.array([d["residual_autocorr_lag1"] for d in valid_residuals
                              if d["residual_autocorr_lag1"] is not None])
        residual_summary = {
            "mean_residual_std": float(np.mean(res_stds)),
            "median_residual_std": float(np.median(res_stds)),
            "mean_lag1_autocorr": float(np.mean(res_acorr)) if len(res_acorr) > 0 else None,
            "pct_autocorr_above_05": float(100.0 * np.sum(np.abs(res_acorr) > 0.5) / len(res_acorr))
                if len(res_acorr) > 0 else None,
            "interpretation": (
                "High residual autocorrelation suggests structured non-polynomial "
                "dynamics in alignment — alignment carries temporally complex information."
            ),
        }

    # ── Build histogram ────────────────────────────────────────────────
    bin_edges = np.linspace(-0.5, 1.0, 31)
    hist_counts, hist_edges = np.histogram(r2_valid, bins=bin_edges)

    histogram = {
        "counts": hist_counts.tolist(),
        "bin_edges": hist_edges.tolist(),
    }

    # ── Save results JSON ──────────────────────────────────────────────
    print("[4/5] Saving results...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "task_id": TASK_ID,
        "mode": "full",
        "polynomial_degree": POLY_DEGREE,
        "gate_decision": "PASS" if gate_passes else "FAIL",
        "gate_passes": gate_passes,
        "h7_falsified": h7_falsified,
        "gate_threshold_pct": GATE_THRESHOLD_PCT,
        "r2_threshold": R2_THRESHOLD,
        "statistics": stats,
        "per_method_stats": per_method,
        "per_source_stats": per_source,
        "per_layer_type_stats": per_layer_type,
        "per_seed_stats": per_seed,
        "residual_analysis": residual_summary,
        "histogram": histogram,
        "comparison_with_pilot": {
            "pilot_pct_above_085": 11.83,
            "pilot_mean_r2": 0.5285,
            "pilot_data_epochs": 10,
            "full_pct_above_085": pct_above,
            "full_mean_r2": float(np.mean(r2_valid)),
            "full_data_epochs": 200,
            "note": "Full 200-epoch data provides more reliable polynomial fitting than 10-epoch pilot."
        },
        "recommendation": (
            "Alignment signal carries information beyond time. Proceed with direct alignment comparisons."
            if gate_passes else
            "Alignment is largely predictable from time alone. Use residual alignment for subsequent comparisons."
        ),
        "sources_used": sorted(sources),
        "data_epochs": 200,
        "seeds_used": [int(s) if isinstance(s, (int, float)) else s for s in seeds],
        "timestamp": datetime.now().isoformat(),
    }

    output_file = OUTPUT_DIR / "h7_gate_result.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to: {output_file}")

    # Save detailed per-layer results (for downstream use)
    detail_file = OUTPUT_DIR / "h7_detailed_r2.json"
    with open(detail_file, "w") as f:
        json.dump(detailed_results, f, indent=2)
    print(f"  Detailed results saved to: {detail_file}")

    # ── Generate visualizations ────────────────────────────────────────
    print("[5/5] Generating visualizations...")

    # Set publication-quality defaults
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'serif',
    })

    # Colorblind-safe palette
    colors = {
        'main': '#2271B2',
        'accent': '#D55E00',
        'threshold': '#CC3311',
        'gate_line': '#009988',
        'conv': '#2271B2',
        'bn': '#EE7733',
        'fc': '#009988',
        'other': '#AA3377',
    }

    # ─── Figure 1: Main R² histogram ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    n, bins, patches = ax.hist(r2_valid, bins=30, range=(-0.1, 1.0),
                                color=colors['main'], alpha=0.75, edgecolor='white',
                                linewidth=0.5)
    ax.axvline(x=R2_THRESHOLD, color=colors['threshold'], linestyle='--',
               linewidth=2, label=f'R² threshold = {R2_THRESHOLD}')
    ax.set_xlabel('R² (Degree-4 Polynomial Fit)')
    ax.set_ylabel('Count')
    ax.set_title(f'H7 Temporal Predictability Gate\n'
                 f'{pct_above:.1f}% exceed R²={R2_THRESHOLD} → Gate {"PASSES" if gate_passes else "FAILS"}')
    ax.legend(loc='upper left')

    # Add annotation
    ax.annotate(f'Only {pct_above:.1f}% of layer-method\ncombinations are\ntemporally predictable',
                xy=(R2_THRESHOLD, max(n) * 0.8), xytext=(0.5, max(n) * 0.9),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color=colors['threshold']),
                color=colors['threshold'])

    fig_path = FIGURES_DIR / "h7_r2_histogram.png"
    fig.savefig(fig_path)
    fig.savefig(FIGURES_DIR / "h7_r2_histogram.pdf")
    plt.close(fig)
    print(f"  Histogram saved: {fig_path}")

    # ─── Figure 2: R² by layer type ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    layer_type_order = ['conv/weight', 'conv/bias', 'bn/weight', 'bn/bias',
                        'fc/weight', 'fc/bias', 'shortcut', 'other']
    layer_type_order = [lt for lt in layer_type_order if lt in per_layer_type]

    positions = range(len(layer_type_order))
    box_data = []
    for lt in layer_type_order:
        lt_r2 = [r2_arr[i] for i, r in enumerate(records) if r["layer_type"] == lt and np.isfinite(r2_arr[i])]
        box_data.append(lt_r2)

    bp = ax.boxplot(box_data, positions=positions, patch_artist=True,
                     showmeans=True, meanline=True,
                     meanprops=dict(color='red', linewidth=1.5),
                     medianprops=dict(color='black', linewidth=1.5))

    type_colors = {
        'conv/weight': colors['conv'], 'conv/bias': colors['conv'],
        'bn/weight': colors['bn'], 'bn/bias': colors['bn'],
        'fc/weight': colors['fc'], 'fc/bias': colors['fc'],
        'shortcut': colors['other'], 'other': '#888888',
    }
    for patch, lt in zip(bp['boxes'], layer_type_order):
        patch.set_facecolor(type_colors.get(lt, '#888888'))
        patch.set_alpha(0.6)

    ax.axhline(y=R2_THRESHOLD, color=colors['threshold'], linestyle='--',
               linewidth=1.5, label=f'R² = {R2_THRESHOLD}')
    ax.set_xticks(positions)
    ax.set_xticklabels(layer_type_order, rotation=30, ha='right')
    ax.set_ylabel('R² (Degree-4 Polynomial Fit)')
    ax.set_title('R² Distribution by Layer Type')
    ax.legend(loc='upper right')

    fig_path = FIGURES_DIR / "h7_r2_by_layer_type.png"
    fig.savefig(fig_path)
    fig.savefig(FIGURES_DIR / "h7_r2_by_layer_type.pdf")
    plt.close(fig)
    print(f"  Layer type boxplot saved: {fig_path}")

    # ─── Figure 3: R² by method ──────────────────────────────────────
    # Group Phase 1 methods (main methods) and Phase 2 methods (ablation variants)
    phase1_methods = sorted([m for m in methods if not any(
        x in m for x in ['Kp_only', 'Ki_only', 'Kd_only', 'PI_control', 'PD_control', 'Full_PID']
    )])
    phase2_variants = sorted([m for m in methods if any(
        x in m for x in ['Kp_only', 'Ki_only', 'Kd_only', 'PI_control', 'PD_control', 'Full_PID']
    )])

    all_methods_ordered = phase1_methods + phase2_variants

    fig, ax = plt.subplots(figsize=(14, 6))
    method_means = []
    method_pct_high = []
    for m in all_methods_ordered:
        m_info = per_method.get(m, {})
        method_means.append(m_info.get("mean_r2", 0))
        method_pct_high.append(m_info.get("pct_high_r2", 0))

    x_pos = np.arange(len(all_methods_ordered))
    bars = ax.bar(x_pos, method_means, color=colors['main'], alpha=0.7, edgecolor='white')
    ax.axhline(y=R2_THRESHOLD, color=colors['threshold'], linestyle='--',
               linewidth=1.5, label=f'R² = {R2_THRESHOLD}')

    # Add pct_high annotation
    for i, (m, p) in enumerate(zip(method_means, method_pct_high)):
        ax.text(i, m + 0.02, f'{p:.0f}%', ha='center', va='bottom', fontsize=8, color=colors['accent'])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_methods_ordered, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean R²')
    ax.set_title('Mean R² by Method (% exceeding threshold annotated)')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)

    fig_path = FIGURES_DIR / "h7_r2_by_method.png"
    fig.savefig(fig_path)
    fig.savefig(FIGURES_DIR / "h7_r2_by_method.pdf")
    plt.close(fig)
    print(f"  Method bar chart saved: {fig_path}")

    # ─── Figure 4: Pilot vs Full comparison ──────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Distribution comparison (can only show full since pilot raw data isn't available)
    ax1.hist(r2_valid, bins=30, range=(-0.1, 1.0), color=colors['main'],
             alpha=0.7, edgecolor='white', label='Full (200 epochs)')
    ax1.axvline(x=R2_THRESHOLD, color=colors['threshold'], linestyle='--',
                linewidth=2, label=f'Threshold = {R2_THRESHOLD}')
    ax1.set_xlabel('R²')
    ax1.set_ylabel('Count')
    ax1.set_title('R² Distribution (Full 200-epoch)')
    ax1.legend(fontsize=9)

    # Panel B: Summary comparison
    comparison_data = {
        'Pilot (10ep)': [11.83, 0.5285],
        'Full (200ep)': [pct_above, float(np.mean(r2_valid))],
    }
    labels = list(comparison_data.keys())
    pct_vals = [v[0] for v in comparison_data.values()]
    mean_vals = [v[1] for v in comparison_data.values()]

    x = np.arange(2)
    width = 0.35
    ax2_twin = ax2.twinx()
    bars1 = ax2.bar(x - width/2, pct_vals, width, label='% ≥ R²=0.85',
                     color=colors['accent'], alpha=0.7)
    bars2 = ax2_twin.bar(x + width/2, mean_vals, width, label='Mean R²',
                          color=colors['main'], alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('% ≥ R²=0.85', color=colors['accent'])
    ax2_twin.set_ylabel('Mean R²', color=colors['main'])
    ax2.set_title('Pilot vs Full Comparison')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    fig.tight_layout()
    fig_path = FIGURES_DIR / "h7_pilot_vs_full.png"
    fig.savefig(fig_path)
    fig.savefig(FIGURES_DIR / "h7_pilot_vs_full.pdf")
    plt.close(fig)
    print(f"  Pilot vs Full comparison saved: {fig_path}")

    # ── Print summary ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("H7 TEMPORAL PREDICTABILITY GATE — RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Data: {len(r2_valid)} layer-method-seed combinations")
    print(f"  Training epochs: 200 (full)")
    print(f"  Seeds used: {seeds}")
    print(f"  Sources: Phase 1 (CIFAR-10/ResNet-20) + Phase 2 (CIFAR-100/VGG-16-BN)")
    print(f"  Polynomial degree: {POLY_DEGREE}")
    print()
    print(f"  Mean R²:   {stats['mean_r2']:.4f}")
    print(f"  Median R²: {stats['median_r2']:.4f}")
    print(f"  Std R²:    {stats['std_r2']:.4f}")
    print(f"  [Q10, Q90]: [{stats['q10_r2']:.3f}, {stats['q90_r2']:.3f}]")
    print()
    print(f"  % with R² ≥ 0.85: {pct_above:.1f}% ({count_above}/{len(r2_valid)})")
    print(f"  % with R² ≥ 0.70: {stats['pct_r2_above_070']:.1f}%")
    print(f"  % with R² < 0.30: {stats['pct_r2_below_030']:.1f}%")
    print()
    print(f"  GATE DECISION: {'PASS ✓' if gate_passes else 'FAIL ✗'}")
    print(f"  H7 FALSIFIED: {h7_falsified}")
    print()

    if gate_passes:
        print("  INTERPRETATION: Alignment trajectories are NOT well-approximated by")
        print("  simple polynomial functions of time. The alignment signal carries")
        print("  information beyond temporal trends, supporting its use as an")
        print("  independent input signal for dynamic weight decay control.")
    else:
        print("  INTERPRETATION: Alignment trajectories ARE well-approximated by")
        print("  polynomial functions of time. Alignment is largely a time proxy.")
        print("  Residual alignment should be used for subsequent comparisons.")

    print()
    print("  Layer type analysis:")
    for lt in ['conv/weight', 'bn/weight', 'fc/weight']:
        if lt in per_layer_type:
            info = per_layer_type[lt]
            print(f"    {lt:15s}: mean R² = {info['mean_r2']:.3f}, "
                  f"% high = {info['pct_high_r2']:.1f}%  (n={info['n_layers']})")

    print()
    print("  Key findings per method (Phase 1 main methods):")
    for m in phase1_methods:
        if m in per_method:
            info = per_method[m]
            print(f"    {m:22s}: mean R² = {info['mean_r2']:.3f}, "
                  f"% ≥ 0.85 = {info['pct_high_r2']:.1f}%")

    if residual_summary:
        print()
        print("  Residual analysis (after removing polynomial trend):")
        print(f"    Mean residual std: {residual_summary['mean_residual_std']:.5f}")
        if residual_summary.get('mean_lag1_autocorr') is not None:
            print(f"    Mean lag-1 autocorrelation: {residual_summary['mean_lag1_autocorr']:.3f}")
            print(f"    % with |autocorr| > 0.5: {residual_summary.get('pct_autocorr_above_05', 0):.1f}%")

    print()
    print("=" * 70)

    # ── Write DONE marker ──────────────────────────────────────────────
    results_dir = WORKSPACE / "exp/results"
    done_marker = results_dir / f"{TASK_ID}_DONE"
    done_data = {
        "task_id": TASK_ID,
        "status": "success",
        "summary": f"H7 gate {'PASSES' if gate_passes else 'FAILS'}: "
                   f"{pct_above:.1f}% exceed R²={R2_THRESHOLD} (threshold: {GATE_THRESHOLD_PCT}%). "
                   f"Mean R²={stats['mean_r2']:.4f}. "
                   f"Alignment carries {'meaningful' if gate_passes else 'limited'} information beyond time.",
        "final_progress": {
            "pct_above_threshold": pct_above,
            "mean_r2": stats['mean_r2'],
            "gate_decision": "PASS" if gate_passes else "FAIL",
        },
        "timestamp": datetime.now().isoformat(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(done_marker, "w") as f:
        json.dump(done_data, f, indent=2)
    print(f"DONE marker written: {done_marker}")

    return result


if __name__ == "__main__":
    result = main()
