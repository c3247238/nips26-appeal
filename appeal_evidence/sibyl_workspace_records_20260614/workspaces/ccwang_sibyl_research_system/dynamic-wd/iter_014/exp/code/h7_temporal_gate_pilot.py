#!/usr/bin/env python3
"""
H7 Temporal Predictability Gate (Pilot Mode)

Tests whether the alignment signal alpha_t is merely a proxy for time (epoch).
For each layer and each run, fit a degree-4 polynomial to alpha_t as a function
of epoch. Compute R-squared. H7 is falsified if R^2 > 0.85 in >= 70% of
layer-runs (alignment is just a time proxy).

If gate fails (alignment is time-predictable), all subsequent alignment
comparisons should use residual alignment (alpha_t minus polynomial fit).

Pilot mode: uses available pilot data (10 epochs, seed 42).
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

# --- PID file for system monitor ---
TASK_ID = "h7_temporal_gate"
RESULTS_DIR = Path("exp/results/pilots/h7_temporal_gate")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(task_id, results_dir, epoch, total_epochs, step=0,
                    total_steps=0, loss=None, metric=None):
    """Write progress file for system monitor to track."""
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id, results_dir, status="success", summary=""):
    """Write DONE marker file for system monitor to detect."""
    pid_f = Path(results_dir) / f"{task_id}.pid"
    if pid_f.exists():
        pid_f.unlink()
    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def load_trajectory_file(filepath):
    """Load a trajectory JSON file and return structured data."""
    with open(filepath) as f:
        data = json.load(f)
    return data


def fit_polynomial_and_r2(epochs, alpha_values, degree=4):
    """
    Fit a polynomial of given degree to alpha_t vs epoch.
    Returns coefficients and R-squared.
    """
    epochs = np.array(epochs, dtype=float)
    alpha_values = np.array(alpha_values, dtype=float)

    # Handle NaN/Inf
    valid_mask = np.isfinite(alpha_values)
    if valid_mask.sum() < degree + 1:
        return None, float('nan')

    epochs_clean = epochs[valid_mask]
    alpha_clean = alpha_values[valid_mask]

    # Fit polynomial
    try:
        coeffs = np.polyfit(epochs_clean, alpha_clean, min(degree, len(epochs_clean) - 1))
        poly = np.poly1d(coeffs)
        predicted = poly(epochs_clean)

        # R-squared
        ss_res = np.sum((alpha_clean - predicted) ** 2)
        ss_tot = np.sum((alpha_clean - np.mean(alpha_clean)) ** 2)

        if ss_tot < 1e-15:
            # Constant alignment -> perfectly predictable by time (trivially R^2=1)
            # But this is the degenerate case; mark differently
            r2 = 1.0
        else:
            r2 = 1.0 - ss_res / ss_tot

        return coeffs.tolist(), float(r2)
    except Exception as e:
        print(f"  Warning: polynomial fit failed: {e}")
        return None, float('nan')


def analyze_source(source_name, trajectory_files, poly_degree=4):
    """
    Analyze all trajectory files from a single experiment source.
    Returns per-layer, per-method R^2 values.
    """
    results = []

    for traj_file in trajectory_files:
        data = load_trajectory_file(traj_file)
        method = data.get('method', 'unknown')
        seed = data.get('seed', 'unknown')
        model = data.get('model', 'unknown')
        dataset = data.get('dataset', 'unknown')
        layers = data.get('layers', {})

        num_epochs = None
        for layer_name, layer_data in layers.items():
            alpha_t = layer_data.get('alpha_t', [])
            if not alpha_t:
                continue
            if num_epochs is None:
                num_epochs = len(alpha_t)

            epochs = list(range(1, len(alpha_t) + 1))
            coeffs, r2 = fit_polynomial_and_r2(epochs, alpha_t, poly_degree)

            results.append({
                'source': source_name,
                'method': method,
                'seed': seed,
                'model': model,
                'dataset': dataset,
                'layer': layer_name,
                'n_epochs': len(alpha_t),
                'r_squared': r2,
                'coefficients': coeffs,
                'alpha_mean': float(np.mean(alpha_t)),
                'alpha_std': float(np.std(alpha_t)),
                'alpha_range': float(np.max(alpha_t) - np.min(alpha_t)),
            })

    return results


def main():
    workspace = Path(".")
    pilot_dir = workspace / "exp" / "results" / "pilots"
    output_dir = RESULTS_DIR

    print("=" * 70)
    print("H7 Temporal Predictability Gate — Pilot Analysis")
    print("=" * 70)

    all_results = []
    total_steps = 2  # diagnostic + ablation sources
    step = 0

    # Source 1: diagnostic_cifar10
    print("\n--- Source 1: diagnostic_cifar10 (ResNet-20/CIFAR-10) ---")
    diag_dir = pilot_dir / "diagnostic_cifar10"
    diag_files = []
    if diag_dir.exists():
        for method_dir in sorted(diag_dir.iterdir()):
            if method_dir.is_dir():
                for f in method_dir.glob("*_trajectories.json"):
                    diag_files.append(f)
        if not diag_files:
            # Also check flat layout
            for f in diag_dir.glob("*_trajectories.json"):
                diag_files.append(f)

    if diag_files:
        print(f"  Found {len(diag_files)} trajectory files")
        diag_results = analyze_source("diagnostic_cifar10", diag_files)
        all_results.extend(diag_results)
        print(f"  Analyzed {len(diag_results)} layer-method combinations")
    else:
        print("  WARNING: No trajectory files found for diagnostic_cifar10")

    step += 1
    report_progress(TASK_ID, output_dir, step, total_steps,
                    metric={"diagnostic_layers": len(diag_results) if diag_files else 0})

    # Source 2: ablation_cifar100
    print("\n--- Source 2: ablation_cifar100 (VGG-16-BN/CIFAR-100) ---")
    ablation_dir = pilot_dir / "ablation_cifar100"
    ablation_files = []
    if ablation_dir.exists():
        for f in ablation_dir.glob("*_trajectories.json"):
            ablation_files.append(f)

    if ablation_files:
        print(f"  Found {len(ablation_files)} trajectory files")
        ablation_results = analyze_source("ablation_cifar100", ablation_files)
        all_results.extend(ablation_results)
        print(f"  Analyzed {len(ablation_results)} layer-method combinations")
    else:
        print("  WARNING: No trajectory files found for ablation_cifar100")

    step += 1
    report_progress(TASK_ID, output_dir, step, total_steps,
                    metric={"total_layers_analyzed": len(all_results)})

    if not all_results:
        print("\nERROR: No data available for analysis!")
        mark_task_done(TASK_ID, output_dir, status="failed",
                       summary="No trajectory data found from dependent tasks")
        sys.exit(1)

    # --- Aggregate Analysis ---
    print("\n" + "=" * 70)
    print("AGGREGATE R² ANALYSIS")
    print("=" * 70)

    r2_values = [r['r_squared'] for r in all_results if np.isfinite(r['r_squared'])]
    r2_arr = np.array(r2_values)

    print(f"\nTotal layer-method combinations: {len(all_results)}")
    print(f"Valid R² values: {len(r2_values)}")
    print(f"R² statistics:")
    print(f"  Mean:   {np.mean(r2_arr):.4f}")
    print(f"  Median: {np.median(r2_arr):.4f}")
    print(f"  Std:    {np.std(r2_arr):.4f}")
    print(f"  Min:    {np.min(r2_arr):.4f}")
    print(f"  Max:    {np.max(r2_arr):.4f}")
    print(f"  Q25:    {np.percentile(r2_arr, 25):.4f}")
    print(f"  Q75:    {np.percentile(r2_arr, 75):.4f}")

    # Gate criterion: H7 falsified if R^2 > 0.85 in >= 70% of runs
    high_r2_count = np.sum(r2_arr > 0.85)
    high_r2_pct = high_r2_count / len(r2_arr) * 100
    gate_threshold_pct = 70.0

    print(f"\n--- H7 Gate Criterion ---")
    print(f"  R² > 0.85 count: {high_r2_count} / {len(r2_arr)} = {high_r2_pct:.1f}%")
    print(f"  Gate threshold: {gate_threshold_pct}%")

    gate_passes = bool(high_r2_pct < gate_threshold_pct)
    gate_decision = "PASS" if gate_passes else "FAIL"

    if gate_passes:
        print(f"  => GATE PASSES: Alignment carries non-trivial information beyond time proxy")
        print(f"     (Only {high_r2_pct:.1f}% of layers have R² > 0.85, < {gate_threshold_pct}% threshold)")
    else:
        print(f"  => GATE FAILS (H7 FALSIFIED): Alignment is largely a time proxy")
        print(f"     ({high_r2_pct:.1f}% of layers have R² > 0.85, >= {gate_threshold_pct}% threshold)")
        print(f"     Subsequent alignment comparisons should use RESIDUAL alignment")

    # --- Per-method breakdown ---
    print("\n--- Per-Method R² Breakdown ---")
    methods = sorted(set(r['method'] for r in all_results))
    method_stats = {}
    for method in methods:
        m_r2 = [r['r_squared'] for r in all_results
                if r['method'] == method and np.isfinite(r['r_squared'])]
        if m_r2:
            m_arr = np.array(m_r2)
            pct_high = np.sum(m_arr > 0.85) / len(m_arr) * 100
            method_stats[method] = {
                'mean_r2': float(np.mean(m_arr)),
                'median_r2': float(np.median(m_arr)),
                'std_r2': float(np.std(m_arr)),
                'pct_high_r2': float(pct_high),
                'n_layers': len(m_r2),
            }
            print(f"  {method:25s}: mean R²={np.mean(m_arr):.4f}, "
                  f"median={np.median(m_arr):.4f}, "
                  f"%>0.85={pct_high:.1f}%, "
                  f"n={len(m_r2)}")

    # --- Per-source breakdown ---
    print("\n--- Per-Source R² Breakdown ---")
    sources = sorted(set(r['source'] for r in all_results))
    source_stats = {}
    for source in sources:
        s_r2 = [r['r_squared'] for r in all_results
                if r['source'] == source and np.isfinite(r['r_squared'])]
        if s_r2:
            s_arr = np.array(s_r2)
            pct_high = np.sum(s_arr > 0.85) / len(s_arr) * 100
            source_stats[source] = {
                'mean_r2': float(np.mean(s_arr)),
                'median_r2': float(np.median(s_arr)),
                'pct_high_r2': float(pct_high),
                'n_layers': len(s_r2),
            }
            print(f"  {source:25s}: mean R²={np.mean(s_arr):.4f}, "
                  f"median={np.median(s_arr):.4f}, "
                  f"%>0.85={pct_high:.1f}%, "
                  f"n={len(s_r2)}")

    # --- Layer type analysis (conv vs bn vs bias) ---
    print("\n--- Layer Type R² Analysis ---")
    layer_type_stats = {}
    for ltype_name, ltype_filter in [
        ("conv/weight", lambda l: "conv" in l and "weight" in l),
        ("bn/weight", lambda l: "bn" in l and "weight" in l),
        ("bn/bias", lambda l: "bn" in l and "bias" in l),
        ("fc/weight", lambda l: "fc" in l or "classifier" in l or "linear" in l),
        ("other", lambda l: True),  # catch-all
    ]:
        lt_r2 = [r['r_squared'] for r in all_results
                 if ltype_filter(r['layer']) and np.isfinite(r['r_squared'])]
        if lt_r2:
            lt_arr = np.array(lt_r2)
            pct_high = np.sum(lt_arr > 0.85) / len(lt_arr) * 100
            layer_type_stats[ltype_name] = {
                'mean_r2': float(np.mean(lt_arr)),
                'pct_high_r2': float(pct_high),
                'n_layers': len(lt_r2),
            }
            print(f"  {ltype_name:15s}: mean R²={np.mean(lt_arr):.4f}, "
                  f"%>0.85={pct_high:.1f}%, n={len(lt_r2)}")

    # --- Find layers with lowest R² (most informative beyond time) ---
    print("\n--- Top 10 Layers with LOWEST R² (most non-temporal information) ---")
    sorted_by_r2 = sorted(all_results, key=lambda x: x['r_squared'] if np.isfinite(x['r_squared']) else 999)
    for i, r in enumerate(sorted_by_r2[:10]):
        print(f"  {i+1:2d}. {r['method']:20s} | {r['layer']:30s} | "
              f"R²={r['r_squared']:.4f} | alpha_std={r['alpha_std']:.6f}")

    # --- Find layers with highest R² (most time-like) ---
    print("\n--- Top 10 Layers with HIGHEST R² (most time-predictable) ---")
    sorted_by_r2_desc = sorted(
        [r for r in all_results if np.isfinite(r['r_squared'])],
        key=lambda x: -x['r_squared']
    )
    for i, r in enumerate(sorted_by_r2_desc[:10]):
        print(f"  {i+1:2d}. {r['method']:20s} | {r['layer']:30s} | "
              f"R²={r['r_squared']:.4f} | alpha_std={r['alpha_std']:.6f}")

    # --- Generate R² histogram data ---
    hist_counts, hist_edges = np.histogram(r2_arr, bins=20, range=(-0.5, 1.0))

    # --- Save results ---
    gate_result = {
        "task_id": TASK_ID,
        "mode": "pilot",
        "polynomial_degree": 4,
        "gate_decision": gate_decision,
        "gate_passes": gate_passes,
        "h7_falsified": bool(not gate_passes),
        "gate_threshold_pct": gate_threshold_pct,
        "r2_threshold": 0.85,
        "statistics": {
            "total_layer_method_combinations": len(all_results),
            "valid_r2_count": len(r2_values),
            "mean_r2": float(np.mean(r2_arr)),
            "median_r2": float(np.median(r2_arr)),
            "std_r2": float(np.std(r2_arr)),
            "min_r2": float(np.min(r2_arr)),
            "max_r2": float(np.max(r2_arr)),
            "q25_r2": float(np.percentile(r2_arr, 25)),
            "q75_r2": float(np.percentile(r2_arr, 75)),
            "pct_r2_above_085": float(high_r2_pct),
            "count_r2_above_085": int(int(high_r2_count)),
        },
        "per_method_stats": method_stats,
        "per_source_stats": source_stats,
        "layer_type_stats": layer_type_stats,
        "histogram": {
            "counts": hist_counts.tolist(),
            "bin_edges": hist_edges.tolist(),
        },
        "recommendation": (
            "Alignment signal carries information beyond time. "
            "Proceed with direct alignment comparisons."
            if gate_passes else
            "Alignment is largely time-predictable. "
            "Use residual alignment (alpha_t - poly_fit) for subsequent analyses."
        ),
        "sources_used": sources,
        "data_epochs": 10,
        "data_seed": 42,
        "timestamp": datetime.now().isoformat(),
    }

    # Save main result
    result_path = output_dir / "h7_gate_result.json"
    with open(result_path, 'w') as f:
        json.dump(gate_result, f, indent=2)
    print(f"\nSaved gate result to: {result_path}")

    # Save detailed per-layer results (for downstream analysis)
    detail_path = output_dir / "h7_per_layer_results.json"
    # Strip large coefficient arrays for readability
    detail_data = []
    for r in all_results:
        detail_entry = {k: v for k, v in r.items() if k != 'coefficients'}
        detail_data.append(detail_entry)
    with open(detail_path, 'w') as f:
        json.dump(detail_data, f, indent=2)
    print(f"Saved per-layer details to: {detail_path}")

    # --- Generate visualization: R² histogram ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Overall R² histogram
        ax = axes[0]
        ax.hist(r2_arr, bins=30, range=(-0.5, 1.0), color='steelblue',
                edgecolor='black', alpha=0.7)
        ax.axvline(x=0.85, color='red', linestyle='--', linewidth=2,
                   label=f'Gate threshold (R²=0.85)')
        ax.set_xlabel('R² (Degree-4 Polynomial Fit)', fontsize=12)
        ax.set_ylabel('Count (Layer-Method combinations)', fontsize=12)
        ax.set_title(f'H7 Temporal Predictability Gate\n'
                     f'Gate: {"PASS" if gate_passes else "FAIL"} '
                     f'({high_r2_pct:.1f}% > 0.85, threshold {gate_threshold_pct}%)',
                     fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot 2: Per-method R² distribution
        ax2 = axes[1]
        method_r2_lists = []
        method_labels = []
        for method in methods:
            m_r2 = [r['r_squared'] for r in all_results
                    if r['method'] == method and np.isfinite(r['r_squared'])]
            if m_r2:
                method_r2_lists.append(m_r2)
                method_labels.append(method)

        bp = ax2.boxplot(method_r2_lists, labels=method_labels, vert=True,
                         patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(method_labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax2.axhline(y=0.85, color='red', linestyle='--', linewidth=2,
                    label='Gate threshold (R²=0.85)')
        ax2.set_ylabel('R² (Degree-4 Polynomial Fit)', fontsize=12)
        ax2.set_title('R² Distribution by Method', fontsize=13)
        ax2.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=9)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig_path = output_dir / "h7_r2_distribution.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to: {fig_path}")

        # Plot 3: Per-source R² histogram
        fig2, axes2 = plt.subplots(1, len(sources), figsize=(7 * len(sources), 5))
        if len(sources) == 1:
            axes2 = [axes2]
        for ax, source in zip(axes2, sources):
            s_r2 = [r['r_squared'] for r in all_results
                    if r['source'] == source and np.isfinite(r['r_squared'])]
            ax.hist(s_r2, bins=25, range=(-0.5, 1.0), color='steelblue',
                    edgecolor='black', alpha=0.7)
            ax.axvline(x=0.85, color='red', linestyle='--', linewidth=2)
            pct = np.sum(np.array(s_r2) > 0.85) / len(s_r2) * 100 if s_r2 else 0
            ax.set_title(f'{source}\n({pct:.1f}% > 0.85)', fontsize=12)
            ax.set_xlabel('R²', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig2_path = output_dir / "h7_r2_by_source.png"
        plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved per-source visualization to: {fig2_path}")

    except ImportError as e:
        print(f"WARNING: Could not generate visualization (matplotlib not available): {e}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("H7 TEMPORAL PREDICTABILITY GATE — PILOT SUMMARY")
    print("=" * 70)
    print(f"  Data sources: {', '.join(sources)}")
    print(f"  Epochs per run: 10 (pilot)")
    print(f"  Total layer-method combos: {len(all_results)}")
    print(f"  Polynomial degree: 4")
    print(f"  Mean R²: {np.mean(r2_arr):.4f}")
    print(f"  Median R²: {np.median(r2_arr):.4f}")
    print(f"  % with R² > 0.85: {high_r2_pct:.1f}%")
    print(f"  Gate decision: {gate_decision}")
    if gate_passes:
        print(f"  => Alignment carries non-trivial temporal information.")
        print(f"     At least one layer shows R² < 0.85 (pilot pass criterion met).")
    else:
        print(f"  => Alignment is largely predictable from time alone.")
        print(f"     Use residual alignment for fair comparisons.")
    print("=" * 70)

    # Mark done
    mark_task_done(
        TASK_ID, output_dir,
        status="success",
        summary=(
            f"H7 gate {gate_decision}. "
            f"Mean R²={np.mean(r2_arr):.4f}, "
            f"{high_r2_pct:.1f}% > 0.85 "
            f"({'< ' if gate_passes else '>= '}{gate_threshold_pct}% threshold). "
            f"Analyzed {len(all_results)} layer-method combos from {len(sources)} sources."
        )
    )

    return gate_result


if __name__ == "__main__":
    result = main()
