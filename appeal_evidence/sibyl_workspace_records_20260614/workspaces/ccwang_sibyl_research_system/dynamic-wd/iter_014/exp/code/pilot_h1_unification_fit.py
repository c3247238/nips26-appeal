"""H1 Unification Fit: Pilot Task

For each of the 5 existing dynamic methods (CWD, SWD, CPR, DefazioCorrective, NoWD-as-degenerate),
using the lambda_t trajectories recorded in phase1_diagnostic pilot, find the UDWDC gain settings
(K_p, K_i, K_d) that minimize ||lambda_t^UDWDC - lambda_t^method||_2.

This directly tests H1 (unified control law subsumption).

Pilot pass criteria: Optimization converges for all 5 methods; at least 3 methods have
relative error < 15%.
"""

import json
import os
import sys
import numpy as np
from scipy.optimize import minimize, differential_evolution
from pathlib import Path
from datetime import datetime
import traceback

# Paths
WORKSPACE = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current")
DIAG_DIR = WORKSPACE / "exp" / "results" / "pilots" / "diagnostic_cifar10"
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots" / "h1_unification"
TASK_ID = "h1_unification_fit"

# PID file for system tracking
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PROGRESS_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"

# Methods to fit (excluding FixedWD which is the trivial K_p=K_i=K_d=0 case)
METHODS_TO_FIT = ["CWD", "SWD", "CPR", "DefazioCorrective", "NoWD"]

def write_pid():
    """Write PID file for system tracking."""
    PID_FILE.write_text(str(os.getpid()))

def report_progress(stage, total_stages, status="running", details=None):
    """Write progress file for system monitor."""
    progress = {
        "task_id": TASK_ID,
        "epoch": stage,
        "total_epochs": total_stages,
        "step": stage,
        "total_steps": total_stages,
        "loss": None,
        "metric": details or {},
        "updated_at": datetime.now().isoformat(),
        "status": status,
    }
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))

def mark_done(status="success", summary=""):
    """Write DONE marker file."""
    if PID_FILE.exists():
        PID_FILE.unlink()
    final_progress = {}
    if PROGRESS_FILE.exists():
        try:
            final_progress = json.loads(PROGRESS_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))


def load_trajectories(method, seed=42):
    """Load per-layer trajectory data for a given method."""
    path = DIAG_DIR / method / f"{method}_seed{seed}_trajectories.json"
    with open(path) as f:
        return json.load(f)


def load_epoch_data(method, seed=42):
    """Load epoch-level data (for LR schedule)."""
    path = DIAG_DIR / method / f"{method}_seed{seed}_epochs.json"
    with open(path) as f:
        return json.load(f)


def simulate_udwdc_lambda(rho_t_series, alpha_t_series, lr_series,
                          K_p, K_i, K_d, lambda_base, ema_beta=0.999,
                          lambda_min=0.0, lambda_max_factor=10.0):
    """Simulate what UDWDC would produce for given gains on observed rho_t/alpha_t.

    Args:
        rho_t_series: array of per-epoch rho_t values for this layer
        alpha_t_series: array of per-epoch alpha_t values for this layer
        lr_series: array of per-epoch learning rates
        K_p, K_i, K_d: control gains
        lambda_base: base WD coefficient
        ema_beta: EMA smoothing factor

    Returns:
        Array of effective lambda_t values that UDWDC would produce
    """
    n_epochs = len(rho_t_series)
    lambda_max = lambda_max_factor * lambda_base

    # Compute tau from initial conditions: tau = rho_0 / eta_0
    rho_0 = rho_t_series[0]
    eta_0 = lr_series[0]
    eps = 1e-8
    tau = rho_0 / (eta_0 + eps) if rho_0 > eps else lambda_base

    lambda_out = np.zeros(n_epochs)
    ema_error = 0.0
    ema_alpha = alpha_t_series[0] if len(alpha_t_series) > 0 else 0.0

    for t in range(n_epochs):
        # Target ratio: rho*(t) = eta_t / tau
        rho_target = lr_series[t] / (tau + eps)

        # Control error
        e_t = rho_t_series[t] - rho_target

        # EMA of alignment
        ema_alpha = ema_beta * ema_alpha + (1 - ema_beta) * alpha_t_series[t]

        # EMA of control error (integral term)
        ema_error = ema_beta * ema_error + (1 - ema_beta) * e_t

        # PID control law
        lambda_t = (lambda_base
                    + K_p * e_t
                    + K_i * ema_error
                    - K_d * ema_alpha * e_t)

        # Clipping
        lambda_t = max(lambda_min, min(lambda_max, lambda_t))
        lambda_out[t] = lambda_t

    return lambda_out


def compute_aggregate_lambda_trajectory(method, seed=42):
    """Compute the aggregate (mean across conv layers) effective_wd trajectory."""
    traj = load_trajectories(method, seed)
    epochs = load_epoch_data(method, seed)

    layers = traj['layers']
    lr_series = [e['lr'] for e in epochs['epochs']]

    # Collect per-layer trajectories (only conv/weight layers, not bias/bn)
    lambda_trajs = []
    rho_trajs = []
    alpha_trajs = []
    layer_names = []

    for name, layer_data in layers.items():
        # Skip bias and bn layers (they typically have effective_wd=0)
        if '.bias' in name or 'bn' in name or 'norm' in name:
            continue
        lambda_trajs.append(np.array(layer_data['effective_wd']))
        rho_trajs.append(np.array(layer_data['rho_t']))
        alpha_trajs.append(np.array(layer_data['alpha_t']))
        layer_names.append(name)

    return {
        'lambda_trajs': lambda_trajs,
        'rho_trajs': rho_trajs,
        'alpha_trajs': alpha_trajs,
        'lr_series': np.array(lr_series),
        'layer_names': layer_names,
        'n_epochs': len(lr_series),
    }


def objective_function(params, method_data, lambda_base=1e-4):
    """Objective: minimize sum of squared differences across all layers.

    Uses UDWDC simulation to predict lambda_t from observed rho_t/alpha_t,
    and compares against the method's actual effective_wd.
    """
    K_p, K_i, K_d = params
    total_error = 0.0
    total_norm = 0.0

    for i in range(len(method_data['lambda_trajs'])):
        target_lambda = method_data['lambda_trajs'][i]
        rho_t = method_data['rho_trajs'][i]
        alpha_t = method_data['alpha_trajs'][i]
        lr = method_data['lr_series']

        predicted = simulate_udwdc_lambda(
            rho_t, alpha_t, lr, K_p, K_i, K_d, lambda_base
        )

        diff = predicted - target_lambda
        total_error += np.sum(diff ** 2)
        total_norm += np.sum(target_lambda ** 2)

    # Avoid division by zero
    if total_norm < 1e-20:
        return total_error
    return total_error / (total_norm + 1e-20)


def fit_method(method, lambda_base=1e-4):
    """Find optimal K_p, K_i, K_d to approximate a method's lambda_t trajectory."""
    print(f"\n{'='*60}")
    print(f"Fitting UDWDC gains to approximate: {method}")
    print(f"{'='*60}")

    method_data = compute_aggregate_lambda_trajectory(method)
    n_layers = len(method_data['lambda_trajs'])
    n_epochs = method_data['n_epochs']

    print(f"  Layers: {n_layers} (conv/weight only)")
    print(f"  Epochs: {n_epochs}")

    # Show target lambda statistics
    all_lambdas = np.concatenate(method_data['lambda_trajs'])
    print(f"  Target lambda range: [{all_lambdas.min():.6f}, {all_lambdas.max():.6f}]")
    print(f"  Target lambda mean: {all_lambdas.mean():.6f}")

    # Handle NoWD case specially (all zeros)
    if method == "NoWD":
        # NoWD is the degenerate case: lambda_t = 0 for all t
        # UDWDC can approximate this with very large negative K_p to drive lambda below 0,
        # but clipping to 0. Let's check if a simple gain can produce near-zero outputs.
        pass

    # Bounds for optimization
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]

    # First try differential evolution (global optimizer) for robustness
    print("  Running differential evolution (global search)...")
    result_de = differential_evolution(
        objective_function,
        bounds=bounds,
        args=(method_data, lambda_base),
        seed=42,
        maxiter=200,
        tol=1e-10,
        polish=True,
    )

    # Also try multiple L-BFGS-B starts for comparison
    best_result = result_de
    best_fun = result_de.fun

    initial_guesses = [
        [0.0, 0.0, 0.0],           # FixedWD-like
        [0.5, 0.1, 0.3],           # Default UDWDC
        [0.0, 0.0, 0.5],           # CWD-like
        [0.5, 0.5, 0.0],           # PI-like (CPR)
        [1.0, 0.0, 0.0],           # P-only
        [-0.5, -0.5, 0.0],        # Negative gains (for reducing WD)
        result_de.x.tolist(),       # Start from DE solution
    ]

    for x0 in initial_guesses:
        try:
            result = minimize(
                objective_function,
                x0=x0,
                args=(method_data, lambda_base),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15},
            )
            if result.fun < best_fun:
                best_result = result
                best_fun = result.fun
        except Exception:
            continue

    K_p_opt, K_i_opt, K_d_opt = best_result.x

    # Compute relative error with proper handling
    total_error_sq = 0.0
    total_norm_sq = 0.0
    per_layer_errors = []

    for i in range(n_layers):
        target_lambda = method_data['lambda_trajs'][i]
        rho_t = method_data['rho_trajs'][i]
        alpha_t = method_data['alpha_trajs'][i]
        lr = method_data['lr_series']

        predicted = simulate_udwdc_lambda(
            rho_t, alpha_t, lr, K_p_opt, K_i_opt, K_d_opt, lambda_base
        )

        diff = predicted - target_lambda
        layer_error = np.sqrt(np.mean(diff ** 2))
        layer_target_rms = np.sqrt(np.mean(target_lambda ** 2))

        if layer_target_rms > 1e-10:
            layer_rel_error = layer_error / layer_target_rms
        else:
            # For NoWD, use absolute error
            layer_rel_error = layer_error

        per_layer_errors.append({
            'layer': method_data['layer_names'][i],
            'rmse': float(layer_error),
            'target_rms': float(layer_target_rms),
            'relative_error': float(layer_rel_error),
        })

        total_error_sq += np.sum(diff ** 2)
        total_norm_sq += np.sum(target_lambda ** 2)

    # Overall relative error
    if total_norm_sq > 1e-20:
        overall_rel_error = np.sqrt(total_error_sq / total_norm_sq)
    else:
        # NoWD case: check absolute error
        overall_rel_error = np.sqrt(total_error_sq / max(n_layers * n_epochs, 1))

    print(f"\n  Results for {method}:")
    print(f"    Optimal K_p = {K_p_opt:.6f}")
    print(f"    Optimal K_i = {K_i_opt:.6f}")
    print(f"    Optimal K_d = {K_d_opt:.6f}")
    print(f"    Relative Error = {overall_rel_error*100:.2f}%")
    print(f"    Optimizer success: {best_result.success}")
    print(f"    Optimizer message: {best_result.message if hasattr(best_result, 'message') else 'N/A'}")

    # Generate predicted vs actual trajectories for visualization
    pred_trajs = []
    for i in range(n_layers):
        predicted = simulate_udwdc_lambda(
            method_data['rho_trajs'][i],
            method_data['alpha_trajs'][i],
            method_data['lr_series'],
            K_p_opt, K_i_opt, K_d_opt, lambda_base
        )
        pred_trajs.append(predicted.tolist())

    return {
        'method': method,
        'K_p': float(K_p_opt),
        'K_i': float(K_i_opt),
        'K_d': float(K_d_opt),
        'relative_error_pct': float(overall_rel_error * 100),
        'optimizer_success': bool(best_result.success),
        'objective_value': float(best_fun),
        'per_layer_errors': per_layer_errors,
        'predicted_trajectories': pred_trajs,
        'actual_trajectories': [t.tolist() for t in method_data['lambda_trajs']],
        'layer_names': method_data['layer_names'],
        'n_epochs': n_epochs,
        'n_layers': n_layers,
    }


def generate_visualizations(all_results, method_data_cache):
    """Generate lambda_t trajectory comparison plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: lambda_t trajectories: original vs best-fit UDWDC (one subplot per method)
    n_methods = len(all_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, result in enumerate(all_results):
        ax = axes[idx]
        method = result['method']
        n_epochs = result['n_epochs']
        epochs = np.arange(n_epochs)

        # Plot mean across layers
        actual_mean = np.mean(result['actual_trajectories'], axis=0)
        predicted_mean = np.mean(result['predicted_trajectories'], axis=0)

        ax.plot(epochs, actual_mean, 'b-', linewidth=2, label=f'{method} (actual)')
        ax.plot(epochs, predicted_mean, 'r--', linewidth=2, label='UDWDC fit')

        # Add shaded region for min/max across layers
        actual_arr = np.array(result['actual_trajectories'])
        pred_arr = np.array(result['predicted_trajectories'])

        if actual_arr.shape[0] > 1:
            ax.fill_between(epochs, actual_arr.min(axis=0), actual_arr.max(axis=0),
                            alpha=0.15, color='blue')
            ax.fill_between(epochs, pred_arr.min(axis=0), pred_arr.max(axis=0),
                            alpha=0.15, color='red')

        ax.set_title(f"{method}\nK_p={result['K_p']:.3f}, K_i={result['K_i']:.3f}, "
                     f"K_d={result['K_d']:.3f}\nRel. Error: {result['relative_error_pct']:.1f}%",
                     fontsize=10)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Effective WD ($\\lambda_t$)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Remove unused subplot
    if n_methods < 6:
        for i in range(n_methods, 6):
            axes[i].set_visible(False)

    fig.suptitle('H1 Unification Test: UDWDC Gain Fitting to Approximate Each Method\'s $\\lambda_t$',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / "h1_lambda_trajectories.png", dpi=300, bbox_inches='tight')
    plt.savefig(fig_dir / "h1_lambda_trajectories.pdf", bbox_inches='tight')
    plt.close()
    print(f"\n  Saved trajectory comparison plot to {fig_dir / 'h1_lambda_trajectories.png'}")

    # Figure 2: Bar chart of relative errors
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = [r['method'] for r in all_results]
    errors = [r['relative_error_pct'] for r in all_results]
    colors = ['green' if e < 15 else ('orange' if e < 20 else 'red') for e in errors]

    bars = ax.bar(methods, errors, color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(y=15, color='green', linestyle='--', linewidth=1, label='15% threshold (pilot pass)')
    ax.axhline(y=20, color='red', linestyle='--', linewidth=1, label='20% threshold (H1 falsification)')

    for bar, error in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{error:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Relative Error (%)', fontsize=12)
    ax.set_title('H1 Unification: UDWDC Fitting Error per Method', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(fig_dir / "h1_relative_errors.png", dpi=300, bbox_inches='tight')
    plt.savefig(fig_dir / "h1_relative_errors.pdf", bbox_inches='tight')
    plt.close()
    print(f"  Saved relative error bar chart to {fig_dir / 'h1_relative_errors.png'}")


def main():
    write_pid()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("H1 Unification Fit — Pilot Task")
    print("Testing if UDWDC (K_p, K_i, K_d) can approximate each method's lambda_t")
    print("=" * 70)
    print(f"\nMethods to fit: {METHODS_TO_FIT}")
    print(f"Using diagnostic data from: {DIAG_DIR}")
    print(f"Results will be saved to: {RESULTS_DIR}")

    report_progress(0, len(METHODS_TO_FIT), "running", {"phase": "starting"})

    all_results = []
    method_data_cache = {}
    converged_count = 0
    low_error_count = 0

    for idx, method in enumerate(METHODS_TO_FIT):
        try:
            result = fit_method(method)
            all_results.append(result)

            if result['optimizer_success']:
                converged_count += 1
            if result['relative_error_pct'] < 15:
                low_error_count += 1

            report_progress(idx + 1, len(METHODS_TO_FIT), "running", {
                "method": method,
                "K_p": result['K_p'],
                "K_i": result['K_i'],
                "K_d": result['K_d'],
                "relative_error_pct": result['relative_error_pct'],
                "converged": result['optimizer_success'],
            })
        except Exception as e:
            print(f"\n  ERROR fitting {method}: {e}")
            traceback.print_exc()
            all_results.append({
                'method': method,
                'K_p': 0.0, 'K_i': 0.0, 'K_d': 0.0,
                'relative_error_pct': 100.0,
                'optimizer_success': False,
                'objective_value': float('inf'),
                'error': str(e),
                'per_layer_errors': [],
                'predicted_trajectories': [],
                'actual_trajectories': [],
                'layer_names': [],
                'n_epochs': 0,
                'n_layers': 0,
            })

    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    try:
        generate_visualizations(all_results, method_data_cache)
    except Exception as e:
        print(f"  Warning: Visualization generation failed: {e}")
        traceback.print_exc()

    # Build unifying table
    print("\n" + "=" * 70)
    print("UNIFYING TABLE: Control Law Parameter Mapping")
    print("=" * 70)
    print(f"{'Method':<20} {'K_p':>8} {'K_i':>8} {'K_d':>8} {'Rel.Err(%)':>12} {'Status':>10}")
    print("-" * 70)

    # Add FixedWD as trivial reference
    print(f"{'FixedWD':<20} {'0.000':>8} {'0.000':>8} {'0.000':>8} {'0.00':>12} {'TRIVIAL':>10}")

    for r in all_results:
        status = "PASS" if r['relative_error_pct'] < 15 else ("WARN" if r['relative_error_pct'] < 20 else "FAIL")
        print(f"{r['method']:<20} {r['K_p']:>8.3f} {r['K_i']:>8.3f} {r['K_d']:>8.3f} "
              f"{r['relative_error_pct']:>12.2f} {status:>10}")

    # Pilot pass/fail assessment
    print("\n" + "=" * 70)
    print("PILOT ASSESSMENT")
    print("=" * 70)
    print(f"  Converged: {converged_count}/{len(METHODS_TO_FIT)}")
    print(f"  Methods with relative error < 15%: {low_error_count}/{len(METHODS_TO_FIT)}")

    # Check pass criteria
    all_converged = converged_count == len(METHODS_TO_FIT)
    enough_low_error = low_error_count >= 3

    if all_converged and enough_low_error:
        verdict = "GO"
        print(f"\n  VERDICT: {verdict}")
        print(f"  All {len(METHODS_TO_FIT)} methods converged and {low_error_count} have < 15% error.")
    elif all_converged:
        verdict = "REFINE"
        print(f"\n  VERDICT: {verdict}")
        print(f"  All methods converged but only {low_error_count} have < 15% error.")
        print(f"  Consider: expanding search bounds, using more sophisticated objective, "
              f"or acknowledging fundamental approximation limits.")
    else:
        verdict = "NO_GO"
        print(f"\n  VERDICT: {verdict}")
        print(f"  {len(METHODS_TO_FIT) - converged_count} methods failed to converge.")

    # H1 falsification check
    high_error_count = sum(1 for r in all_results if r['relative_error_pct'] > 20)
    if high_error_count > 2:
        print(f"\n  WARNING: H1 falsification triggered — {high_error_count} methods have > 20% error.")
        print(f"  H1 claims all methods can be approximated within 20% relative error.")

    # Save fitting results JSON
    fitting_results = {
        'task_id': TASK_ID,
        'mode': 'pilot',
        'seed': 42,
        'data_source': 'diagnostic_cifar10 pilot (10 epochs)',
        'lambda_base': 1e-4,
        'methods_fitted': len(METHODS_TO_FIT),
        'converged_count': converged_count,
        'low_error_count': low_error_count,
        'verdict': verdict,
        'h1_falsified': high_error_count > 2,
        'results': all_results,
        'unifying_table': [
            {'method': 'FixedWD', 'K_p': 0.0, 'K_i': 0.0, 'K_d': 0.0,
             'relative_error_pct': 0.0, 'status': 'TRIVIAL'},
        ] + [
            {'method': r['method'], 'K_p': r['K_p'], 'K_i': r['K_i'], 'K_d': r['K_d'],
             'relative_error_pct': r['relative_error_pct'],
             'status': 'PASS' if r['relative_error_pct'] < 15 else ('WARN' if r['relative_error_pct'] < 20 else 'FAIL')}
            for r in all_results
        ],
        'timestamp': datetime.now().isoformat(),
    }

    results_file = RESULTS_DIR / "fitting_results.json"
    results_file.write_text(json.dumps(fitting_results, indent=2, default=str))
    print(f"\n  Saved fitting results to {results_file}")

    # Save unifying table as markdown
    table_md = RESULTS_DIR / "unifying_table.md"
    with open(table_md, 'w') as f:
        f.write("# H1 Unification: Control Law Parameter Mapping (Pilot)\n\n")
        f.write("| Method | K_p | K_i | K_d | Relative Error (%) | Status |\n")
        f.write("|--------|-----|-----|-----|--------------------:|--------|\n")
        f.write(f"| FixedWD | 0.000 | 0.000 | 0.000 | 0.00 | TRIVIAL |\n")
        for r in all_results:
            status = "PASS" if r['relative_error_pct'] < 15 else ("WARN" if r['relative_error_pct'] < 20 else "FAIL")
            f.write(f"| {r['method']} | {r['K_p']:.3f} | {r['K_i']:.3f} | {r['K_d']:.3f} | "
                    f"{r['relative_error_pct']:.2f} | {status} |\n")
        f.write(f"\n**Verdict**: {verdict}\n")
        f.write(f"- Converged: {converged_count}/{len(METHODS_TO_FIT)}\n")
        f.write(f"- Methods with < 15% error: {low_error_count}/{len(METHODS_TO_FIT)}\n")
        if high_error_count > 2:
            f.write(f"- **H1 falsification triggered**: {high_error_count} methods > 20% error\n")
    print(f"  Saved unifying table to {table_md}")

    # Mark done
    mark_done(
        status="success" if verdict != "NO_GO" else "partial",
        summary=f"H1 pilot {verdict}: {converged_count}/{len(METHODS_TO_FIT)} converged, "
                f"{low_error_count}/{len(METHODS_TO_FIT)} < 15% error"
    )

    print(f"\n{'='*70}")
    print(f"H1 Unification Fit pilot complete. Verdict: {verdict}")
    print(f"{'='*70}")

    return 0 if verdict != "NO_GO" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        mark_done(status="failed", summary=str(e))
        sys.exit(1)
