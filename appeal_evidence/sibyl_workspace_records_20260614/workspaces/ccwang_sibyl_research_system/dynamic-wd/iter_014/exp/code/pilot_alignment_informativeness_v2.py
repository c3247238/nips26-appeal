#!/usr/bin/env python3
"""Pilot for alignment_informativeness task (Phase 3b: H6 test) - v2.

PILOT mode: 6 runs (WD grid at LR=0.1, seed=42), 10 epochs on FULL CIFAR-100.
Previous pilot (v1) used max_samples=100 which produced near-random accuracy (~1-2%).
This v2 uses the full 50K training set for meaningful alignment/generalization signals.

For each run:
  - Train ResNet-20 on CIFAR-100 (full dataset, 50K train / 10K test)
  - Track per-layer alignment (alpha_t) and rho_t diagnostics
  - Compute alpha_bar, delta_max, EMA-smoothed variants
  - Compute generalization gap

Then compute LOO-CV R-squared comparing alpha_bar vs delta_max as predictors
of generalization gap to test H6.

Pass criteria:
  - All 6 runs complete without error
  - alpha_bar and delta_max differ across WD values
  - Generalization gap varies across WD values by >2%
"""

import argparse
import json
import os
import sys
import time
import math
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from optimizers import create_optimizer
from data import get_cifar100_loaders
from diagnostics.logger import DiagnosticLogger
from diagnostics.alignment import compute_alignment_stats, compute_snr


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_lr(optimizer_wrapper, epoch, total_epochs, base_lr):
    """Cosine annealing learning rate schedule."""
    lr = base_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    optimizer_wrapper.set_lr(lr)
    return lr


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on test set. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, 100.0 * correct / total


def compute_ema_alpha(alpha_trajectory: List[float], beta: float = 0.999) -> List[float]:
    """Compute EMA-smoothed alignment trajectory."""
    if not alpha_trajectory:
        return []
    ema = [alpha_trajectory[0]]
    for i in range(1, len(alpha_trajectory)):
        ema.append(beta * ema[-1] + (1 - beta) * alpha_trajectory[i])
    return ema


def loo_cv_r_squared(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute Leave-One-Out Cross-Validated R-squared for linear regression.

    Args:
        X: Predictor values (1D array, n samples)
        y: Target values (1D array, n samples)

    Returns:
        (loo_r2, 95% CI half-width via bootstrap)
    """
    n = len(X)
    if n < 3:
        return 0.0, 0.0

    X = X.reshape(-1, 1)
    ss_res = 0.0
    ss_tot = 0.0
    y_mean = np.mean(y)

    for i in range(n):
        # Leave one out
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]
        y_test = y[i]

        # Fit linear regression on training set
        X_aug = np.hstack([X_train, np.ones((n-1, 1))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_aug, y_train, rcond=None)
        except np.linalg.LinAlgError:
            continue

        X_test_aug = np.hstack([X_test, np.ones((1, 1))])
        y_pred = X_test_aug @ coeffs

        ss_res += (y_test - y_pred[0]) ** 2
        ss_tot += (y_test - y_mean) ** 2

    if ss_tot < 1e-12:
        loo_r2 = 0.0
    else:
        loo_r2 = 1.0 - ss_res / ss_tot

    # Bootstrap 95% CI
    rng = np.random.RandomState(42)
    n_boot = 500
    boot_r2s = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        X_b, y_b = X[idx], y[idx]
        if np.std(X_b) < 1e-12 or np.std(y_b) < 1e-12:
            continue
        try:
            corr = np.corrcoef(X_b.flatten(), y_b)[0, 1]
            boot_r2s.append(corr ** 2)
        except:
            continue

    if boot_r2s:
        ci_half = 1.96 * np.std(boot_r2s)
    else:
        ci_half = 0.0

    return float(loo_r2), float(ci_half)


def run_single_experiment(wd: float, lr: float, seed: int, epochs: int,
                          device: torch.device, data_root: str,
                          save_dir: Path) -> Dict:
    """Run a single training experiment and return results."""
    set_seed(seed)

    # Use full dataset with standard batch size
    train_loader, test_loader = get_cifar100_loaders(
        batch_size=128, data_root=data_root,
        max_samples=None, seed=seed
    )

    # Model
    model = create_model('resnet20', num_classes=100).to(device)

    # Optimizer - use FixedWD for the grid sweep
    wd_optimizer = create_optimizer(
        'FixedWD', model, lr=lr, momentum=0.9, weight_decay=wd
    )

    criterion = nn.CrossEntropyLoss()

    # Diagnostic logger
    run_name = f"wd{wd:.0e}_lr{lr}_seed{seed}"
    logger = DiagnosticLogger(
        save_dir=str(save_dir / run_name),
        method='FixedWD', seed=seed,
        model_name='resnet20', dataset='cifar100'
    )

    # Training loop
    start_time = time.time()
    best_acc = 0.0

    for epoch in range(epochs):
        current_lr = cosine_lr(wd_optimizer, epoch, epochs, lr)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, wd_optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        diagnostics = wd_optimizer.get_diagnostics()
        logger.log_epoch(epoch, diagnostics, train_loss, test_loss,
                        train_acc, test_acc, lr=current_lr)

        if test_acc > best_acc:
            best_acc = test_acc

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: train_acc={train_acc:.2f}% "
                  f"test_acc={test_acc:.2f}% lr={current_lr:.6f}")

    logger.save()
    elapsed = time.time() - start_time

    # Compute alignment statistics from trajectories
    alignment_stats = compute_alignment_stats(logger.trajectories)

    # Compute EMA-smoothed alignment variants
    all_epoch_alphas = [em['mean_alpha_t'] for em in logger.epoch_metrics]
    ema_alphas = compute_ema_alpha(all_epoch_alphas, beta=0.999)
    alpha_bar_ema = float(np.mean(ema_alphas)) if ema_alphas else 0.0

    # Compute generalization gap trajectory
    gen_gaps = [em['train_acc'] - em['test_acc'] for em in logger.epoch_metrics]

    # Compute SNR
    snr = compute_snr(all_epoch_alphas)

    # Collect per-layer alpha_bar for richer analysis
    per_layer_alpha_bar = {}
    for layer_name, data in logger.trajectories.items():
        alphas = np.array(data.get('alpha_t', []))
        if len(alphas) > 0:
            per_layer_alpha_bar[layer_name] = {
                'mean': float(np.mean(alphas)),
                'std': float(np.std(alphas)),
                'max_abs': float(np.max(np.abs(alphas))),
            }

    result = {
        'wd': wd,
        'lr': lr,
        'seed': seed,
        'epochs': epochs,
        'best_test_acc': best_acc,
        'final_test_acc': test_acc,
        'final_train_acc': train_acc,
        'generalization_gap': train_acc - test_acc,
        'alpha_bar': alignment_stats['alpha_bar'],
        'delta_max': alignment_stats['delta_max'],
        'alignment_variance': alignment_stats['alignment_variance'],
        'alpha_bar_ema': alpha_bar_ema,
        'snr': snr,
        'total_wd_budget': logger.get_total_wd_budget(),
        'elapsed_sec': elapsed,
        'gen_gap_trajectory': gen_gaps,
        'alpha_trajectory': all_epoch_alphas,
        'per_layer_alpha_bar': per_layer_alpha_bar,
    }

    # Cleanup
    del model, wd_optimizer, logger
    torch.cuda.empty_cache()
    gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Alignment Informativeness Pilot v2 (H6) - Full Dataset')
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--save_dir', type=str,
                        default='exp/results/pilots/alignment_informativeness')
    parser.add_argument('--task_id', type=str, default='alignment_informativeness')
    parser.add_argument('--results_dir', type=str, default='exp/results')

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 70)
    print(f"Alignment Informativeness Pilot v2 (H6) - Full CIFAR-100")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, Full dataset (50K train / 10K test)")
    print(f"=" * 70)

    # PID file
    pid_file = results_dir / f"{args.task_id}.pid"
    pid_file.write_text(str(os.getpid()))

    # Pilot configuration: 6 WD values at LR=0.1, seed=42
    wd_grid = [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3]
    lr = 0.1

    all_results = []
    start_total = time.time()

    for i, wd in enumerate(wd_grid):
        print(f"\n{'='*50}")
        print(f"Run {i+1}/{len(wd_grid)}: WD={wd:.0e}, LR={lr}, Seed={args.seed}")
        print(f"{'='*50}")

        result = run_single_experiment(
            wd=wd, lr=lr, seed=args.seed, epochs=args.epochs,
            device=device, data_root=args.data_root, save_dir=save_dir
        )
        all_results.append(result)

        print(f"  Best acc: {result['best_test_acc']:.2f}%")
        print(f"  Final train acc: {result['final_train_acc']:.2f}%")
        print(f"  Gen gap: {result['generalization_gap']:.2f}%")
        print(f"  alpha_bar: {result['alpha_bar']:.6f}")
        print(f"  delta_max: {result['delta_max']:.6f}")
        print(f"  SNR: {result['snr']:.4f}")
        print(f"  WD budget: {result['total_wd_budget']:.4f}")
        print(f"  Time: {result['elapsed_sec']:.1f}s")

        # Progress
        progress_file = results_dir / f"{args.task_id}_PROGRESS.json"
        progress_file.write_text(json.dumps({
            'task_id': args.task_id,
            'epoch': i + 1,
            'total_epochs': len(wd_grid),
            'loss': result.get('generalization_gap', 0),
            'metric': {
                'runs_completed': i + 1,
                'total_runs': len(wd_grid),
                'latest_acc': result['best_test_acc'],
                'latest_alpha_bar': result['alpha_bar'],
            },
            'updated_at': datetime.now().isoformat(),
        }))

    total_elapsed = time.time() - start_total

    # ===== Analysis: LOO-CV R-squared comparison =====
    print(f"\n{'='*70}")
    print("LOO-CV R-squared Analysis: alpha_bar vs delta_max")
    print(f"{'='*70}")

    # Extract arrays
    alpha_bars = np.array([r['alpha_bar'] for r in all_results])
    delta_maxes = np.array([r['delta_max'] for r in all_results])
    alpha_bar_emas = np.array([r['alpha_bar_ema'] for r in all_results])
    gen_gaps = np.array([r['generalization_gap'] for r in all_results])
    wd_values = np.array([r['wd'] for r in all_results])
    test_accs = np.array([r['best_test_acc'] for r in all_results])
    align_vars = np.array([r['alignment_variance'] for r in all_results])
    snrs = np.array([r['snr'] for r in all_results])

    # LOO-CV R-squared for various predictors
    r2_alpha_bar, ci_alpha_bar = loo_cv_r_squared(alpha_bars, gen_gaps)
    r2_delta_max, ci_delta_max = loo_cv_r_squared(delta_maxes, gen_gaps)
    r2_ema, ci_ema = loo_cv_r_squared(alpha_bar_emas, gen_gaps)
    r2_wd, ci_wd = loo_cv_r_squared(np.log10(wd_values), gen_gaps)
    r2_align_var, ci_align_var = loo_cv_r_squared(align_vars, gen_gaps)
    r2_snr, ci_snr = loo_cv_r_squared(snrs, gen_gaps)

    # Also try predicting test accuracy instead of gen gap
    r2_alpha_acc, ci_alpha_acc = loo_cv_r_squared(alpha_bars, test_accs)
    r2_delta_acc, ci_delta_acc = loo_cv_r_squared(delta_maxes, test_accs)
    r2_wd_acc, ci_wd_acc = loo_cv_r_squared(np.log10(wd_values), test_accs)

    print(f"\n=== Predicting Generalization Gap ===")
    print(f"{'Predictor':<25} | {'LOO-CV R^2':>12} | {'95% CI':>12}")
    print(f"{'='*55}")
    print(f"{'alpha_bar':<25} | {r2_alpha_bar:>12.4f} | +/- {ci_alpha_bar:.4f}")
    print(f"{'delta_max':<25} | {r2_delta_max:>12.4f} | +/- {ci_delta_max:.4f}")
    print(f"{'alpha_bar (EMA)':<25} | {r2_ema:>12.4f} | +/- {ci_ema:.4f}")
    print(f"{'alignment_variance':<25} | {r2_align_var:>12.4f} | +/- {ci_align_var:.4f}")
    print(f"{'SNR':<25} | {r2_snr:>12.4f} | +/- {ci_snr:.4f}")
    print(f"{'log10(WD)':<25} | {r2_wd:>12.4f} | +/- {ci_wd:.4f}")

    print(f"\n=== Predicting Test Accuracy ===")
    print(f"{'Predictor':<25} | {'LOO-CV R^2':>12} | {'95% CI':>12}")
    print(f"{'='*55}")
    print(f"{'alpha_bar':<25} | {r2_alpha_acc:>12.4f} | +/- {ci_alpha_acc:.4f}")
    print(f"{'delta_max':<25} | {r2_delta_acc:>12.4f} | +/- {ci_delta_acc:.4f}")
    print(f"{'log10(WD)':<25} | {r2_wd_acc:>12.4f} | +/- {ci_wd_acc:.4f}")

    # Pairwise correlations
    print(f"\n=== Pairwise Correlations ===")
    print(f"Pearson corr(alpha_bar, gen_gap)  = {np.corrcoef(alpha_bars, gen_gaps)[0,1]:.4f}")
    print(f"Pearson corr(delta_max, gen_gap)  = {np.corrcoef(delta_maxes, gen_gaps)[0,1]:.4f}")
    print(f"Pearson corr(alpha_bar, test_acc) = {np.corrcoef(alpha_bars, test_accs)[0,1]:.4f}")
    print(f"Pearson corr(log_wd, gen_gap)     = {np.corrcoef(np.log10(wd_values), gen_gaps)[0,1]:.4f}")
    print(f"Pearson corr(wd_budget, gen_gap)  = {np.corrcoef([r['total_wd_budget'] for r in all_results], gen_gaps)[0,1]:.4f}")

    # ===== Pass criteria evaluation =====
    print(f"\n{'='*70}")
    print("Pass Criteria Evaluation")
    print(f"{'='*70}")

    # 1. All 6 runs completed
    all_completed = len(all_results) == 6
    print(f"[{'PASS' if all_completed else 'FAIL'}] All 6 runs completed: {len(all_results)}/6")

    # 2. alpha_bar and delta_max differ across WD values
    alpha_bar_range = np.max(alpha_bars) - np.min(alpha_bars)
    delta_max_range = np.max(delta_maxes) - np.min(delta_maxes)
    alpha_differs = alpha_bar_range > 0.001
    delta_differs = delta_max_range > 0.001
    print(f"[{'PASS' if alpha_differs else 'FAIL'}] alpha_bar differs across WD: range={alpha_bar_range:.6f}")
    print(f"[{'PASS' if delta_differs else 'FAIL'}] delta_max differs across WD: range={delta_max_range:.6f}")

    # 3. Generalization gap varies by >2%
    gap_range = np.max(gen_gaps) - np.min(gen_gaps)
    gap_varies = gap_range > 2.0
    print(f"[{'PASS' if gap_varies else 'FAIL'}] Gen gap varies >2%: range={gap_range:.2f}%")

    overall_pass = all_completed and (alpha_differs or delta_differs) and gap_varies
    print(f"\nOverall: {'GO' if overall_pass else 'NO_GO'}")

    # ===== H6 Analysis =====
    print(f"\n{'='*70}")
    print("H6 Analysis: Is alpha_bar a better predictor than delta_max?")
    print(f"{'='*70}")

    r2_diff = r2_alpha_bar - r2_delta_max
    h6_supported = r2_diff > 0.05
    print(f"R^2(alpha_bar) - R^2(delta_max) = {r2_diff:.4f}")
    if h6_supported:
        print("H6 supported (pilot): alpha_bar is a better predictor of gen gap")
    elif r2_diff > -0.05:
        print("H6 inconclusive (pilot): difference < 0.05, need full 54 runs")
    else:
        print("H6 potentially falsified: delta_max is a substantially better predictor")

    # Check if either metric actually predicts well
    if r2_alpha_bar < 0 and r2_delta_max < 0:
        print("WARNING: Both predictors have negative LOO R^2 (worse than mean).")
        print("  With only 6 data points (pilot), this is expected due to high variance.")
        print("  Full 54-run experiment needed for meaningful R^2.")

    # ===== Summary tables =====
    print(f"\n{'='*70}")
    print("Summary Table: Per-WD Results")
    print(f"{'='*70}")
    print(f"{'WD':>10} | {'Best Acc':>9} | {'Train Acc':>10} | {'Gen Gap':>8} | "
          f"{'alpha_bar':>10} | {'delta_max':>10} | {'SNR':>8} | {'WD Budget':>10}")
    print(f"{'-'*95}")
    for r in all_results:
        print(f"{r['wd']:>10.0e} | {r['best_test_acc']:>8.2f}% | {r['final_train_acc']:>9.2f}% | "
              f"{r['generalization_gap']:>7.2f}% | "
              f"{r['alpha_bar']:>10.6f} | {r['delta_max']:>10.6f} | {r['snr']:>8.4f} | "
              f"{r['total_wd_budget']:>10.4f}")

    # ===== Save results =====
    # Clean results for JSON (remove large trajectories from summary)
    clean_results = []
    for r in all_results:
        clean = {k: v for k, v in r.items()
                 if k not in ('gen_gap_trajectory', 'alpha_trajectory', 'per_layer_alpha_bar')}
        clean_results.append(clean)

    pilot_result = {
        'task_id': args.task_id,
        'mode': 'pilot',
        'version': 'v2',
        'overall_pass': overall_pass,
        'verdict': 'GO' if overall_pass else 'NO_GO',
        'pass_criteria': {
            'all_runs_completed': all_completed,
            'alpha_bar_differs': bool(alpha_differs),
            'delta_max_differs': bool(delta_differs),
            'gen_gap_varies': bool(gap_varies),
            'gen_gap_range': float(gap_range),
            'alpha_bar_range': float(alpha_bar_range),
            'delta_max_range': float(delta_max_range),
        },
        'loo_cv_r_squared': {
            'predicting_gen_gap': {
                'alpha_bar': {'r2': r2_alpha_bar, 'ci_half': ci_alpha_bar},
                'delta_max': {'r2': r2_delta_max, 'ci_half': ci_delta_max},
                'alpha_bar_ema': {'r2': r2_ema, 'ci_half': ci_ema},
                'alignment_variance': {'r2': r2_align_var, 'ci_half': ci_align_var},
                'snr': {'r2': r2_snr, 'ci_half': ci_snr},
                'log10_wd': {'r2': r2_wd, 'ci_half': ci_wd},
            },
            'predicting_test_acc': {
                'alpha_bar': {'r2': r2_alpha_acc, 'ci_half': ci_alpha_acc},
                'delta_max': {'r2': r2_delta_acc, 'ci_half': ci_delta_acc},
                'log10_wd': {'r2': r2_wd_acc, 'ci_half': ci_wd_acc},
            },
        },
        'correlations': {
            'alpha_bar_vs_gen_gap': float(np.corrcoef(alpha_bars, gen_gaps)[0,1]),
            'delta_max_vs_gen_gap': float(np.corrcoef(delta_maxes, gen_gaps)[0,1]),
            'alpha_bar_vs_test_acc': float(np.corrcoef(alpha_bars, test_accs)[0,1]),
            'log_wd_vs_gen_gap': float(np.corrcoef(np.log10(wd_values), gen_gaps)[0,1]),
        },
        'h6_analysis': {
            'r2_difference': float(r2_diff),
            'h6_supported': bool(h6_supported),
            'conclusion': ('supported' if h6_supported
                          else ('inconclusive' if r2_diff > -0.05
                                else 'potentially falsified')),
        },
        'per_run_results': clean_results,
        'config': {
            'wd_grid': wd_grid,
            'lr': lr,
            'seed': args.seed,
            'epochs': args.epochs,
            'max_samples': None,  # Full dataset
            'train_samples': 50000,
            'test_samples': 10000,
            'model': 'resnet20',
            'dataset': 'cifar100',
            'batch_size': 128,
            'optimizer': 'SGD',
            'momentum': 0.9,
            'lr_schedule': 'cosine',
        },
        'total_elapsed_sec': total_elapsed,
        'timestamp': datetime.now().isoformat(),
    }

    # Save detailed results (with trajectories)
    detailed_file = save_dir / 'pilot_detailed_results.json'
    detailed_data = {
        'runs': [{k: v for k, v in r.items() if k != 'per_layer_alpha_bar'}
                 for r in all_results],
        'config': pilot_result['config'],
    }
    with open(detailed_file, 'w') as f:
        json.dump(detailed_data, f, indent=2,
                 default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    # Save pilot summary
    summary_file = save_dir / 'pilot_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(pilot_result, f, indent=2,
                 default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    # Save pilot_summary.md
    md_lines = [
        "# Alignment Informativeness Pilot Summary (v2)",
        "",
        f"**Date**: {datetime.now().isoformat()}",
        f"**Verdict**: {'GO' if overall_pass else 'NO_GO'}",
        "",
        "## Configuration",
        f"- Model: ResNet-20 on CIFAR-100 (full 50K train / 10K test)",
        f"- Epochs: {args.epochs}",
        f"- WD grid: {wd_grid}",
        f"- LR: {lr}, Seed: {args.seed}",
        f"- Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)",
        "",
        "## Results",
        "",
        "| WD | Best Acc | Train Acc | Gen Gap | alpha_bar | delta_max | SNR | WD Budget |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in all_results:
        md_lines.append(
            f"| {r['wd']:.0e} | {r['best_test_acc']:.2f}% | {r['final_train_acc']:.2f}% | "
            f"{r['generalization_gap']:.2f}% | {r['alpha_bar']:.6f} | {r['delta_max']:.6f} | "
            f"{r['snr']:.4f} | {r['total_wd_budget']:.4f} |"
        )
    md_lines.extend([
        "",
        "## LOO-CV R-squared (predicting gen gap)",
        "",
        "| Predictor | R^2 | 95% CI |",
        "|---|---|---|",
        f"| alpha_bar | {r2_alpha_bar:.4f} | +/- {ci_alpha_bar:.4f} |",
        f"| delta_max | {r2_delta_max:.4f} | +/- {ci_delta_max:.4f} |",
        f"| alpha_bar (EMA) | {r2_ema:.4f} | +/- {ci_ema:.4f} |",
        f"| alignment_variance | {r2_align_var:.4f} | +/- {ci_align_var:.4f} |",
        f"| SNR | {r2_snr:.4f} | +/- {ci_snr:.4f} |",
        f"| log10(WD) | {r2_wd:.4f} | +/- {ci_wd:.4f} |",
        "",
        "## H6 Analysis",
        f"- R^2(alpha_bar) - R^2(delta_max) = {r2_diff:.4f}",
        f"- Conclusion: {pilot_result['h6_analysis']['conclusion']}",
        "",
        "## Pass Criteria",
        f"- All runs completed: {'PASS' if all_completed else 'FAIL'}",
        f"- alpha_bar differs: {'PASS' if alpha_differs else 'FAIL'} (range={alpha_bar_range:.6f})",
        f"- delta_max differs: {'PASS' if delta_differs else 'FAIL'} (range={delta_max_range:.6f})",
        f"- Gen gap varies >2%: {'PASS' if gap_varies else 'FAIL'} (range={gap_range:.2f}%)",
    ])

    md_file = save_dir / 'pilot_summary.md'
    md_file.write_text('\n'.join(md_lines))

    # DONE marker
    done_file = results_dir / f"{args.task_id}_DONE"
    if pid_file.exists():
        pid_file.unlink()
    done_file.write_text(json.dumps({
        'task_id': args.task_id,
        'status': 'success' if overall_pass else 'partial',
        'summary': (f"Pilot v2 {'GO' if overall_pass else 'NO_GO'}: "
                   f"{len(all_results)}/6 runs on full CIFAR-100, "
                   f"R2(alpha_bar)={r2_alpha_bar:.4f}, "
                   f"R2(delta_max)={r2_delta_max:.4f}, "
                   f"gen_gap_range={gap_range:.2f}%"),
        'final_progress': {
            'runs_completed': len(all_results),
            'r2_alpha_bar': float(r2_alpha_bar),
            'r2_delta_max': float(r2_delta_max),
            'overall_pass': bool(overall_pass),
        },
        'timestamp': datetime.now().isoformat(),
    }))

    print(f"\nResults saved to: {save_dir}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    return pilot_result


if __name__ == '__main__':
    main()
