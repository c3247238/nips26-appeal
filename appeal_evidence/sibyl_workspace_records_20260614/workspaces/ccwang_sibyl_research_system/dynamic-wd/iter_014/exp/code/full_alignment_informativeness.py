#!/usr/bin/env python3
"""FULL alignment_informativeness experiment (Phase 3b: H6 test).

54-run grid sweep: 6 WDs x 3 LRs x 3 seeds on CIFAR-100, 30 epochs each.
Tests whether alpha_bar is a better predictor of generalization gap than delta_max.

Pilot findings (v3, 18 runs at seed=42):
  - ALL LOO-CV R^2 values were NEGATIVE (alpha_bar: -0.235, delta_max: -0.215)
  - Neither metric has meaningful linear predictive power for gen gap
  - H6 conclusion was "inconclusive" based on pilot alone
  - This FULL experiment with 3 seeds aims to confirm or reverse that finding

Grid:
  WD: {1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3}
  LR: {0.05, 0.1, 0.2}
  Seeds: {42, 123, 456}
  Total: 6 x 3 x 3 = 54 runs

Model: ResNet-20 on CIFAR-100 (full 50K train / 10K test)
Optimizer: SGD + momentum=0.9, cosine LR schedule
Epochs: 30 per run
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
from scipy import stats as scipy_stats

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
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]
        y_test = y[i]

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
    n_boot = 1000
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

    train_loader, test_loader = get_cifar100_loaders(
        batch_size=128, data_root=data_root,
        max_samples=None, seed=seed
    )

    model = create_model('resnet20', num_classes=100).to(device)

    wd_optimizer = create_optimizer(
        'FixedWD', model, lr=lr, momentum=0.9, weight_decay=wd
    )

    criterion = nn.CrossEntropyLoss()

    run_name = f"wd{wd:.0e}_lr{lr}_seed{seed}"
    logger = DiagnosticLogger(
        save_dir=str(save_dir / run_name),
        method='FixedWD', seed=seed,
        model_name='resnet20', dataset='cifar100'
    )

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

    logger.save()
    elapsed = time.time() - start_time

    alignment_stats = compute_alignment_stats(logger.trajectories)

    all_epoch_alphas = [em['mean_alpha_t'] for em in logger.epoch_metrics]
    ema_alphas = compute_ema_alpha(all_epoch_alphas, beta=0.999)
    alpha_bar_ema = float(np.mean(ema_alphas)) if ema_alphas else 0.0

    gen_gaps = [em['train_acc'] - em['test_acc'] for em in logger.epoch_metrics]
    snr = compute_snr(all_epoch_alphas)

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
    }

    del model, wd_optimizer, logger
    torch.cuda.empty_cache()
    gc.collect()

    return result


def compute_seed_averaged_results(all_results: List[Dict]) -> List[Dict]:
    """Average results across seeds for each (wd, lr) configuration."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        key = (r['wd'], r['lr'])
        grouped[key].append(r)

    averaged = []
    for (wd, lr), runs in sorted(grouped.items()):
        n = len(runs)
        avg = {
            'wd': wd,
            'lr': lr,
            'n_seeds': n,
            'seeds': [r['seed'] for r in runs],
        }
        # Average numeric fields
        for field in ['best_test_acc', 'final_test_acc', 'final_train_acc',
                      'generalization_gap', 'alpha_bar', 'delta_max',
                      'alignment_variance', 'alpha_bar_ema', 'snr',
                      'total_wd_budget', 'elapsed_sec']:
            vals = [r[field] for r in runs]
            avg[f'{field}_mean'] = float(np.mean(vals))
            avg[f'{field}_std'] = float(np.std(vals))
        averaged.append(avg)
    return averaged


def main():
    parser = argparse.ArgumentParser(
        description='FULL Alignment Informativeness (H6) - 54-run grid sweep')
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--save_dir', type=str,
                        default='exp/results/full/phase3_alignment/informativeness')
    parser.add_argument('--task_id', type=str, default='alignment_informativeness')
    parser.add_argument('--results_dir', type=str, default='exp/results')

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Grid configuration
    wd_grid = [1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3]
    lr_grid = [0.05, 0.1, 0.2]
    seed_grid = [42, 123, 456]
    total_runs = len(wd_grid) * len(lr_grid) * len(seed_grid)  # 54

    print(f"=" * 70)
    print(f"FULL Alignment Informativeness (H6) - 54-Run Grid Sweep")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, Full CIFAR-100 (50K train / 10K test)")
    print(f"Grid: {len(wd_grid)} WDs x {len(lr_grid)} LRs x {len(seed_grid)} seeds = {total_runs} runs")
    print(f"WD grid: {wd_grid}")
    print(f"LR grid: {lr_grid}")
    print(f"Seed grid: {seed_grid}")
    print(f"=" * 70)

    # PID file
    pid_file = results_dir / f"{args.task_id}.pid"
    pid_file.write_text(str(os.getpid()))

    all_results = []
    start_total = time.time()
    run_idx = 0

    for seed in seed_grid:
        for lr in lr_grid:
            for wd in wd_grid:
                run_idx += 1
                print(f"\n{'='*60}")
                print(f"Run {run_idx}/{total_runs}: WD={wd:.0e}, LR={lr}, Seed={seed}")
                print(f"{'='*60}")

                result = run_single_experiment(
                    wd=wd, lr=lr, seed=seed, epochs=args.epochs,
                    device=device, data_root=args.data_root, save_dir=save_dir
                )
                all_results.append(result)

                elapsed_so_far = time.time() - start_total
                avg_per_run = elapsed_so_far / run_idx
                eta = avg_per_run * (total_runs - run_idx)

                print(f"  Best acc: {result['best_test_acc']:.2f}%, "
                      f"Gen gap: {result['generalization_gap']:.2f}%, "
                      f"alpha_bar: {result['alpha_bar']:.6f}, "
                      f"Time: {result['elapsed_sec']:.1f}s, "
                      f"ETA: {eta/60:.0f}min")

                # Progress update
                progress_file = results_dir / f"{args.task_id}_PROGRESS.json"
                progress_file.write_text(json.dumps({
                    'task_id': args.task_id,
                    'epoch': run_idx,
                    'total_epochs': total_runs,
                    'loss': result.get('generalization_gap', 0),
                    'metric': {
                        'runs_completed': run_idx,
                        'total_runs': total_runs,
                        'latest_acc': result['best_test_acc'],
                        'latest_alpha_bar': result['alpha_bar'],
                        'current_seed': seed,
                        'elapsed_min': elapsed_so_far / 60,
                        'eta_min': eta / 60,
                    },
                    'updated_at': datetime.now().isoformat(),
                }))

    total_elapsed = time.time() - start_total

    # ===================================================================
    # COMPREHENSIVE ANALYSIS
    # ===================================================================

    # Extract arrays
    alpha_bars = np.array([r['alpha_bar'] for r in all_results])
    delta_maxes = np.array([r['delta_max'] for r in all_results])
    alpha_bar_emas = np.array([r['alpha_bar_ema'] for r in all_results])
    gen_gaps = np.array([r['generalization_gap'] for r in all_results])
    wd_values = np.array([r['wd'] for r in all_results])
    lr_values = np.array([r['lr'] for r in all_results])
    test_accs = np.array([r['best_test_acc'] for r in all_results])
    align_vars = np.array([r['alignment_variance'] for r in all_results])
    snrs = np.array([r['snr'] for r in all_results])
    wd_budgets = np.array([r['total_wd_budget'] for r in all_results])
    seeds = np.array([r['seed'] for r in all_results])

    print(f"\n{'='*70}")
    print("FULL ANALYSIS: LOO-CV R-squared (n=54)")
    print(f"{'='*70}")

    # === LOO-CV R-squared for all predictor candidates ===
    r2_alpha_bar, ci_alpha_bar = loo_cv_r_squared(alpha_bars, gen_gaps)
    r2_delta_max, ci_delta_max = loo_cv_r_squared(delta_maxes, gen_gaps)
    r2_ema, ci_ema = loo_cv_r_squared(alpha_bar_emas, gen_gaps)
    r2_wd, ci_wd = loo_cv_r_squared(np.log10(wd_values), gen_gaps)
    r2_lr, ci_lr = loo_cv_r_squared(lr_values, gen_gaps)
    r2_align_var, ci_align_var = loo_cv_r_squared(align_vars, gen_gaps)
    r2_snr, ci_snr = loo_cv_r_squared(snrs, gen_gaps)
    r2_budget, ci_budget = loo_cv_r_squared(wd_budgets, gen_gaps)

    # Predicting test accuracy
    r2_alpha_acc, ci_alpha_acc = loo_cv_r_squared(alpha_bars, test_accs)
    r2_delta_acc, ci_delta_acc = loo_cv_r_squared(delta_maxes, test_accs)
    r2_wd_acc, ci_wd_acc = loo_cv_r_squared(np.log10(wd_values), test_accs)
    r2_lr_acc, ci_lr_acc = loo_cv_r_squared(lr_values, test_accs)

    print(f"\n=== Predicting Generalization Gap (n={len(all_results)}) ===")
    print(f"{'Predictor':<25} | {'LOO-CV R^2':>12} | {'95% CI':>12}")
    print(f"{'='*55}")
    predictors_gap = [
        ("alpha_bar", r2_alpha_bar, ci_alpha_bar),
        ("delta_max", r2_delta_max, ci_delta_max),
        ("alpha_bar (EMA)", r2_ema, ci_ema),
        ("alignment_variance", r2_align_var, ci_align_var),
        ("SNR", r2_snr, ci_snr),
        ("log10(WD)", r2_wd, ci_wd),
        ("LR", r2_lr, ci_lr),
        ("WD_budget", r2_budget, ci_budget),
    ]
    for name, r2, ci in predictors_gap:
        marker = " ***" if r2 > 0 else ""
        print(f"{name:<25} | {r2:>12.4f} | +/- {ci:.4f}{marker}")

    print(f"\n=== Predicting Test Accuracy (n={len(all_results)}) ===")
    print(f"{'Predictor':<25} | {'LOO-CV R^2':>12} | {'95% CI':>12}")
    print(f"{'='*55}")
    predictors_acc = [
        ("alpha_bar", r2_alpha_acc, ci_alpha_acc),
        ("delta_max", r2_delta_acc, ci_delta_acc),
        ("log10(WD)", r2_wd_acc, ci_wd_acc),
        ("LR", r2_lr_acc, ci_lr_acc),
    ]
    for name, r2, ci in predictors_acc:
        marker = " ***" if r2 > 0 else ""
        print(f"{name:<25} | {r2:>12.4f} | +/- {ci:.4f}{marker}")

    # === Pearson and Spearman correlations ===
    print(f"\n=== Pairwise Correlations (n={len(all_results)}) ===")
    print(f"{'Pair':<35} | {'Pearson r':>10} | {'p-value':>10} | {'Spearman rho':>12} | {'p-value':>10}")
    print(f"{'='*85}")

    corr_pairs = [
        ("alpha_bar vs gen_gap", alpha_bars, gen_gaps),
        ("delta_max vs gen_gap", delta_maxes, gen_gaps),
        ("alpha_bar vs test_acc", alpha_bars, test_accs),
        ("delta_max vs test_acc", delta_maxes, test_accs),
        ("log10(WD) vs gen_gap", np.log10(wd_values), gen_gaps),
        ("LR vs gen_gap", lr_values, gen_gaps),
        ("WD_budget vs gen_gap", wd_budgets, gen_gaps),
        ("alpha_bar vs delta_max", alpha_bars, delta_maxes),
        ("SNR vs gen_gap", snrs, gen_gaps),
    ]

    correlation_results = {}
    for name, xv, yv in corr_pairs:
        pearson_r, pearson_p = scipy_stats.pearsonr(xv, yv)
        spearman_rho, spearman_p = scipy_stats.spearmanr(xv, yv)
        print(f"{name:<35} | {pearson_r:>10.4f} | {pearson_p:>10.4f} | {spearman_rho:>12.4f} | {spearman_p:>10.4f}")
        correlation_results[name] = {
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_rho': float(spearman_rho),
            'spearman_p': float(spearman_p),
        }

    # === Per-seed analysis ===
    print(f"\n=== Per-Seed LOO-CV R^2 (alpha_bar -> gen_gap) ===")
    per_seed_r2 = {}
    for s in seed_grid:
        mask = seeds == s
        r2_s, ci_s = loo_cv_r_squared(alpha_bars[mask], gen_gaps[mask])
        r2_d, ci_d = loo_cv_r_squared(delta_maxes[mask], gen_gaps[mask])
        per_seed_r2[s] = {
            'alpha_bar_r2': float(r2_s), 'alpha_bar_ci': float(ci_s),
            'delta_max_r2': float(r2_d), 'delta_max_ci': float(ci_d),
        }
        print(f"  Seed {s}: R2(alpha_bar)={r2_s:.4f}, R2(delta_max)={r2_d:.4f}, "
              f"diff={r2_s - r2_d:.4f}")

    # === Seed-averaged analysis (18 configurations, each with mean+std) ===
    seed_avg = compute_seed_averaged_results(all_results)
    avg_alpha_bars = np.array([r['alpha_bar_mean'] for r in seed_avg])
    avg_gen_gaps = np.array([r['generalization_gap_mean'] for r in seed_avg])
    avg_delta_maxes = np.array([r['delta_max_mean'] for r in seed_avg])

    r2_avg_alpha, ci_avg_alpha = loo_cv_r_squared(avg_alpha_bars, avg_gen_gaps)
    r2_avg_delta, ci_avg_delta = loo_cv_r_squared(avg_delta_maxes, avg_gen_gaps)

    print(f"\n=== Seed-Averaged LOO-CV R^2 (18 configurations) ===")
    print(f"  R2(mean alpha_bar) = {r2_avg_alpha:.4f} +/- {ci_avg_alpha:.4f}")
    print(f"  R2(mean delta_max) = {r2_avg_delta:.4f} +/- {ci_avg_delta:.4f}")

    # === H6 Formal Analysis ===
    print(f"\n{'='*70}")
    print("H6 FORMAL ANALYSIS")
    print(f"{'='*70}")
    print(f"Null hypothesis: alpha_bar is NOT a better predictor of gen gap than delta_max")
    print(f"Alternative: alpha_bar has higher LOO-CV R^2 than delta_max")
    print()

    r2_diff = r2_alpha_bar - r2_delta_max
    r2_diff_avg = r2_avg_alpha - r2_avg_delta

    # Determine H6 status
    all_r2_negative = (r2_alpha_bar < 0) and (r2_delta_max < 0)
    h6_supported = r2_diff > 0.05 and r2_alpha_bar > 0
    h6_falsified = r2_diff < -0.05

    print(f"Individual runs (n=54):")
    print(f"  R^2(alpha_bar)  = {r2_alpha_bar:.4f}")
    print(f"  R^2(delta_max)  = {r2_delta_max:.4f}")
    print(f"  Difference      = {r2_diff:.4f}")
    print()
    print(f"Seed-averaged (n=18):")
    print(f"  R^2(alpha_bar)  = {r2_avg_alpha:.4f}")
    print(f"  R^2(delta_max)  = {r2_avg_delta:.4f}")
    print(f"  Difference      = {r2_diff_avg:.4f}")
    print()

    if all_r2_negative:
        h6_conclusion = "falsified"
        h6_narrative = (
            "H6 FALSIFIED: Neither alpha_bar nor delta_max has meaningful linear "
            "predictive power for generalization gap (all LOO-CV R^2 negative). "
            "The alignment signal as currently defined does not capture the "
            "factors driving generalization in a simple linear relationship. "
            "This is a genuine negative result that should be reported honestly. "
            "Possible explanations: (1) the relationship is nonlinear, "
            "(2) alignment interacts with other hyperparameters in complex ways, "
            "(3) alignment metrics need redefinition, or "
            "(4) generalization gap is driven by factors orthogonal to alignment."
        )
    elif h6_supported:
        h6_conclusion = "supported"
        h6_narrative = (
            "H6 SUPPORTED: alpha_bar has meaningfully higher LOO-CV R^2 than delta_max "
            f"(difference = {r2_diff:.4f}), confirming that mean alignment is a better "
            "predictor of generalization gap than peak alignment deviation."
        )
    elif h6_falsified:
        h6_conclusion = "falsified"
        h6_narrative = (
            "H6 FALSIFIED: delta_max is a substantially better predictor than alpha_bar "
            f"(R^2 difference = {r2_diff:.4f}), contrary to the hypothesis."
        )
    else:
        h6_conclusion = "inconclusive"
        h6_narrative = (
            f"H6 INCONCLUSIVE: R^2 difference ({r2_diff:.4f}) is within the noise band. "
            "Neither metric clearly dominates."
        )

    print(f"CONCLUSION: {h6_conclusion.upper()}")
    print(f"\n{h6_narrative}")

    # Linear slope analysis
    slope_alpha = np.polyfit(alpha_bars, gen_gaps, 1)[0]
    slope_delta = np.polyfit(delta_maxes, gen_gaps, 1)[0]
    print(f"\nLinear slopes:")
    print(f"  alpha_bar -> gen_gap: slope={slope_alpha:.4f}")
    print(f"  delta_max -> gen_gap: slope={slope_delta:.4f}")

    # === Pass criteria ===
    print(f"\n{'='*70}")
    print("Pass Criteria Evaluation")
    print(f"{'='*70}")

    all_completed = len(all_results) == total_runs
    alpha_bar_range = np.max(alpha_bars) - np.min(alpha_bars)
    delta_max_range = np.max(delta_maxes) - np.min(delta_maxes)
    alpha_differs = alpha_bar_range > 0.001
    delta_differs = delta_max_range > 0.001
    gap_range = np.max(gen_gaps) - np.min(gen_gaps)
    gap_varies = gap_range > 2.0

    print(f"[{'PASS' if all_completed else 'FAIL'}] All {total_runs} runs completed: {len(all_results)}/{total_runs}")
    print(f"[{'PASS' if alpha_differs else 'FAIL'}] alpha_bar differs: range={alpha_bar_range:.6f}")
    print(f"[{'PASS' if delta_differs else 'FAIL'}] delta_max differs: range={delta_max_range:.6f}")
    print(f"[{'PASS' if gap_varies else 'FAIL'}] Gen gap varies >2%: range={gap_range:.2f}%")

    overall_pass = all_completed and (alpha_differs or delta_differs) and gap_varies

    # === Descriptive Statistics ===
    print(f"\n{'='*70}")
    print("Descriptive Statistics")
    print(f"{'='*70}")
    for name, arr in [("gen_gap", gen_gaps), ("alpha_bar", alpha_bars),
                      ("delta_max", delta_maxes), ("test_acc", test_accs),
                      ("snr", snrs), ("wd_budget", wd_budgets)]:
        print(f"  {name:>15}: mean={np.mean(arr):.4f}, std={np.std(arr):.4f}, "
              f"min={np.min(arr):.4f}, max={np.max(arr):.4f}, "
              f"range={np.max(arr)-np.min(arr):.4f}")

    # === Summary Table (seed-averaged) ===
    print(f"\n{'='*70}")
    print("Seed-Averaged Results (mean +/- std across 3 seeds)")
    print(f"{'='*70}")
    print(f"{'LR':>6} {'WD':>8} | {'BestAcc':>14} | {'GenGap':>14} | "
          f"{'alpha_bar':>16} | {'delta_max':>16}")
    print(f"{'-'*85}")
    for r in seed_avg:
        print(f"{r['wd']:>6.0e}  {r['lr']:>6.2f} | "
              f"{r['best_test_acc_mean']:>6.2f}+/-{r['best_test_acc_std']:.2f} | "
              f"{r['generalization_gap_mean']:>6.2f}+/-{r['generalization_gap_std']:.2f} | "
              f"{r['alpha_bar_mean']:>7.6f}+/-{r['alpha_bar_std']:.4f} | "
              f"{r['delta_max_mean']:>7.4f}+/-{r['delta_max_std']:.4f}")

    # ===================================================================
    # SAVE RESULTS
    # ===================================================================

    # Clean results (remove large trajectories for JSON)
    clean_results = []
    for r in all_results:
        clean = {k: v for k, v in r.items()
                 if k not in ('gen_gap_trajectory', 'alpha_trajectory')}
        clean_results.append(clean)

    full_result = {
        'task_id': args.task_id,
        'mode': 'full',
        'overall_pass': overall_pass,
        'n_runs': len(all_results),
        'n_configs': len(seed_avg),
        'h6_analysis': {
            'conclusion': h6_conclusion,
            'narrative': h6_narrative,
            'individual_runs': {
                'n': len(all_results),
                'r2_alpha_bar': float(r2_alpha_bar),
                'r2_delta_max': float(r2_delta_max),
                'r2_difference': float(r2_diff),
                'all_r2_negative': bool(all_r2_negative),
            },
            'seed_averaged': {
                'n': len(seed_avg),
                'r2_alpha_bar': float(r2_avg_alpha),
                'r2_delta_max': float(r2_avg_delta),
                'r2_difference': float(r2_diff_avg),
            },
            'per_seed': per_seed_r2,
            'linear_slopes': {
                'alpha_bar_to_gen_gap': float(slope_alpha),
                'delta_max_to_gen_gap': float(slope_delta),
            },
        },
        'loo_cv_r_squared': {
            'predicting_gen_gap': {
                'alpha_bar': {'r2': r2_alpha_bar, 'ci_half': ci_alpha_bar},
                'delta_max': {'r2': r2_delta_max, 'ci_half': ci_delta_max},
                'alpha_bar_ema': {'r2': r2_ema, 'ci_half': ci_ema},
                'alignment_variance': {'r2': r2_align_var, 'ci_half': ci_align_var},
                'snr': {'r2': r2_snr, 'ci_half': ci_snr},
                'log10_wd': {'r2': r2_wd, 'ci_half': ci_wd},
                'lr': {'r2': r2_lr, 'ci_half': ci_lr},
                'wd_budget': {'r2': r2_budget, 'ci_half': ci_budget},
            },
            'predicting_test_acc': {
                'alpha_bar': {'r2': r2_alpha_acc, 'ci_half': ci_alpha_acc},
                'delta_max': {'r2': r2_delta_acc, 'ci_half': ci_delta_acc},
                'log10_wd': {'r2': r2_wd_acc, 'ci_half': ci_wd_acc},
                'lr': {'r2': r2_lr_acc, 'ci_half': ci_lr_acc},
            },
        },
        'correlations': correlation_results,
        'pass_criteria': {
            'all_runs_completed': all_completed,
            'total_runs': total_runs,
            'alpha_bar_differs': bool(alpha_differs),
            'delta_max_differs': bool(delta_differs),
            'gen_gap_varies': bool(gap_varies),
            'gen_gap_range': float(gap_range),
            'alpha_bar_range': float(alpha_bar_range),
            'delta_max_range': float(delta_max_range),
        },
        'descriptive_stats': {
            'gen_gap': {'mean': float(np.mean(gen_gaps)), 'std': float(np.std(gen_gaps)),
                       'min': float(np.min(gen_gaps)), 'max': float(np.max(gen_gaps))},
            'alpha_bar': {'mean': float(np.mean(alpha_bars)), 'std': float(np.std(alpha_bars)),
                         'min': float(np.min(alpha_bars)), 'max': float(np.max(alpha_bars))},
            'delta_max': {'mean': float(np.mean(delta_maxes)), 'std': float(np.std(delta_maxes)),
                         'min': float(np.min(delta_maxes)), 'max': float(np.max(delta_maxes))},
            'test_acc': {'mean': float(np.mean(test_accs)), 'std': float(np.std(test_accs)),
                        'min': float(np.min(test_accs)), 'max': float(np.max(test_accs))},
        },
        'seed_averaged_results': seed_avg,
        'per_run_results': clean_results,
        'config': {
            'wd_grid': wd_grid,
            'lr_grid': lr_grid,
            'seed_grid': seed_grid,
            'epochs': args.epochs,
            'model': 'resnet20',
            'dataset': 'cifar100',
            'batch_size': 128,
            'optimizer': 'SGD',
            'momentum': 0.9,
            'lr_schedule': 'cosine',
            'train_samples': 50000,
            'test_samples': 10000,
        },
        'total_elapsed_sec': total_elapsed,
        'timestamp': datetime.now().isoformat(),
    }

    # Save detailed results (including trajectories)
    detailed_file = save_dir / 'full_detailed_results.json'
    detailed_data = {
        'runs': all_results,
        'config': full_result['config'],
    }
    with open(detailed_file, 'w') as f:
        json.dump(detailed_data, f, indent=2,
                 default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    # Save full summary
    summary_file = save_dir / 'informativeness_results.json'
    with open(summary_file, 'w') as f:
        json.dump(full_result, f, indent=2,
                 default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    # Also save to the expected output path
    expected_output = Path('exp/results/full/phase3_alignment/informativeness_results.json')
    expected_output.parent.mkdir(parents=True, exist_ok=True)
    with open(expected_output, 'w') as f:
        json.dump(full_result, f, indent=2,
                 default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    # Save markdown summary
    md_lines = [
        "# Alignment Informativeness FULL Results (H6 Test)",
        "",
        f"**Date**: {datetime.now().isoformat()}",
        f"**H6 Conclusion**: {h6_conclusion.upper()}",
        f"**Total Runs**: {len(all_results)}/{total_runs}",
        f"**Total Time**: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)",
        "",
        "## H6 Summary",
        "",
        h6_narrative,
        "",
        "## Configuration",
        f"- Model: ResNet-20 on CIFAR-100 (full 50K/10K)",
        f"- Epochs: {args.epochs}",
        f"- WD grid: {wd_grid}",
        f"- LR grid: {lr_grid}",
        f"- Seeds: {seed_grid}",
        "",
        "## LOO-CV R-squared: Predicting Generalization Gap",
        "",
        "| Predictor | R^2 | 95% CI |",
        "|---|---|---|",
    ]
    for name, r2, ci in predictors_gap:
        md_lines.append(f"| {name} | {r2:.4f} | +/- {ci:.4f} |")

    md_lines.extend([
        "",
        "## LOO-CV R-squared: Predicting Test Accuracy",
        "",
        "| Predictor | R^2 | 95% CI |",
        "|---|---|---|",
    ])
    for name, r2, ci in predictors_acc:
        md_lines.append(f"| {name} | {r2:.4f} | +/- {ci:.4f} |")

    md_lines.extend([
        "",
        "## Per-Seed Analysis",
        "",
        "| Seed | R^2(alpha_bar) | R^2(delta_max) | Difference |",
        "|---|---|---|---|",
    ])
    for s in seed_grid:
        d = per_seed_r2[s]
        diff = d['alpha_bar_r2'] - d['delta_max_r2']
        md_lines.append(f"| {s} | {d['alpha_bar_r2']:.4f} | {d['delta_max_r2']:.4f} | {diff:.4f} |")

    md_lines.extend([
        "",
        "## Key Correlations",
        "",
        "| Pair | Pearson r | p-value | Spearman rho | p-value |",
        "|---|---|---|---|---|",
    ])
    for name, data in correlation_results.items():
        md_lines.append(
            f"| {name} | {data['pearson_r']:.4f} | {data['pearson_p']:.4f} | "
            f"{data['spearman_rho']:.4f} | {data['spearman_p']:.4f} |"
        )

    md_lines.extend([
        "",
        "## Seed-Averaged Results",
        "",
        "| WD | LR | Test Acc (mean+/-std) | Gen Gap (mean+/-std) | alpha_bar (mean+/-std) |",
        "|---|---|---|---|---|",
    ])
    for r in seed_avg:
        md_lines.append(
            f"| {r['wd']:.0e} | {r['lr']} | "
            f"{r['best_test_acc_mean']:.2f}+/-{r['best_test_acc_std']:.2f} | "
            f"{r['generalization_gap_mean']:.2f}+/-{r['generalization_gap_std']:.2f} | "
            f"{r['alpha_bar_mean']:.6f}+/-{r['alpha_bar_std']:.4f} |"
        )

    md_file = save_dir / 'full_summary.md'
    md_file.write_text('\n'.join(md_lines))

    # DONE marker
    done_file = results_dir / f"{args.task_id}_DONE"
    if pid_file.exists():
        pid_file.unlink()
    done_file.write_text(json.dumps({
        'task_id': args.task_id,
        'status': 'success',
        'summary': (f"FULL H6 {h6_conclusion.upper()}: "
                   f"{len(all_results)}/{total_runs} runs, "
                   f"R2(alpha_bar)={r2_alpha_bar:.4f}, "
                   f"R2(delta_max)={r2_delta_max:.4f}, "
                   f"gen_gap_range={gap_range:.2f}%"),
        'final_progress': {
            'runs_completed': len(all_results),
            'total_runs': total_runs,
            'r2_alpha_bar': float(r2_alpha_bar),
            'r2_delta_max': float(r2_delta_max),
            'h6_conclusion': h6_conclusion,
            'gen_gap_range': float(gap_range),
            'overall_pass': bool(overall_pass),
        },
        'timestamp': datetime.now().isoformat(),
    }))

    print(f"\n{'='*70}")
    print(f"Results saved to: {save_dir}")
    print(f"Main output: {expected_output}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"{'='*70}")

    return full_result


if __name__ == '__main__':
    main()
