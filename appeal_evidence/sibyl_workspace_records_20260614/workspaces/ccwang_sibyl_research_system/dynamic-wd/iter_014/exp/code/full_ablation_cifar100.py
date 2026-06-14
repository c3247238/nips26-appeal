#!/usr/bin/env python3
"""Full: Phase 2a Ablation -- UDWDC-v2 gain configurations on CIFAR-100/VGG-16-BN.

200 epochs, 3 seeds (42, 123, 456), 8 variants.

Variants:
1. FixedWD baseline (constant lambda)
2. K_p-only (proportional control)
3. K_i-only (integral control)
4. K_d-only (alignment-derivative control)
5. PI control (K_p + K_i, CPR-like)
6. PD control (K_p + K_d, CWD-like)
7. Full PID (K_p + K_i + K_d)
8. UDWDC-v2 default (K_p=0.5, K_i=0.1, K_d=0.3 with EMA + floor)

All UDWDC variants use v2 with floor clipping (0.1 * lambda_base).
"""

import json
import os
import sys
import time
import math
import traceback
import gc
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from optimizers import create_optimizer
from data import get_cifar100_loaders
from diagnostics.logger import DiagnosticLogger
from diagnostics.metrics import compute_csi


TASK_ID = "ablation_cifar100"
TOTAL_EPOCHS = 200
SEEDS = [42, 123, 456]
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WD = 1e-4
MODEL = "vgg16_bn"
DATASET = "cifar100"
NUM_CLASSES = 100

# Workspace paths
WORKSPACE = '/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current'
SAVE_DIR = f'{WORKSPACE}/exp/results/full/phase2_ablation'
RESULTS_BASE = f'{WORKSPACE}/exp/results'

# UDWDC default gains
DEFAULT_KP = 0.5
DEFAULT_KI = 0.1
DEFAULT_KD = 0.3

# Ablation variants: (name, K_p, K_i, K_d, description)
VARIANTS = [
    ("FixedWD",      0.0,        0.0,        0.0,        "Fixed WD baseline", True),
    ("Kp_only",      DEFAULT_KP, 0.0,        0.0,        "Proportional only", False),
    ("Ki_only",      0.0,        DEFAULT_KI, 0.0,        "Integral only", False),
    ("Kd_only",      0.0,        0.0,        DEFAULT_KD, "Derivative/alignment only", False),
    ("PI_control",   DEFAULT_KP, DEFAULT_KI, 0.0,        "PI (CPR-like)", False),
    ("PD_control",   DEFAULT_KP, 0.0,        DEFAULT_KD, "PD (CWD-like)", False),
    ("Full_PID",     DEFAULT_KP, DEFAULT_KI, DEFAULT_KD, "Full PID", False),
    ("UDWDC_v2",     DEFAULT_KP, DEFAULT_KI, DEFAULT_KD, "UDWDC-v2 default", False),
]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_lr(wd_optimizer, epoch, total_epochs, base_lr):
    """Cosine annealing LR schedule."""
    lr = base_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    wd_optimizer.set_lr(lr)
    return lr


def train_one_epoch(model, loader, optimizer, criterion, device):
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


def check_lambda_health(diagnostics):
    """Check for NaN/Inf in effective WD values."""
    for layer_name, metrics in diagnostics.items():
        wd = metrics.get('effective_wd', 0.0)
        if math.isnan(wd) or math.isinf(wd):
            return False, f"NaN/Inf lambda at layer {layer_name}: {wd}"
    return True, "OK"


def report_progress(completed, total, current_variant="", current_seed=0,
                    epoch=0, test_acc=0.0, best_acc=0.0):
    """Write progress file for system monitor."""
    progress_file = Path(RESULTS_BASE) / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        'task_id': TASK_ID,
        'epoch': completed,
        'total_epochs': total,
        'step': completed,
        'total_steps': total,
        'loss': 0.0,
        'metric': {
            'current_variant': current_variant,
            'current_seed': current_seed,
            'current_epoch': epoch,
            'test_acc': test_acc,
            'best_acc': best_acc,
        },
        'updated_at': datetime.now().isoformat(),
    }))


def run_single_variant(variant_name, K_p, K_i, K_d, is_baseline, seed,
                        device, train_loader, test_loader, variant_save_dir):
    """Run a single ablation variant for one seed."""
    set_seed(seed)

    model = create_model(MODEL, num_classes=NUM_CLASSES).to(device)

    if is_baseline:
        wd_optimizer = create_optimizer(
            'FixedWD', model, lr=LR, momentum=MOMENTUM, weight_decay=WD
        )
    else:
        wd_optimizer = create_optimizer(
            'UDWDC-v2', model, lr=LR, momentum=MOMENTUM, weight_decay=WD,
            K_p=K_p, K_i=K_i, K_d=K_d
        )

    criterion = nn.CrossEntropyLoss()
    method_label = "FixedWD" if is_baseline else f"UDWDC_v2_{variant_name}"
    logger = DiagnosticLogger(
        save_dir=variant_save_dir, method=method_label, seed=seed,
        model_name=MODEL, dataset=DATASET
    )

    start_time = time.time()
    best_acc = 0.0
    nan_inf_detected = False
    wd_budget_per_epoch = []
    effective_lr_trajectory = []  # For CSI computation

    for epoch in range(TOTAL_EPOCHS):
        lr = cosine_lr(wd_optimizer, epoch, TOTAL_EPOCHS, LR)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, wd_optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        diagnostics = wd_optimizer.get_diagnostics()
        logger.log_epoch(epoch, diagnostics, train_loss, test_loss, train_acc, test_acc, lr=lr)

        # Check for NaN/Inf
        healthy, detail = check_lambda_health(diagnostics)
        if not healthy:
            nan_inf_detected = True

        # Compute effective WD stats
        wd_values = [m['effective_wd'] for m in diagnostics.values()]
        mean_wd = np.mean(wd_values)
        epoch_wd_budget = float(np.sum(wd_values))
        wd_budget_per_epoch.append(epoch_wd_budget)

        # Effective LR for CSI: lr * (1 - mean_wd * lr)
        effective_lr = lr * (1 - mean_wd * lr)
        effective_lr_trajectory.append(effective_lr)

        if test_acc > best_acc:
            best_acc = test_acc

        # Call end_epoch_check for UDWDC-v2
        if hasattr(wd_optimizer, 'end_epoch_check'):
            wd_optimizer.end_epoch_check()

        # Print progress every 20 epochs and at start/end
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch == TOTAL_EPOCHS - 1:
            elapsed = time.time() - start_time
            eta_min = elapsed / (epoch + 1) * (TOTAL_EPOCHS - epoch - 1) / 60
            print(f"  [{method_label}|s{seed}] Epoch {epoch+1}/{TOTAL_EPOCHS} | "
                  f"Train: {train_acc:.1f}% Test: {test_acc:.1f}% Best: {best_acc:.1f}% | "
                  f"Loss: {train_loss:.4f} WD: {mean_wd:.6f} LR: {lr:.6f} | "
                  f"Time: {elapsed:.0f}s ETA: {eta_min:.1f}min")

    logger.save()
    elapsed = time.time() - start_time

    final_test_acc = test_acc
    final_train_acc = train_acc
    gen_gap = final_train_acc - final_test_acc
    total_wd_budget = logger.get_total_wd_budget()

    # Compute CSI from effective LR trajectory (last 25% of training)
    csi_start = int(0.75 * len(effective_lr_trajectory))
    csi_traj = effective_lr_trajectory[csi_start:]
    csi_value = compute_csi(csi_traj) if len(csi_traj) > 2 else 0.0

    # Get v2 cumulative budget
    v2_cumulative_budget = None
    if hasattr(wd_optimizer, 'get_cumulative_wd_budget'):
        v2_cumulative_budget = wd_optimizer.get_cumulative_wd_budget()

    result = {
        'variant': variant_name,
        'method': method_label,
        'K_p': K_p, 'K_i': K_i, 'K_d': K_d,
        'is_baseline': is_baseline,
        'seed': seed,
        'epochs': TOTAL_EPOCHS,
        'best_test_acc': best_acc,
        'final_test_acc': final_test_acc,
        'final_train_acc': final_train_acc,
        'generalization_gap': gen_gap,
        'total_wd_budget': total_wd_budget,
        'v2_cumulative_wd_budget': v2_cumulative_budget,
        'csi': csi_value,
        'nan_inf_detected': nan_inf_detected,
        'elapsed_sec': elapsed,
    }

    # Save per-run summary
    summary_file = Path(variant_save_dir) / f'{method_label}_seed{seed}_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def compute_aggregate_stats(variant_results):
    """Compute mean/std across seeds for a variant."""
    test_accs = [r['best_test_acc'] for r in variant_results]
    train_accs = [r['final_train_acc'] for r in variant_results]
    gen_gaps = [r['generalization_gap'] for r in variant_results]
    wd_budgets = [r['total_wd_budget'] for r in variant_results]
    csi_values = [r['csi'] for r in variant_results]
    times = [r['elapsed_sec'] for r in variant_results]

    return {
        'test_acc_mean': float(np.mean(test_accs)),
        'test_acc_std': float(np.std(test_accs)),
        'train_acc_mean': float(np.mean(train_accs)),
        'train_acc_std': float(np.std(train_accs)),
        'gen_gap_mean': float(np.mean(gen_gaps)),
        'gen_gap_std': float(np.std(gen_gaps)),
        'wd_budget_mean': float(np.mean(wd_budgets)),
        'wd_budget_std': float(np.std(wd_budgets)),
        'csi_mean': float(np.mean(csi_values)),
        'csi_std': float(np.std(csi_values)),
        'total_time_sec': float(np.sum(times)),
        'seeds': SEEDS,
        'per_seed': variant_results,
    }


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Write PID file
    pid_file = Path(RESULTS_BASE) / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    # CUDA_VISIBLE_DEVICES remaps GPUs
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {gpu_mem:.1f} GB")

    # Data -- load once, reuse for all variants and seeds
    data_root = f'{WORKSPACE}/exp/code/data'
    print(f"\nLoading CIFAR-100...")
    train_loader, test_loader = get_cifar100_loaders(
        batch_size=BATCH_SIZE, data_root=data_root,
        max_samples=None, seed=42  # Data loading seed fixed
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    print("\n" + "=" * 100)
    print("FULL: Phase 2a Ablation -- UDWDC-v2 Variants on CIFAR-100/VGG-16-BN")
    print(f"Epochs: {TOTAL_EPOCHS}, Seeds: {SEEDS}, Batch size: {BATCH_SIZE}")
    print(f"Model: {MODEL}, Dataset: {DATASET}")
    print("All UDWDC variants use v2 with floor clipping (0.1 * lambda_base)")
    print("=" * 100)

    all_results = {}
    start_total = time.time()
    total_runs = len(VARIANTS) * len(SEEDS)
    completed_runs = 0

    # Check for already completed runs (resume support)
    completed_variants_seeds = set()
    for variant_name, _, _, _, _, _ in VARIANTS:
        for seed in SEEDS:
            variant_dir = Path(SAVE_DIR) / f"{variant_name}_seed{seed}"
            method_label = "FixedWD" if variant_name == "FixedWD" else f"UDWDC_v2_{variant_name}"
            summary_file = variant_dir / f"{method_label}_seed{seed}_summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        existing = json.load(f)
                    if existing.get('epochs') == TOTAL_EPOCHS and existing.get('best_test_acc', 0) > 0:
                        completed_variants_seeds.add((variant_name, seed))
                        if variant_name not in all_results:
                            all_results[variant_name] = []
                        all_results[variant_name].append(existing)
                        completed_runs += 1
                        print(f"  [RESUME] {variant_name} seed={seed}: already complete "
                              f"(best={existing['best_test_acc']:.2f}%)")
                except Exception:
                    pass

    if completed_runs > 0:
        print(f"\n  Resuming: {completed_runs}/{total_runs} runs already complete.\n")

    for variant_name, K_p, K_i, K_d, desc, is_baseline in VARIANTS:
        if variant_name not in all_results:
            all_results[variant_name] = []

        for seed in SEEDS:
            if (variant_name, seed) in completed_variants_seeds:
                continue

            completed_runs += 1
            variant_dir = Path(SAVE_DIR) / f"{variant_name}_seed{seed}"
            os.makedirs(variant_dir, exist_ok=True)

            print(f"\n--- [{completed_runs}/{total_runs}] {variant_name} seed={seed} "
                  f"(K_p={K_p}, K_i={K_i}, K_d={K_d}) [{desc}] ---")

            try:
                result = run_single_variant(
                    variant_name, K_p, K_i, K_d, is_baseline, seed,
                    device, train_loader, test_loader, str(variant_dir)
                )
                all_results[variant_name].append(result)

                print(f"  -> Best: {result['best_test_acc']:.2f}% | "
                      f"WD Budget: {result['total_wd_budget']:.6f} | "
                      f"CSI: {result['csi']:.4f} | "
                      f"Time: {result['elapsed_sec']:.0f}s")

                report_progress(
                    completed_runs, total_runs, variant_name, seed,
                    TOTAL_EPOCHS, result['final_test_acc'], result['best_test_acc']
                )

            except Exception as e:
                print(f"ERROR in {variant_name} seed={seed}: {e}")
                traceback.print_exc()
                all_results[variant_name].append({
                    'variant': variant_name,
                    'seed': seed,
                    'error': str(e),
                })

            # Free GPU memory between runs
            torch.cuda.empty_cache()
            gc.collect()

    total_elapsed = time.time() - start_total

    # === Aggregate results ===
    print("\n" + "=" * 100)
    print("AGGREGATE RESULTS (mean +/- std over 3 seeds)")
    print("=" * 100)

    aggregate = {}
    for variant_name, _, _, _, _, _ in VARIANTS:
        results_list = all_results.get(variant_name, [])
        valid_results = [r for r in results_list if 'error' not in r]
        if valid_results:
            stats = compute_aggregate_stats(valid_results)
            aggregate[variant_name] = stats

    # Print table
    print(f"\n{'Variant':<15} {'K_p':>5} {'K_i':>5} {'K_d':>5} "
          f"{'Test Acc%':>15} {'Gen Gap%':>12} "
          f"{'WD Budget':>12} {'CSI':>12} {'Time(s)':>8}")
    print("-" * 110)

    for variant_name, K_p, K_i, K_d, desc, is_baseline in VARIANTS:
        if variant_name in aggregate:
            s = aggregate[variant_name]
            print(f"{variant_name:<15} {K_p:>5.1f} {K_i:>5.1f} {K_d:>5.1f} "
                  f"{s['test_acc_mean']:>7.2f}+/-{s['test_acc_std']:.2f} "
                  f"{s['gen_gap_mean']:>6.2f}+/-{s['gen_gap_std']:.2f} "
                  f"{s['wd_budget_mean']:>6.4f}+/-{s['wd_budget_std']:.4f} "
                  f"{s['csi_mean']:>6.4f}+/-{s['csi_std']:.4f} "
                  f"{s['total_time_sec']:>8.0f}")
        else:
            print(f"{variant_name:<15} --- INCOMPLETE ---")

    print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")

    # === Save comprehensive results ===

    # 1. Aggregate summary JSON
    summary = {
        'task_id': TASK_ID,
        'model': MODEL,
        'dataset': DATASET,
        'epochs': TOTAL_EPOCHS,
        'seeds': SEEDS,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'weight_decay': WD,
        'total_elapsed_sec': total_elapsed,
        'total_runs': total_runs,
        'completed_runs': sum(len([r for r in v if 'error' not in r])
                              for v in all_results.values()),
        'aggregate': {},
    }

    for variant_name in aggregate:
        s = aggregate[variant_name]
        summary['aggregate'][variant_name] = {
            'K_p': next(K_p for n, K_p, _, _, _, _ in VARIANTS if n == variant_name),
            'K_i': next(K_i for n, _, K_i, _, _, _ in VARIANTS if n == variant_name),
            'K_d': next(K_d for n, _, _, K_d, _, _ in VARIANTS if n == variant_name),
            'test_acc_mean': s['test_acc_mean'],
            'test_acc_std': s['test_acc_std'],
            'train_acc_mean': s['train_acc_mean'],
            'train_acc_std': s['train_acc_std'],
            'gen_gap_mean': s['gen_gap_mean'],
            'gen_gap_std': s['gen_gap_std'],
            'wd_budget_mean': s['wd_budget_mean'],
            'wd_budget_std': s['wd_budget_std'],
            'csi_mean': s['csi_mean'],
            'csi_std': s['csi_std'],
            'total_time_sec': s['total_time_sec'],
            'per_seed_accs': [r['best_test_acc'] for r in s['per_seed']],
        }

    summary_file = Path(SAVE_DIR) / 'ablation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # 2. Summary markdown
    md_file = Path(SAVE_DIR) / 'summary.md'
    with open(md_file, 'w') as f:
        f.write("# Full: Phase 2a Ablation -- UDWDC-v2 Variants on CIFAR-100/VGG-16-BN\n\n")
        f.write(f"- Epochs: {TOTAL_EPOCHS}, Seeds: {SEEDS}\n")
        f.write(f"- Model: {MODEL}, Dataset: {DATASET}\n")
        f.write(f"- Batch size: {BATCH_SIZE}, LR: {LR}, WD: {WD}\n")
        f.write(f"- All UDWDC variants use v2 with floor clipping (0.1 * lambda_base)\n")
        f.write(f"- Total time: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)\n\n")

        f.write("## Results\n\n")
        f.write("| Variant | K_p | K_i | K_d | Test Acc (%) | Gen Gap (%) | WD Budget | CSI |\n")
        f.write("|---------|-----|-----|-----|-------------|-------------|-----------|-----|\n")
        for variant_name, K_p, K_i, K_d, desc, is_baseline in VARIANTS:
            if variant_name in aggregate:
                s = aggregate[variant_name]
                f.write(f"| {variant_name} | {K_p:.1f} | {K_i:.1f} | {K_d:.1f} | "
                        f"{s['test_acc_mean']:.2f} +/- {s['test_acc_std']:.2f} | "
                        f"{s['gen_gap_mean']:.2f} +/- {s['gen_gap_std']:.2f} | "
                        f"{s['wd_budget_mean']:.4f} | "
                        f"{s['csi_mean']:.4f} |\n")

        f.write("\n## Per-Seed Results\n\n")
        f.write("| Variant | Seed 42 | Seed 123 | Seed 456 |\n")
        f.write("|---------|---------|----------|----------|\n")
        for variant_name, _, _, _, _, _ in VARIANTS:
            if variant_name in aggregate:
                per_seed = aggregate[variant_name]['per_seed']
                accs = [r['best_test_acc'] for r in per_seed]
                accs_str = " | ".join(f"{a:.2f}%" for a in accs)
                f.write(f"| {variant_name} | {accs_str} |\n")

        f.write("\n## Key Observations\n\n")
        # Find best variant
        if aggregate:
            best_variant = max(aggregate.items(), key=lambda x: x[1]['test_acc_mean'])
            f.write(f"- **Best variant**: {best_variant[0]} "
                    f"({best_variant[1]['test_acc_mean']:.2f}% +/- {best_variant[1]['test_acc_std']:.2f}%)\n")

            # Check WD budget
            for vn, vs in aggregate.items():
                if vs['wd_budget_mean'] <= 0:
                    f.write(f"- **WARNING**: {vn} has zero WD budget!\n")

    # 3. Detailed per-seed results JSON
    detail_file = Path(SAVE_DIR) / 'ablation_detailed_results.json'
    detail_data = {}
    for variant_name, results_list in all_results.items():
        detail_data[variant_name] = [
            r for r in results_list if 'error' not in r
        ]
    with open(detail_file, 'w') as f:
        json.dump(detail_data, f, indent=2)

    print(f"\nResults saved to: {SAVE_DIR}")
    print(f"  - ablation_summary.json")
    print(f"  - summary.md")
    print(f"  - ablation_detailed_results.json")

    # DONE marker
    pid_file = Path(RESULTS_BASE) / f"{TASK_ID}.pid"
    pid_file.unlink(missing_ok=True)
    done_file = Path(RESULTS_BASE) / f"{TASK_ID}_DONE"

    completed_count = summary['completed_runs']
    status = 'success' if completed_count == total_runs else 'partial'

    # Find best result for summary
    best_overall = 0.0
    best_variant_name = ""
    for vn, vs in aggregate.items():
        if vs['test_acc_mean'] > best_overall:
            best_overall = vs['test_acc_mean']
            best_variant_name = vn

    done_file.write_text(json.dumps({
        'task_id': TASK_ID,
        'status': status,
        'summary': f"Ablation full: {completed_count}/{total_runs} runs complete. "
                   f"Best: {best_variant_name} ({best_overall:.2f}%). "
                   f"Time: {total_elapsed/3600:.1f}h",
        'final_progress': {
            'completed_runs': completed_count,
            'total_runs': total_runs,
            'best_variant': best_variant_name,
            'best_acc_mean': best_overall,
            'aggregate': {vn: {'test_acc_mean': vs['test_acc_mean'],
                               'test_acc_std': vs['test_acc_std'],
                               'csi_mean': vs['csi_mean']}
                          for vn, vs in aggregate.items()},
        },
        'timestamp': datetime.now().isoformat(),
    }))

    # Update gpu_progress.json
    gpu_progress_file = Path(WORKSPACE) / 'exp' / 'gpu_progress.json'
    try:
        with open(gpu_progress_file, 'r') as f:
            gpu_progress = json.load(f)

        # Mark task completed
        if TASK_ID not in gpu_progress.get('completed', []):
            gpu_progress.setdefault('completed', []).append(TASK_ID)

        # Remove from running
        gpu_progress.get('running', {}).pop(TASK_ID, None)

        # Add timing
        gpu_progress.setdefault('timings', {})[TASK_ID] = {
            'planned_min': 60,
            'actual_min': int(total_elapsed / 60),
            'start_time': datetime.now().isoformat(),
            'end_time': datetime.now().isoformat(),
            'config_snapshot': {
                'model': MODEL,
                'batch_size': BATCH_SIZE,
                'dataset': DATASET,
                'epochs': TOTAL_EPOCHS,
                'seeds': len(SEEDS),
                'variants': len(VARIANTS),
                'total_runs': total_runs,
                'gpu_model': 'RTX PRO 6000 Blackwell',
                'gpu_count': 1,
            }
        }

        with open(gpu_progress_file, 'w') as f:
            json.dump(gpu_progress, f, indent=2)
        print(f"Updated gpu_progress.json")
    except Exception as e:
        print(f"Warning: Could not update gpu_progress.json: {e}")

    return status == 'success'


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
