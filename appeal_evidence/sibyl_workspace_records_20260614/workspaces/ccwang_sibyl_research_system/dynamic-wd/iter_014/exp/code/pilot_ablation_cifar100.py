#!/usr/bin/env python3
"""Pilot: Phase 2 Ablation — UDWDC gain configurations on CIFAR-100/VGG-16-BN.

Runs 7 UDWDC variants for 10 epochs on CIFAR-100 (pilot sample budget)
to verify all variants train without errors and produce non-degenerate results.

Variants:
1. K_p-only (proportional control)
2. K_i-only (integral control)
3. K_d-only (alignment-derivative control)
4. K_p+K_i (PI control, CPR-like)
5. K_p+K_d (PD control, CWD-like)
6. Full PID (K_p+K_i+K_d)
7. FixedWD baseline

Pass criteria:
- All 7 variants complete 10 epochs without error
- No NaN/Inf in lambda_t
- All variants achieve >20% accuracy at epoch 10 (training is progressing)
"""

import json
import os
import sys
import time
import math
import traceback
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


TASK_ID = "ablation_cifar100"
PILOT_EPOCHS = 10
PILOT_MAX_SAMPLES = None  # Use full dataset; 10 epochs on VGG-16-BN/CIFAR-100 is already fast
SEED = 42
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WD = 1e-4
MODEL = "vgg16_bn"
DATASET = "cifar100"
NUM_CLASSES = 100

# UDWDC default gains
DEFAULT_KP = 0.5
DEFAULT_KI = 0.1
DEFAULT_KD = 0.3

# Ablation variants: (name, K_p, K_i, K_d)
VARIANTS = [
    ("Kp_only",   DEFAULT_KP,  0.0,       0.0),       # Proportional only
    ("Ki_only",   0.0,         DEFAULT_KI, 0.0),       # Integral only
    ("Kd_only",   0.0,         0.0,        DEFAULT_KD), # Derivative only
    ("PI_control", DEFAULT_KP, DEFAULT_KI, 0.0),       # PI (CPR-like)
    ("PD_control", DEFAULT_KP, 0.0,        DEFAULT_KD), # PD (CWD-like)
    ("Full_PID",   DEFAULT_KP, DEFAULT_KI, DEFAULT_KD), # Full PID
]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_lr(wd_optimizer, epoch, total_epochs, base_lr):
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


def run_variant(variant_name, K_p, K_i, K_d, device, save_dir,
                train_loader, test_loader, is_baseline=False):
    """Run a single ablation variant and return results."""
    set_seed(SEED)

    model = create_model(MODEL, num_classes=NUM_CLASSES).to(device)

    if is_baseline:
        wd_optimizer = create_optimizer(
            'FixedWD', model, lr=LR, momentum=MOMENTUM, weight_decay=WD
        )
    else:
        wd_optimizer = create_optimizer(
            'UDWDC', model, lr=LR, momentum=MOMENTUM, weight_decay=WD,
            K_p=K_p, K_i=K_i, K_d=K_d
        )

    criterion = nn.CrossEntropyLoss()
    method_label = "FixedWD" if is_baseline else f"UDWDC_{variant_name}"
    logger = DiagnosticLogger(
        save_dir=save_dir, method=method_label, seed=SEED,
        model_name=MODEL, dataset=DATASET
    )

    start_time = time.time()
    epoch_results = []
    nan_inf_detected = False
    nan_inf_detail = ""

    for epoch in range(PILOT_EPOCHS):
        lr = cosine_lr(wd_optimizer, epoch, 200, LR)  # cosine schedule over 200 epochs

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
            nan_inf_detail = f"Epoch {epoch}: {detail}"

        # Compute effective WD stats
        wd_values = [m['effective_wd'] for m in diagnostics.values()]
        mean_wd = np.mean(wd_values)
        std_wd = np.std(wd_values)

        epoch_results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'lr': lr,
            'mean_effective_wd': float(mean_wd),
            'std_effective_wd': float(std_wd),
        })

        print(f"  [{method_label}] Epoch {epoch+1}/{PILOT_EPOCHS} | "
              f"Train: {train_acc:.1f}% Test: {test_acc:.1f}% | "
              f"Loss: {train_loss:.4f} | WD: {mean_wd:.6f}+/-{std_wd:.6f} | "
              f"LR: {lr:.6f}")

    logger.save()
    elapsed = time.time() - start_time

    final_test_acc = epoch_results[-1]['test_acc']
    final_train_acc = epoch_results[-1]['train_acc']
    gen_gap = final_train_acc - final_test_acc
    total_wd_budget = logger.get_total_wd_budget()

    return {
        'variant': variant_name,
        'method': method_label,
        'K_p': K_p,
        'K_i': K_i,
        'K_d': K_d,
        'is_baseline': is_baseline,
        'final_test_acc': final_test_acc,
        'final_train_acc': final_train_acc,
        'generalization_gap': gen_gap,
        'total_wd_budget': total_wd_budget,
        'elapsed_sec': elapsed,
        'nan_inf_detected': nan_inf_detected,
        'nan_inf_detail': nan_inf_detail,
        'epochs_completed': PILOT_EPOCHS,
        'epoch_history': epoch_results,
    }


def main():
    save_dir = '/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/pilots/ablation_cifar100'
    results_base = '/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results'
    os.makedirs(save_dir, exist_ok=True)

    # Write PID file
    pid_file = Path(results_base) / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    # CUDA_VISIBLE_DEVICES remaps GPUs: physical GPU 2 becomes cuda:0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data — load once, reuse for all variants
    data_root = '/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/code/data'
    print(f"\nLoading CIFAR-100 (max_samples={PILOT_MAX_SAMPLES})...")
    train_loader, test_loader = get_cifar100_loaders(
        batch_size=BATCH_SIZE, data_root=data_root,
        max_samples=PILOT_MAX_SAMPLES, seed=SEED
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    print("\n" + "=" * 80)
    print("PILOT: Phase 2 Ablation — UDWDC Variants on CIFAR-100/VGG-16-BN")
    print(f"Epochs: {PILOT_EPOCHS}, Samples: {PILOT_MAX_SAMPLES}, Seed: {SEED}")
    print("=" * 80)

    all_results = {}
    start_total = time.time()

    # Progress reporting helper
    def report_progress(completed, total, current_result=None):
        progress_file = Path(results_base) / f"{TASK_ID}_PROGRESS.json"
        metric = {}
        if current_result:
            metric = {
                'last_variant': current_result['variant'],
                'last_test_acc': current_result['final_test_acc'],
            }
        progress_file.write_text(json.dumps({
            'task_id': TASK_ID,
            'epoch': completed,
            'total_epochs': total,
            'loss': 0.0,
            'metric': metric,
            'updated_at': datetime.now().isoformat(),
        }))

    # 1. Run FixedWD baseline
    print("\n--- Running FixedWD baseline ---")
    try:
        result = run_variant("FixedWD", 0.0, 0.0, 0.0, device, save_dir,
                             train_loader, test_loader, is_baseline=True)
        all_results["FixedWD"] = result
        report_progress(1, 7, result)
    except Exception as e:
        print(f"ERROR in FixedWD: {e}")
        traceback.print_exc()
        all_results["FixedWD"] = {'error': str(e), 'variant': 'FixedWD'}

    # 2. Run UDWDC variants
    for idx, (name, kp, ki, kd) in enumerate(VARIANTS):
        print(f"\n--- Running UDWDC variant: {name} (K_p={kp}, K_i={ki}, K_d={kd}) ---")
        try:
            result = run_variant(name, kp, ki, kd, device, save_dir,
                                 train_loader, test_loader, is_baseline=False)
            all_results[name] = result
            report_progress(idx + 2, 7, result)
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            traceback.print_exc()
            all_results[name] = {'error': str(e), 'variant': name}

    total_elapsed = time.time() - start_total

    # === Pass criteria checks ===
    print("\n" + "=" * 80)
    print("PILOT PASS CRITERIA CHECKS")
    print("=" * 80)

    checks = {}

    # Check 1: All 7 variants complete
    completed_count = sum(1 for r in all_results.values() if 'error' not in r)
    checks['all_7_variants_complete'] = completed_count == 7
    print(f"{'PASS' if checks['all_7_variants_complete'] else 'FAIL'}: "
          f"All 7 variants complete ({completed_count}/7)")

    # Check 2: No NaN/Inf in lambda_t
    nan_variants = [r['variant'] for r in all_results.values()
                    if 'error' not in r and r.get('nan_inf_detected', False)]
    checks['no_nan_inf'] = len(nan_variants) == 0
    print(f"{'PASS' if checks['no_nan_inf'] else 'FAIL'}: "
          f"No NaN/Inf in lambda_t" +
          (f" (violations: {nan_variants})" if nan_variants else ""))

    # Check 3: All variants >20% accuracy at epoch 10
    low_acc_variants = []
    for name, r in all_results.items():
        if 'error' not in r:
            acc = r['final_test_acc']
            if acc < 20.0:
                low_acc_variants.append((name, acc))
    checks['all_above_20pct'] = len(low_acc_variants) == 0
    print(f"{'PASS' if checks['all_above_20pct'] else 'FAIL'}: "
          f"All variants >20% accuracy" +
          (f" (below threshold: {low_acc_variants})" if low_acc_variants else ""))

    overall_pass = all(checks.values())
    recommendation = "GO" if overall_pass else "NO_GO"

    # === Results table ===
    print("\n" + "-" * 100)
    print(f"{'Variant':<15} {'K_p':>5} {'K_i':>5} {'K_d':>5} "
          f"{'Test Acc%':>10} {'Train Acc%':>11} {'Gap%':>7} "
          f"{'WD Budget':>10} {'Time(s)':>8}")
    print("-" * 100)
    for name, r in all_results.items():
        if 'error' in r:
            print(f"{name:<15} --- ERROR: {r['error'][:60]}")
        else:
            print(f"{r['variant']:<15} {r['K_p']:>5.1f} {r['K_i']:>5.1f} {r['K_d']:>5.1f} "
                  f"{r['final_test_acc']:>10.2f} {r['final_train_acc']:>11.2f} "
                  f"{r['generalization_gap']:>7.2f} {r['total_wd_budget']:>10.4f} "
                  f"{r['elapsed_sec']:>8.1f}")
    print("-" * 100)
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    # === Save pilot summary ===
    pilot_summary = {
        'overall_recommendation': recommendation,
        'candidates': [],
        'task_id': TASK_ID,
        'model': MODEL,
        'dataset': DATASET,
        'pilot_epochs': PILOT_EPOCHS,
        'pilot_samples': PILOT_MAX_SAMPLES,
        'seed': SEED,
        'pass_criteria': checks,
        'variants': {},
        'total_elapsed_sec': total_elapsed,
    }

    for name, r in all_results.items():
        if 'error' not in r:
            pilot_summary['variants'][name] = {
                'K_p': r['K_p'],
                'K_i': r['K_i'],
                'K_d': r['K_d'],
                'is_baseline': r['is_baseline'],
                'final_test_acc': r['final_test_acc'],
                'final_train_acc': r['final_train_acc'],
                'generalization_gap': r['generalization_gap'],
                'total_wd_budget': r['total_wd_budget'],
                'nan_inf_detected': r['nan_inf_detected'],
                'elapsed_sec': r['elapsed_sec'],
            }
        else:
            pilot_summary['variants'][name] = {'error': r['error']}

    # Save to pilots directory
    summary_file = Path(save_dir) / 'pilot_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(pilot_summary, f, indent=2)

    # Also save detailed results with epoch history
    detail_file = Path(save_dir) / 'pilot_detailed_results.json'
    # Prepare serializable version
    detail_data = {}
    for name, r in all_results.items():
        if 'error' not in r:
            detail_data[name] = {
                'variant': r['variant'],
                'K_p': r['K_p'], 'K_i': r['K_i'], 'K_d': r['K_d'],
                'is_baseline': r['is_baseline'],
                'final_test_acc': r['final_test_acc'],
                'final_train_acc': r['final_train_acc'],
                'generalization_gap': r['generalization_gap'],
                'total_wd_budget': r['total_wd_budget'],
                'epoch_history': r['epoch_history'],
            }
    with open(detail_file, 'w') as f:
        json.dump(detail_data, f, indent=2)

    # Summary markdown
    md_file = Path(save_dir) / 'pilot_summary.md'
    with open(md_file, 'w') as f:
        f.write("# Pilot: Phase 2 Ablation — UDWDC Variants on CIFAR-100/VGG-16-BN\n\n")
        f.write(f"**Recommendation: {recommendation}**\n\n")
        f.write(f"- Epochs: {PILOT_EPOCHS} (pilot), Samples: {PILOT_MAX_SAMPLES}, Seed: {SEED}\n")
        f.write(f"- Model: {MODEL}, Dataset: {DATASET}\n")
        f.write(f"- Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)\n\n")
        f.write("## Pass Criteria\n\n")
        for check, passed in checks.items():
            f.write(f"- {'PASS' if passed else 'FAIL'}: {check}\n")
        f.write("\n## Results\n\n")
        f.write("| Variant | K_p | K_i | K_d | Test Acc (%) | Train Acc (%) | Gap (%) | WD Budget | Time (s) |\n")
        f.write("|---------|-----|-----|-----|-------------|--------------|---------|-----------|----------|\n")
        for name, r in all_results.items():
            if 'error' not in r:
                f.write(f"| {r['variant']} | {r['K_p']:.1f} | {r['K_i']:.1f} | {r['K_d']:.1f} | "
                        f"{r['final_test_acc']:.2f} | {r['final_train_acc']:.2f} | "
                        f"{r['generalization_gap']:.2f} | {r['total_wd_budget']:.4f} | "
                        f"{r['elapsed_sec']:.1f} |\n")
            else:
                f.write(f"| {name} | - | - | - | ERROR | - | - | - | - |\n")

    print(f"\nResults saved to: {save_dir}")
    print(f"\nOverall: {recommendation}")

    # DONE marker
    pid_file.unlink(missing_ok=True)
    done_file = Path(results_base) / f"{TASK_ID}_DONE"
    done_file.write_text(json.dumps({
        'task_id': TASK_ID,
        'status': 'success' if overall_pass else 'partial',
        'summary': f"Ablation pilot: {recommendation}. "
                   f"{completed_count}/7 variants complete. "
                   f"Best test acc: {max((r['final_test_acc'] for r in all_results.values() if 'error' not in r), default=0):.2f}%",
        'final_progress': {
            'recommendation': recommendation,
            'variants_completed': completed_count,
            'checks': checks,
        },
        'timestamp': datetime.now().isoformat(),
    }))

    return overall_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
