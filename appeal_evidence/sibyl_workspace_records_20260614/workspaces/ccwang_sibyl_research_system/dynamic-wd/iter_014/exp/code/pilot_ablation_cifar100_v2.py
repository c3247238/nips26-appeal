#!/usr/bin/env python3
"""Pilot: Phase 2a Ablation -- UDWDC-v2 gain configurations on CIFAR-100/VGG-16-BN.

CRITICAL FIX from prior pilot (v1):
  - Prior pilot used UDWDC v1 which had Kp_only and PD_control with ZERO WD budget.
  - This version uses UDWDC-v2 with floor clipping (lambda_min = 0.1 * lambda_base)
    for ALL variants, preventing WD collapse.

Variants (8 total):
1. FixedWD baseline (constant lambda)
2. K_p-only (proportional control)
3. K_i-only (integral control)
4. K_d-only (alignment-derivative control)
5. PI control (K_p + K_i, CPR-like)
6. PD control (K_p + K_d, CWD-like)
7. Full PID (K_p + K_i + K_d)
8. UDWDC-v2 default (K_p=0.5, K_i=0.1, K_d=0.3 with EMA + floor)

Pass criteria:
- All 8 variants complete 10 epochs without error
- No NaN/Inf in lambda_t
- All variants achieve >20% accuracy at epoch 10
- ALL variants have total WD budget > 0 (the critical v2 fix verification)
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
SEED = 42
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WD = 1e-4
MODEL = "vgg16_bn"
DATASET = "cifar100"
NUM_CLASSES = 100

# Workspace paths
WORKSPACE = '/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current'
SAVE_DIR = f'{WORKSPACE}/exp/results/pilots/ablation_cifar100'
RESULTS_BASE = f'{WORKSPACE}/exp/results'

# UDWDC default gains
DEFAULT_KP = 0.5
DEFAULT_KI = 0.1
DEFAULT_KD = 0.3

# Ablation variants: (name, K_p, K_i, K_d, description)
# All use UDWDC-v2 with floor clipping
VARIANTS = [
    ("Kp_only",    DEFAULT_KP,  0.0,       0.0,       "Proportional only"),
    ("Ki_only",    0.0,         DEFAULT_KI, 0.0,       "Integral only"),
    ("Kd_only",    0.0,         0.0,        DEFAULT_KD, "Derivative/alignment only"),
    ("PI_control", DEFAULT_KP,  DEFAULT_KI, 0.0,       "PI (CPR-like)"),
    ("PD_control", DEFAULT_KP,  0.0,        DEFAULT_KD, "PD (CWD-like)"),
    ("Full_PID",   DEFAULT_KP,  DEFAULT_KI, DEFAULT_KD, "Full PID"),
    ("UDWDC_v2",   DEFAULT_KP,  DEFAULT_KI, DEFAULT_KD, "UDWDC-v2 default (same gains, v2 class)"),
]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_lr(wd_optimizer, epoch, total_epochs, base_lr):
    """Cosine annealing LR schedule (over 200 epochs, not pilot epochs)."""
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


def report_progress(results_base, task_id, completed, total, current_result=None):
    """Write progress file for system monitor."""
    progress_file = Path(results_base) / f"{task_id}_PROGRESS.json"
    metric = {}
    if current_result:
        metric = {
            'last_variant': current_result.get('variant', ''),
            'last_test_acc': current_result.get('final_test_acc', 0.0),
        }
    progress_file.write_text(json.dumps({
        'task_id': task_id,
        'epoch': completed,
        'total_epochs': total,
        'step': completed,
        'total_steps': total,
        'loss': 0.0,
        'metric': metric,
        'updated_at': datetime.now().isoformat(),
    }))


def run_variant(variant_name, K_p, K_i, K_d, device, save_dir,
                train_loader, test_loader, is_baseline=False, is_v2_default=False):
    """Run a single ablation variant and return results.

    Args:
        is_baseline: True for FixedWD baseline.
        is_v2_default: True for the UDWDC-v2 default variant (same gains as Full_PID
                       but explicitly uses the v2 class to verify stability fix).
    """
    set_seed(SEED)

    model = create_model(MODEL, num_classes=NUM_CLASSES).to(device)

    if is_baseline:
        wd_optimizer = create_optimizer(
            'FixedWD', model, lr=LR, momentum=MOMENTUM, weight_decay=WD
        )
    else:
        # CRITICAL: Use UDWDC-v2 for ALL non-baseline variants (floor clipping)
        wd_optimizer = create_optimizer(
            'UDWDC-v2', model, lr=LR, momentum=MOMENTUM, weight_decay=WD,
            K_p=K_p, K_i=K_i, K_d=K_d
        )

    criterion = nn.CrossEntropyLoss()
    method_label = "FixedWD" if is_baseline else f"UDWDC_v2_{variant_name}"
    logger = DiagnosticLogger(
        save_dir=save_dir, method=method_label, seed=SEED,
        model_name=MODEL, dataset=DATASET
    )

    start_time = time.time()
    epoch_results = []
    nan_inf_detected = False
    nan_inf_detail = ""
    wd_budget_per_epoch = []

    for epoch in range(PILOT_EPOCHS):
        lr = cosine_lr(wd_optimizer, epoch, 200, LR)  # cosine schedule over full 200 epochs

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

        # v2 epoch WD budget check
        epoch_wd_budget = float(np.sum(wd_values))
        wd_budget_per_epoch.append(epoch_wd_budget)

        # Call end_epoch_check for UDWDC-v2 to track cumulative budget
        if hasattr(wd_optimizer, 'end_epoch_check'):
            wd_optimizer.end_epoch_check()

        epoch_results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'lr': lr,
            'mean_effective_wd': float(mean_wd),
            'std_effective_wd': float(std_wd),
            'epoch_wd_budget': epoch_wd_budget,
        })

        print(f"  [{method_label}] Epoch {epoch+1}/{PILOT_EPOCHS} | "
              f"Train: {train_acc:.1f}% Test: {test_acc:.1f}% | "
              f"Loss: {train_loss:.4f} | WD: {mean_wd:.6f}+/-{std_wd:.6f} | "
              f"LR: {lr:.6f} | EpochBudget: {epoch_wd_budget:.6f}")

    logger.save()
    elapsed = time.time() - start_time

    final_test_acc = epoch_results[-1]['test_acc']
    final_train_acc = epoch_results[-1]['train_acc']
    gen_gap = final_train_acc - final_test_acc
    total_wd_budget = logger.get_total_wd_budget()

    # Also get cumulative budget from v2 internal tracker
    v2_cumulative_budget = None
    if hasattr(wd_optimizer, 'get_cumulative_wd_budget'):
        v2_cumulative_budget = wd_optimizer.get_cumulative_wd_budget()

    return {
        'variant': variant_name,
        'method': method_label,
        'K_p': K_p,
        'K_i': K_i,
        'K_d': K_d,
        'is_baseline': is_baseline,
        'is_v2_default': is_v2_default,
        'final_test_acc': final_test_acc,
        'final_train_acc': final_train_acc,
        'generalization_gap': gen_gap,
        'total_wd_budget': total_wd_budget,
        'v2_cumulative_wd_budget': v2_cumulative_budget,
        'wd_budget_per_epoch': wd_budget_per_epoch,
        'elapsed_sec': elapsed,
        'nan_inf_detected': nan_inf_detected,
        'nan_inf_detail': nan_inf_detail,
        'epochs_completed': PILOT_EPOCHS,
        'epoch_history': epoch_results,
    }


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Write PID file
    pid_file = Path(RESULTS_BASE) / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    # CUDA_VISIBLE_DEVICES remaps GPUs: physical GPU 2 becomes cuda:0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
              if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
              else f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data -- load once, reuse for all variants
    data_root = f'{WORKSPACE}/exp/code/data'
    print(f"\nLoading CIFAR-100...")
    train_loader, test_loader = get_cifar100_loaders(
        batch_size=BATCH_SIZE, data_root=data_root,
        max_samples=None, seed=SEED
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    print("\n" + "=" * 80)
    print("PILOT (v2): Phase 2a Ablation -- UDWDC-v2 Variants on CIFAR-100/VGG-16-BN")
    print(f"Epochs: {PILOT_EPOCHS}, Seed: {SEED}")
    print("KEY FIX: All UDWDC variants use v2 with floor clipping (0.1 * lambda_base)")
    print("=" * 80)

    all_results = {}
    start_total = time.time()
    total_variants = 1 + len(VARIANTS)  # 1 baseline + 7 UDWDC variants

    # 1. Run FixedWD baseline
    print("\n--- Running FixedWD baseline ---")
    try:
        result = run_variant("FixedWD", 0.0, 0.0, 0.0, device, SAVE_DIR,
                             train_loader, test_loader, is_baseline=True)
        all_results["FixedWD"] = result
        report_progress(RESULTS_BASE, TASK_ID, 1, total_variants, result)
        print(f"  -> WD Budget: {result['total_wd_budget']:.6f}")
    except Exception as e:
        print(f"ERROR in FixedWD: {e}")
        traceback.print_exc()
        all_results["FixedWD"] = {'error': str(e), 'variant': 'FixedWD'}

    # 2. Run UDWDC-v2 variants (all with floor clipping)
    for idx, (name, kp, ki, kd, desc) in enumerate(VARIANTS):
        is_v2_default = (name == "UDWDC_v2")
        print(f"\n--- Running UDWDC-v2 variant: {name} (K_p={kp}, K_i={ki}, K_d={kd}) [{desc}] ---")
        try:
            result = run_variant(name, kp, ki, kd, device, SAVE_DIR,
                                 train_loader, test_loader,
                                 is_baseline=False, is_v2_default=is_v2_default)
            all_results[name] = result
            report_progress(RESULTS_BASE, TASK_ID, idx + 2, total_variants, result)
            print(f"  -> WD Budget: {result['total_wd_budget']:.6f} "
                  f"(v2 internal: {result['v2_cumulative_wd_budget']:.6f})")
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
    details = {}

    # Check 1: All 8 variants complete
    completed_count = sum(1 for r in all_results.values() if 'error' not in r)
    checks['all_8_variants_complete'] = completed_count == total_variants
    print(f"{'PASS' if checks['all_8_variants_complete'] else 'FAIL'}: "
          f"All {total_variants} variants complete ({completed_count}/{total_variants})")

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

    # Check 4 (NEW, CRITICAL): ALL variants have total WD budget > 0
    zero_budget_variants = []
    for name, r in all_results.items():
        if 'error' not in r:
            budget = r['total_wd_budget']
            if budget <= 0:
                zero_budget_variants.append((name, budget))
    checks['all_wd_budget_positive'] = len(zero_budget_variants) == 0
    print(f"{'PASS' if checks['all_wd_budget_positive'] else 'FAIL'}: "
          f"ALL variants have WD budget > 0" +
          (f" (ZERO budget: {zero_budget_variants})" if zero_budget_variants else ""))
    if zero_budget_variants:
        details['zero_budget_variants'] = zero_budget_variants

    overall_pass = all(checks.values())
    recommendation = "GO" if overall_pass else "NO_GO"

    # === Results table ===
    print("\n" + "-" * 120)
    print(f"{'Variant':<15} {'K_p':>5} {'K_i':>5} {'K_d':>5} "
          f"{'Test Acc%':>10} {'Train Acc%':>11} {'Gap%':>7} "
          f"{'WD Budget':>10} {'v2 Budget':>10} {'Time(s)':>8}")
    print("-" * 120)
    for name, r in all_results.items():
        if 'error' in r:
            print(f"{name:<15} --- ERROR: {r['error'][:60]}")
        else:
            v2_budget = r.get('v2_cumulative_wd_budget', 'N/A')
            v2_str = f"{v2_budget:.4f}" if isinstance(v2_budget, float) else str(v2_budget)
            print(f"{r['variant']:<15} {r['K_p']:>5.1f} {r['K_i']:>5.1f} {r['K_d']:>5.1f} "
                  f"{r['final_test_acc']:>10.2f} {r['final_train_acc']:>11.2f} "
                  f"{r['generalization_gap']:>7.2f} {r['total_wd_budget']:>10.4f} "
                  f"{v2_str:>10} {r['elapsed_sec']:>8.1f}")
    print("-" * 120)
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")

    # === Compare with prior v1 pilot ===
    print("\n" + "=" * 80)
    print("COMPARISON: v1 vs v2 WD Budget Fix")
    print("=" * 80)
    v1_budgets = {
        'FixedWD': 0.029, 'Kp_only': 0.0, 'Ki_only': 0.005,
        'Kd_only': 0.041, 'PI_control': 0.001, 'PD_control': 0.0,
        'Full_PID': 0.001
    }
    for name in ['FixedWD', 'Kp_only', 'Ki_only', 'Kd_only',
                 'PI_control', 'PD_control', 'Full_PID']:
        r = all_results.get(name)
        if r and 'error' not in r:
            v1_b = v1_budgets.get(name, '?')
            v2_b = r['total_wd_budget']
            fixed = "FIXED" if v1_b == 0.0 and v2_b > 0 else ("OK" if v2_b > 0 else "STILL ZERO!")
            print(f"  {name:<15} v1={v1_b:<10} v2={v2_b:<10.6f} [{fixed}]")

    # === Save pilot summary ===
    pilot_summary = {
        'overall_recommendation': recommendation,
        'candidates': [],
        'task_id': TASK_ID,
        'model': MODEL,
        'dataset': DATASET,
        'pilot_epochs': PILOT_EPOCHS,
        'pilot_samples': None,
        'seed': SEED,
        'version': 'v2',  # Mark as v2 pilot
        'key_fix': 'All UDWDC variants use v2 with floor clipping (0.1 * lambda_base)',
        'pass_criteria': checks,
        'details': details,
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
                'is_v2_default': r.get('is_v2_default', False),
                'final_test_acc': r['final_test_acc'],
                'final_train_acc': r['final_train_acc'],
                'generalization_gap': r['generalization_gap'],
                'total_wd_budget': r['total_wd_budget'],
                'v2_cumulative_wd_budget': r.get('v2_cumulative_wd_budget'),
                'wd_budget_per_epoch': r.get('wd_budget_per_epoch', []),
                'nan_inf_detected': r['nan_inf_detected'],
                'elapsed_sec': r['elapsed_sec'],
            }
        else:
            pilot_summary['variants'][name] = {'error': r['error']}

    # Save to pilots directory
    summary_file = Path(SAVE_DIR) / 'pilot_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(pilot_summary, f, indent=2)

    # Save detailed results with epoch history
    detail_file = Path(SAVE_DIR) / 'pilot_detailed_results.json'
    detail_data = {}
    for name, r in all_results.items():
        if 'error' not in r:
            detail_data[name] = {
                'variant': r['variant'],
                'method': r['method'],
                'K_p': r['K_p'], 'K_i': r['K_i'], 'K_d': r['K_d'],
                'is_baseline': r['is_baseline'],
                'final_test_acc': r['final_test_acc'],
                'final_train_acc': r['final_train_acc'],
                'generalization_gap': r['generalization_gap'],
                'total_wd_budget': r['total_wd_budget'],
                'v2_cumulative_wd_budget': r.get('v2_cumulative_wd_budget'),
                'wd_budget_per_epoch': r.get('wd_budget_per_epoch', []),
                'epoch_history': r['epoch_history'],
            }
    with open(detail_file, 'w') as f:
        json.dump(detail_data, f, indent=2)

    # Summary markdown
    md_file = Path(SAVE_DIR) / 'pilot_summary.md'
    with open(md_file, 'w') as f:
        f.write("# Pilot (v2): Phase 2a Ablation -- UDWDC-v2 Variants on CIFAR-100/VGG-16-BN\n\n")
        f.write(f"**Recommendation: {recommendation}**\n\n")
        f.write(f"- Epochs: {PILOT_EPOCHS} (pilot), Seed: {SEED}\n")
        f.write(f"- Model: {MODEL}, Dataset: {DATASET}\n")
        f.write(f"- **KEY FIX**: All UDWDC variants use v2 with floor clipping (0.1 * lambda_base)\n")
        f.write(f"- Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)\n\n")
        f.write("## Pass Criteria\n\n")
        for check, passed in checks.items():
            f.write(f"- {'PASS' if passed else 'FAIL'}: {check}\n")
        f.write("\n## Results\n\n")
        f.write("| Variant | K_p | K_i | K_d | Test Acc (%) | Train Acc (%) | Gap (%) | WD Budget | v2 Budget | Time (s) |\n")
        f.write("|---------|-----|-----|-----|-------------|--------------|---------|-----------|-----------|----------|\n")
        for name, r in all_results.items():
            if 'error' not in r:
                v2_budget = r.get('v2_cumulative_wd_budget', 'N/A')
                v2_str = f"{v2_budget:.4f}" if isinstance(v2_budget, float) else str(v2_budget)
                f.write(f"| {r['variant']} | {r['K_p']:.1f} | {r['K_i']:.1f} | {r['K_d']:.1f} | "
                        f"{r['final_test_acc']:.2f} | {r['final_train_acc']:.2f} | "
                        f"{r['generalization_gap']:.2f} | {r['total_wd_budget']:.4f} | "
                        f"{v2_str} | {r['elapsed_sec']:.1f} |\n")
            else:
                f.write(f"| {name} | - | - | - | ERROR | - | - | - | - | - |\n")
        f.write("\n## v1 vs v2 WD Budget Comparison\n\n")
        f.write("| Variant | v1 Budget | v2 Budget | Status |\n")
        f.write("|---------|-----------|-----------|--------|\n")
        for vname in ['FixedWD', 'Kp_only', 'Ki_only', 'Kd_only',
                      'PI_control', 'PD_control', 'Full_PID']:
            r = all_results.get(vname)
            if r and 'error' not in r:
                v1_b = v1_budgets.get(vname, '?')
                v2_b = r['total_wd_budget']
                status = "FIXED" if v1_b == 0.0 and v2_b > 0 else ("OK" if v2_b > 0 else "STILL ZERO")
                f.write(f"| {vname} | {v1_b} | {v2_b:.6f} | {status} |\n")

    print(f"\nResults saved to: {SAVE_DIR}")
    print(f"\nOverall: {recommendation}")

    # DONE marker
    pid_file = Path(RESULTS_BASE) / f"{TASK_ID}.pid"
    pid_file.unlink(missing_ok=True)
    done_file = Path(RESULTS_BASE) / f"{TASK_ID}_DONE"
    done_file.write_text(json.dumps({
        'task_id': TASK_ID,
        'status': 'success' if overall_pass else 'partial',
        'summary': f"Ablation pilot v2: {recommendation}. "
                   f"{completed_count}/{total_variants} variants complete. "
                   f"Zero-budget variants: {len(zero_budget_variants)}. "
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
