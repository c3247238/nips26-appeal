#!/usr/bin/env python3
"""Pilot: Phase 2b — Batch-size sweep: binary vs continuous alignment (H3).

Train VGG-16-BN on CIFAR-100 for 5 epochs (pilot) at batch sizes
{64, 128, 256, 512, 1024} with 3 methods:
  - FixedWD (constant lambda)
  - CWD (binary sign alignment)
  - UDWDC (continuous cosine alignment, K_d > 0)

Track per-batch-size:
  (a) Effective alignment SNR = E[alpha]^2 / Var[alpha]
  (b) Test accuracy difference (CWD - FixedWD) and (UDWDC - FixedWD)

LR scaled linearly with batch size: LR = LR_base * (bs / 128).

Pass criteria:
  - All 15 configurations (5 BS x 3 methods) complete 5 epochs
  - SNR varies monotonically with batch size (rough trend acceptable)
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


# ── Constants ──────────────────────────────────────────────────────────────────
TASK_ID = "batchsize_sweep"
PILOT_EPOCHS = 5
PILOT_MAX_SAMPLES = 100  # Pilot uses subset
SEED = 42
LR_BASE = 0.1
LR_REF_BS = 128  # Reference batch size for linear scaling
MOMENTUM = 0.9
WD = 1e-4
MODEL = "vgg16_bn"
DATASET = "cifar100"
NUM_CLASSES = 100

BATCH_SIZES = [64, 128, 256, 512, 1024]
METHODS = ["FixedWD", "CWD", "UDWDC"]

# UDWDC gains (continuous alignment)
UDWDC_KP = 0.5
UDWDC_KI = 0.1
UDWDC_KD = 0.3

# Paths
WORKSPACE = "/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current"
RESULTS_BASE = f"{WORKSPACE}/exp/results"
SAVE_DIR = f"{RESULTS_BASE}/pilots/batchsize_sweep"
DATA_ROOT = f"{WORKSPACE}/exp/code/data"


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
    """Train one epoch. Returns (loss, accuracy, per-batch alpha list)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_alphas = []  # Per-batch mean alignment for SNR calculation

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Collect per-batch alignment BEFORE optimizer step modifies grads
        diag = optimizer.get_diagnostics()
        alpha_values = [m['alpha_t'] for m in diag.values() if 'alpha_t' in m]
        if alpha_values:
            batch_alphas.append(float(np.mean(alpha_values)))

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / max(total, 1), 100.0 * correct / max(total, 1), batch_alphas


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
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def compute_alignment_snr(all_batch_alphas):
    """Compute alignment SNR = E[alpha]^2 / Var[alpha].

    Args:
        all_batch_alphas: list of per-batch mean alpha values across all epochs.
    Returns:
        snr: float, or NaN if variance is zero.
    """
    if not all_batch_alphas or len(all_batch_alphas) < 2:
        return float('nan')
    arr = np.array(all_batch_alphas)
    mean_alpha = np.mean(arr)
    var_alpha = np.var(arr)
    if var_alpha < 1e-12:
        return float('inf')  # No variance => infinite SNR
    return float(mean_alpha ** 2 / var_alpha)


def run_config(batch_size, method, lr, device):
    """Run a single (batch_size, method) configuration.

    Returns dict with results including alignment SNR.
    """
    set_seed(SEED)

    # Data loader with this batch size
    # For very large BS with small sample count, adjust: need at least 1 batch
    effective_max_samples = max(PILOT_MAX_SAMPLES, batch_size * 2)
    train_loader, test_loader = get_cifar100_loaders(
        batch_size=batch_size, data_root=DATA_ROOT,
        max_samples=effective_max_samples, seed=SEED, num_workers=2
    )

    model = create_model(MODEL, num_classes=NUM_CLASSES).to(device)

    opt_kwargs = {}
    if method == "UDWDC":
        opt_kwargs = {'K_p': UDWDC_KP, 'K_i': UDWDC_KI, 'K_d': UDWDC_KD}

    wd_optimizer = create_optimizer(
        method, model, lr=lr, momentum=MOMENTUM, weight_decay=WD, **opt_kwargs
    )

    criterion = nn.CrossEntropyLoss()
    config_label = f"{method}_bs{batch_size}"
    logger = DiagnosticLogger(
        save_dir=SAVE_DIR, method=config_label, seed=SEED,
        model_name=MODEL, dataset=DATASET
    )

    start = time.time()
    epoch_results = []
    all_batch_alphas = []  # Accumulate per-batch alphas across all epochs

    for epoch in range(PILOT_EPOCHS):
        lr_epoch = cosine_lr(wd_optimizer, epoch, 200, lr)

        train_loss, train_acc, batch_alphas = train_one_epoch(
            model, train_loader, wd_optimizer, criterion, device
        )
        all_batch_alphas.extend(batch_alphas)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        diagnostics = wd_optimizer.get_diagnostics()
        logger.log_epoch(epoch, diagnostics, train_loss, test_loss, train_acc, test_acc, lr=lr_epoch)

        wd_values = [m['effective_wd'] for m in diagnostics.values()]
        alpha_values = [m['alpha_t'] for m in diagnostics.values()]

        epoch_results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'lr': lr_epoch,
            'mean_effective_wd': float(np.mean(wd_values)),
            'mean_alpha': float(np.mean(alpha_values)) if alpha_values else 0.0,
            'std_alpha': float(np.std(alpha_values)) if alpha_values else 0.0,
        })

        print(f"    [{config_label}] Epoch {epoch+1}/{PILOT_EPOCHS} | "
              f"Train: {train_acc:.1f}% Test: {test_acc:.1f}% | "
              f"Loss: {train_loss:.4f} | alpha_mean: {epoch_results[-1]['mean_alpha']:.4f}")

    logger.save()
    elapsed = time.time() - start

    # Compute alignment SNR across all batches
    snr = compute_alignment_snr(all_batch_alphas)

    return {
        'method': method,
        'batch_size': batch_size,
        'lr': lr,
        'final_test_acc': epoch_results[-1]['test_acc'],
        'final_train_acc': epoch_results[-1]['train_acc'],
        'generalization_gap': epoch_results[-1]['train_acc'] - epoch_results[-1]['test_acc'],
        'alignment_snr': snr,
        'mean_alpha_final': epoch_results[-1]['mean_alpha'],
        'total_wd_budget': logger.get_total_wd_budget(),
        'elapsed_sec': elapsed,
        'epochs_completed': PILOT_EPOCHS,
        'num_batch_alphas': len(all_batch_alphas),
        'epoch_history': epoch_results,
    }


def check_snr_monotonicity(results_by_bs_method):
    """Check if SNR varies roughly monotonically with batch size.

    We check each method separately. 'Rough' means at most 1 violation
    in the sequence.
    """
    checks = {}
    for method in METHODS:
        snr_seq = []
        for bs in BATCH_SIZES:
            key = f"{method}_bs{bs}"
            r = results_by_bs_method.get(key)
            if r and not math.isnan(r['alignment_snr']):
                snr_seq.append((bs, r['alignment_snr']))

        if len(snr_seq) < 3:
            checks[method] = {
                'monotonic': False,
                'reason': f"Only {len(snr_seq)} valid SNR values",
                'snr_values': snr_seq,
            }
            continue

        # Count direction changes (increasing violations for decreasing trend or vice versa)
        diffs = [snr_seq[i+1][1] - snr_seq[i][1] for i in range(len(snr_seq)-1)]
        # Determine dominant direction
        pos_count = sum(1 for d in diffs if d > 0)
        neg_count = sum(1 for d in diffs if d < 0)
        violations = min(pos_count, neg_count)  # Minority direction = violations

        checks[method] = {
            'monotonic': violations <= 1,  # Allow 1 violation for "rough trend"
            'violations': violations,
            'dominant_direction': 'increasing' if pos_count >= neg_count else 'decreasing',
            'snr_values': snr_seq,
        }

    return checks


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # PID file
    pid_file = Path(RESULTS_BASE) / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    # Device: CUDA_VISIBLE_DEVICES remaps GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
              if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
              else f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n" + "=" * 90)
    print("PILOT: Phase 2b — Batch-size Sweep: Binary vs Continuous Alignment (H3)")
    print(f"Model: {MODEL}, Dataset: {DATASET}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Methods: {METHODS}")
    print(f"Pilot epochs: {PILOT_EPOCHS}, Max samples: {PILOT_MAX_SAMPLES}, Seed: {SEED}")
    print(f"LR scaling: LR = {LR_BASE} * (bs / {LR_REF_BS})")
    print("=" * 90)

    all_results = {}
    total_configs = len(BATCH_SIZES) * len(METHODS)
    completed = 0
    start_total = time.time()

    def report_progress(done, total_cfg, last_result=None):
        progress_file = Path(RESULTS_BASE) / f"{TASK_ID}_PROGRESS.json"
        metric = {}
        if last_result:
            metric = {
                'last_config': f"{last_result['method']}_bs{last_result['batch_size']}",
                'last_test_acc': last_result['final_test_acc'],
                'last_snr': last_result['alignment_snr'],
            }
        progress_file.write_text(json.dumps({
            'task_id': TASK_ID,
            'epoch': done,
            'total_epochs': total_cfg,
            'loss': 0.0,
            'metric': metric,
            'updated_at': datetime.now().isoformat(),
        }))

    for bs in BATCH_SIZES:
        lr = LR_BASE * (bs / LR_REF_BS)
        print(f"\n{'─' * 70}")
        print(f"Batch size: {bs}, LR: {lr:.4f}")
        print(f"{'─' * 70}")

        for method in METHODS:
            config_key = f"{method}_bs{bs}"
            print(f"\n  >>> {config_key} (LR={lr:.4f})")

            try:
                result = run_config(bs, method, lr, device)
                all_results[config_key] = result
                completed += 1
                report_progress(completed, total_configs, result)

                print(f"  <<< {config_key} DONE | "
                      f"Test: {result['final_test_acc']:.1f}% | "
                      f"SNR: {result['alignment_snr']:.4f} | "
                      f"Time: {result['elapsed_sec']:.1f}s")

            except Exception as e:
                print(f"  !!! ERROR in {config_key}: {e}")
                traceback.print_exc()
                all_results[config_key] = {
                    'method': method,
                    'batch_size': bs,
                    'error': str(e),
                }
                completed += 1
                report_progress(completed, total_configs)

    total_elapsed = time.time() - start_total

    # ── Pass criteria checks ──────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("PILOT PASS CRITERIA CHECKS")
    print("=" * 90)

    checks = {}

    # Check 1: All 15 configs complete 5 epochs
    ok_count = sum(1 for r in all_results.values() if 'error' not in r)
    checks['all_15_configs_complete'] = ok_count == total_configs
    print(f"{'PASS' if checks['all_15_configs_complete'] else 'FAIL'}: "
          f"All 15 configurations complete ({ok_count}/{total_configs})")

    # Check 2: SNR varies monotonically with batch size (rough trend)
    snr_checks = check_snr_monotonicity(all_results)
    monotonic_ok = sum(1 for c in snr_checks.values() if c['monotonic'])
    checks['snr_monotonic'] = monotonic_ok >= 2  # At least 2 of 3 methods show trend
    for method, c in snr_checks.items():
        status = "PASS" if c['monotonic'] else "FAIL"
        print(f"  {status}: {method} SNR monotonicity "
              f"(direction={c.get('dominant_direction','?')}, violations={c.get('violations','?')})")
        if 'snr_values' in c:
            for bs_val, snr_val in c['snr_values']:
                print(f"    bs={bs_val}: SNR={snr_val:.4f}")
    print(f"{'PASS' if checks['snr_monotonic'] else 'FAIL'}: "
          f"SNR monotonic in >= 2 methods ({monotonic_ok}/3)")

    overall_pass = all(checks.values())
    recommendation = "GO" if overall_pass else "NO_GO"

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "-" * 110)
    print(f"{'Config':<20} {'BS':>5} {'LR':>7} {'Method':<10} "
          f"{'Test%':>7} {'Train%':>8} {'Gap%':>7} {'SNR':>10} {'WD Bgt':>8} {'Time':>6}")
    print("-" * 110)

    for bs in BATCH_SIZES:
        for method in METHODS:
            key = f"{method}_bs{bs}"
            r = all_results.get(key, {})
            if 'error' in r:
                print(f"{key:<20} {bs:>5} {'':>7} {method:<10} ERROR: {r['error'][:40]}")
            elif r:
                lr_used = r.get('lr', 0)
                print(f"{key:<20} {bs:>5} {lr_used:>7.4f} {method:<10} "
                      f"{r['final_test_acc']:>7.2f} {r['final_train_acc']:>8.2f} "
                      f"{r['generalization_gap']:>7.2f} "
                      f"{r['alignment_snr']:>10.4f} "
                      f"{r['total_wd_budget']:>8.4f} "
                      f"{r['elapsed_sec']:>6.1f}")
    print("-" * 110)

    # ── Accuracy deltas (CWD - FixedWD) and (UDWDC - FixedWD) ────────────────
    print("\nAccuracy Deltas vs FixedWD:")
    print(f"{'BS':>6} {'CWD-Fixed':>12} {'UDWDC-Fixed':>12}")
    for bs in BATCH_SIZES:
        fixed_r = all_results.get(f"FixedWD_bs{bs}", {})
        cwd_r = all_results.get(f"CWD_bs{bs}", {})
        udwdc_r = all_results.get(f"UDWDC_bs{bs}", {})
        fixed_acc = fixed_r.get('final_test_acc', 0)
        cwd_delta = cwd_r.get('final_test_acc', 0) - fixed_acc if 'error' not in cwd_r else float('nan')
        udwdc_delta = udwdc_r.get('final_test_acc', 0) - fixed_acc if 'error' not in udwdc_r else float('nan')
        print(f"{bs:>6} {cwd_delta:>+12.2f} {udwdc_delta:>+12.2f}")

    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"Overall: {recommendation}")

    # ── Save results ──────────────────────────────────────────────────────────
    pilot_summary = {
        'overall_recommendation': recommendation,
        'task_id': TASK_ID,
        'model': MODEL,
        'dataset': DATASET,
        'pilot_epochs': PILOT_EPOCHS,
        'pilot_samples': PILOT_MAX_SAMPLES,
        'seed': SEED,
        'batch_sizes': BATCH_SIZES,
        'methods': METHODS,
        'lr_base': LR_BASE,
        'lr_ref_bs': LR_REF_BS,
        'pass_criteria': checks,
        'snr_monotonicity': {},
        'accuracy_deltas': {},
        'configs': {},
        'total_elapsed_sec': total_elapsed,
    }

    # SNR monotonicity detail
    for method, c in snr_checks.items():
        pilot_summary['snr_monotonicity'][method] = {
            'monotonic': c['monotonic'],
            'dominant_direction': c.get('dominant_direction', 'unknown'),
            'violations': c.get('violations', -1),
            'snr_values': {str(bs): snr for bs, snr in c.get('snr_values', [])},
        }

    # Accuracy deltas
    for bs in BATCH_SIZES:
        fixed_r = all_results.get(f"FixedWD_bs{bs}", {})
        fixed_acc = fixed_r.get('final_test_acc', 0)
        deltas = {}
        for m in ["CWD", "UDWDC"]:
            r = all_results.get(f"{m}_bs{bs}", {})
            if 'error' not in r:
                deltas[m] = r.get('final_test_acc', 0) - fixed_acc
        pilot_summary['accuracy_deltas'][str(bs)] = deltas

    # Per-config results (without epoch history for summary)
    for key, r in all_results.items():
        if 'error' not in r:
            pilot_summary['configs'][key] = {
                'method': r['method'],
                'batch_size': r['batch_size'],
                'lr': r['lr'],
                'final_test_acc': r['final_test_acc'],
                'final_train_acc': r['final_train_acc'],
                'generalization_gap': r['generalization_gap'],
                'alignment_snr': r['alignment_snr'],
                'total_wd_budget': r['total_wd_budget'],
                'elapsed_sec': r['elapsed_sec'],
            }
        else:
            pilot_summary['configs'][key] = {'error': r['error']}

    summary_file = Path(SAVE_DIR) / 'pilot_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(pilot_summary, f, indent=2)

    # Detailed results with epoch history
    detail_file = Path(SAVE_DIR) / 'pilot_detailed_results.json'
    detail_data = {}
    for key, r in all_results.items():
        if 'error' not in r:
            detail_data[key] = {
                'method': r['method'],
                'batch_size': r['batch_size'],
                'lr': r['lr'],
                'final_test_acc': r['final_test_acc'],
                'alignment_snr': r['alignment_snr'],
                'epoch_history': r['epoch_history'],
            }
    with open(detail_file, 'w') as f:
        json.dump(detail_data, f, indent=2)

    # Markdown summary
    md_file = Path(SAVE_DIR) / 'pilot_summary.md'
    with open(md_file, 'w') as f:
        f.write("# Pilot: Phase 2b - Batch-size Sweep (H3)\n\n")
        f.write(f"**Recommendation: {recommendation}**\n\n")
        f.write(f"- Model: {MODEL}, Dataset: {DATASET}\n")
        f.write(f"- Pilot: {PILOT_EPOCHS} epochs, {PILOT_MAX_SAMPLES} samples, Seed: {SEED}\n")
        f.write(f"- Batch sizes: {BATCH_SIZES}\n")
        f.write(f"- Methods: {METHODS}\n")
        f.write(f"- LR scaling: {LR_BASE} * (bs / {LR_REF_BS})\n")
        f.write(f"- Total time: {total_elapsed:.1f}s\n\n")
        f.write("## Pass Criteria\n\n")
        for check, passed in checks.items():
            f.write(f"- {'PASS' if passed else 'FAIL'}: {check}\n")
        f.write("\n## Results\n\n")
        f.write("| BS | Method | LR | Test (%) | SNR | WD Budget | Time (s) |\n")
        f.write("|----|--------|----|---------:|----:|----------:|---------:|\n")
        for bs in BATCH_SIZES:
            for method in METHODS:
                key = f"{method}_bs{bs}"
                r = all_results.get(key, {})
                if 'error' not in r and r:
                    f.write(f"| {bs} | {method} | {r['lr']:.4f} | "
                            f"{r['final_test_acc']:.2f} | {r['alignment_snr']:.4f} | "
                            f"{r['total_wd_budget']:.4f} | {r['elapsed_sec']:.1f} |\n")
                else:
                    f.write(f"| {bs} | {method} | - | ERROR | - | - | - |\n")
        f.write("\n## Accuracy Deltas vs FixedWD\n\n")
        f.write("| BS | CWD - Fixed | UDWDC - Fixed |\n")
        f.write("|----|------------|---------------|\n")
        for bs in BATCH_SIZES:
            deltas = pilot_summary['accuracy_deltas'].get(str(bs), {})
            cwd_d = deltas.get('CWD', float('nan'))
            udwdc_d = deltas.get('UDWDC', float('nan'))
            f.write(f"| {bs} | {cwd_d:+.2f} | {udwdc_d:+.2f} |\n")

    print(f"\nResults saved to: {SAVE_DIR}")

    # ── DONE markers ──────────────────────────────────────────────────────────
    pid_file.unlink(missing_ok=True)

    # Primary DONE marker
    done_file = Path(RESULTS_BASE) / f"{TASK_ID}_DONE"
    done_file.write_text(json.dumps({
        'task_id': TASK_ID,
        'status': 'success' if overall_pass else 'partial',
        'summary': (f"Batchsize sweep pilot: {recommendation}. "
                    f"{ok_count}/{total_configs} configs complete. "
                    f"SNR monotonic in {monotonic_ok}/3 methods."),
        'final_progress': {
            'recommendation': recommendation,
            'configs_completed': ok_count,
            'checks': checks,
        },
        'timestamp': datetime.now().isoformat(),
    }))

    # Secondary DONE marker in pilot directory
    done_file2 = Path(SAVE_DIR) / f"{TASK_ID}_DONE"
    done_file2.write_text(json.dumps({
        'task_id': TASK_ID,
        'status': 'success' if overall_pass else 'partial',
        'timestamp': datetime.now().isoformat(),
    }))

    return overall_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
