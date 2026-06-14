#!/usr/bin/env python3
"""FULL experiment: batchsize_sweep — Phase 2b batch-size sweep for H3.

Train VGG-16-BN on CIFAR-100 for 200 epochs at batch sizes
{64, 128, 256, 512, 1024} with 3 methods (FixedWD, CWD, UDWDC-v2)
and 3 seeds (42, 123, 456).

Purpose: Test H3 — binary vs continuous alignment signal quality
depends on batch size. Track alignment SNR = E[alpha]^2 / Var[alpha].

LR scaled linearly: LR = LR_base * (bs / 128).
H3 falsified if continuous beats binary at batch_size=64.

Optimization: Only collect per-batch alignment alpha every DIAG_INTERVAL
batches (not every batch) to reduce overhead. Full diagnostic snapshot
taken once per epoch at the end.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n sibyl_dynamic-wd python full_batchsize_sweep.py
"""

import json
import os
import sys
import time
import math
import gc
import traceback
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from optimizers import create_optimizer
from data import get_cifar100_loaders
from diagnostics.logger import DiagnosticLogger

# ── Task Configuration ──
TASK_ID = "batchsize_sweep"
TOTAL_EPOCHS = 200
SEEDS = [42, 123, 456]
LR_BASE = 0.1
LR_REF_BS = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
MODEL_NAME = "vgg16_bn"
DATASET = "cifar100"
NUM_CLASSES = 100

BATCH_SIZES = [64, 128, 256, 512, 1024]
METHODS = ["FixedWD", "CWD", "UDWDC-v2"]

# UDWDC-v2 gains
UDWDC_KP = 0.5
UDWDC_KI = 0.1
UDWDC_KD = 0.3

# Diagnostic sampling interval: collect alignment alpha every N batches.
# At bs=64 we have ~781 batches/epoch; sampling every 10 gives ~78 samples/epoch,
# still enough for reliable SNR estimation, while 10x faster.
DIAG_INTERVAL = 10

WORKSPACE = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current")
SAVE_DIR = WORKSPACE / "exp" / "results" / "full" / "phase2_batchsize"
RESULTS_DIR = WORKSPACE / "exp" / "results"
CODE_DIR = WORKSPACE / "exp" / "code"
DATA_ROOT = str(CODE_DIR / "data")


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_lr(optimizer_wrapper, epoch, total_epochs, base_lr):
    """Cosine annealing LR schedule."""
    lr = base_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    optimizer_wrapper.set_lr(lr)
    return lr


def compute_batch_alpha_fast(optimizer):
    """Compute mean alignment alpha across layers, lightweight version.

    Only computes the cosine similarity for weight layers (apply_wd=True)
    that have gradients. Avoids full diagnostic computation.
    """
    alphas = []
    for group in optimizer.param_groups:
        if not group['apply_wd']:
            continue
        param = group['params'][0]
        if param.grad is None:
            continue
        w = param.data
        g = param.grad.data
        w_norm = torch.norm(w).item()
        g_norm = torch.norm(g).item()
        if w_norm > 1e-8 and g_norm > 1e-8:
            dot = torch.dot(w.flatten(), g.flatten()).item()
            alpha = dot / (w_norm * g_norm)
            alphas.append(alpha)
    return float(np.mean(alphas)) if alphas else 0.0


def train_one_epoch(model, loader, optimizer, criterion, device, diag_interval=10):
    """Train one epoch. Returns (loss, accuracy, sampled per-batch alpha list).

    Only collects alignment alpha every diag_interval batches to reduce overhead.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_alphas = []

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Sample alignment alpha every N batches (before step modifies grads)
        if batch_idx % diag_interval == 0:
            alpha = compute_batch_alpha_fast(optimizer)
            batch_alphas.append(alpha)

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / max(total, 1), 100.0 * correct / max(total, 1), batch_alphas


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on test set."""
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
    """Compute alignment SNR = E[alpha]^2 / Var[alpha]."""
    if not all_batch_alphas or len(all_batch_alphas) < 2:
        return float('nan')
    arr = np.array(all_batch_alphas)
    mean_alpha = np.mean(arr)
    var_alpha = np.var(arr)
    if var_alpha < 1e-12:
        return float('inf')
    return float(mean_alpha ** 2 / var_alpha)


def run_single(method, seed, batch_size, lr, device, run_idx, total_runs):
    """Run one (method, seed, batch_size) configuration for 200 epochs.

    Returns dict with results and alignment SNR.
    """
    set_seed(seed)

    # Data loader with the specific batch size
    train_loader, test_loader = get_cifar100_loaders(
        batch_size=batch_size, data_root=DATA_ROOT, seed=seed, num_workers=4
    )

    model = create_model(MODEL_NAME, num_classes=NUM_CLASSES).to(device)

    opt_kwargs = {}
    if method == "UDWDC-v2":
        opt_kwargs = {'K_p': UDWDC_KP, 'K_i': UDWDC_KI, 'K_d': UDWDC_KD}

    wd_optimizer = create_optimizer(
        method, model, lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
        **opt_kwargs
    )

    criterion = nn.CrossEntropyLoss()

    # Create a safe label for file naming
    method_safe = method.replace("-", "_")
    config_label = f"{method_safe}_bs{batch_size}_seed{seed}"
    run_save_dir = str(SAVE_DIR / config_label)

    logger = DiagnosticLogger(
        save_dir=run_save_dir,
        method=f"{method}_bs{batch_size}",
        seed=seed,
        model_name=MODEL_NAME,
        dataset=DATASET
    )

    t0 = time.time()
    best_acc = 0.0
    all_batch_alphas = []  # Accumulate across all epochs for SNR
    epoch_accuracies = []

    # Per-epoch alignment stats
    per_epoch_alpha_stats = []

    for epoch in range(TOTAL_EPOCHS):
        lr_epoch = cosine_lr(wd_optimizer, epoch, TOTAL_EPOCHS, lr)

        train_loss, train_acc, batch_alphas = train_one_epoch(
            model, train_loader, wd_optimizer, criterion, device,
            diag_interval=DIAG_INTERVAL
        )
        all_batch_alphas.extend(batch_alphas)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Full diagnostic snapshot once per epoch (not per batch)
        diagnostics = wd_optimizer.get_diagnostics()
        logger.log_epoch(epoch, diagnostics, train_loss, test_loss, train_acc, test_acc, lr=lr_epoch)

        # Alignment stats for this epoch
        alpha_values = [m['alpha_t'] for m in diagnostics.values() if 'alpha_t' in m]
        wd_values = [m['effective_wd'] for m in diagnostics.values()]
        epoch_mean_alpha = float(np.mean(alpha_values)) if alpha_values else 0.0
        epoch_std_alpha = float(np.std(alpha_values)) if alpha_values else 0.0

        # Per-epoch SNR (from sampled batch alphas)
        epoch_snr = compute_alignment_snr(batch_alphas)
        per_epoch_alpha_stats.append({
            'epoch': epoch,
            'mean_alpha': epoch_mean_alpha,
            'std_alpha': epoch_std_alpha,
            'snr': epoch_snr,
            'n_samples': len(batch_alphas),
        })

        epoch_accuracies.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'lr': lr_epoch,
            'mean_effective_wd': float(np.mean(wd_values)),
            'mean_alpha': epoch_mean_alpha,
        })

        # UDWDC-v2 epoch check
        if method == "UDWDC-v2" and hasattr(wd_optimizer, 'end_epoch_check'):
            wd_optimizer.end_epoch_check()

        if test_acc > best_acc:
            best_acc = test_acc

        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - t0
            eta_sec = elapsed / (epoch + 1) * (TOTAL_EPOCHS - epoch - 1)
            print(f"  [{config_label}] Epoch {epoch+1}/{TOTAL_EPOCHS} | "
                  f"LR: {lr_epoch:.6f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | "
                  f"Best: {best_acc:.2f}% | ETA: {eta_sec/60:.1f}min "
                  f"[run {run_idx}/{total_runs}]", flush=True)

    logger.save()
    elapsed = time.time() - t0

    # Compute full-training alignment SNR
    full_snr = compute_alignment_snr(all_batch_alphas)

    # Compute late-training SNR (last 25% of epochs)
    late_start = int(0.75 * len(per_epoch_alpha_stats))
    batch_offset = 0
    late_batch_alphas = []
    for i, stats in enumerate(per_epoch_alpha_stats):
        n = stats['n_samples']
        if i >= late_start:
            late_batch_alphas.extend(all_batch_alphas[batch_offset:batch_offset + n])
        batch_offset += n
    late_snr = compute_alignment_snr(late_batch_alphas)

    # Total WD budget
    total_wd_budget = logger.get_total_wd_budget()
    udwdc_v2_cumulative = None
    if method == "UDWDC-v2" and hasattr(wd_optimizer, 'get_cumulative_wd_budget'):
        udwdc_v2_cumulative = wd_optimizer.get_cumulative_wd_budget()
        total_wd_budget = udwdc_v2_cumulative

    gen_gap = logger.get_generalization_gap()

    result = {
        'method': method,
        'seed': seed,
        'batch_size': batch_size,
        'lr': lr,
        'best_acc': best_acc,
        'final_test_acc': test_acc,
        'final_train_acc': train_acc,
        'generalization_gap': gen_gap,
        'alignment_snr_full': full_snr,
        'alignment_snr_late': late_snr,
        'total_wd_budget': total_wd_budget,
        'udwdc_v2_cumulative_budget': udwdc_v2_cumulative,
        'elapsed_sec': elapsed,
        'epochs': TOTAL_EPOCHS,
        'n_batch_alphas': len(all_batch_alphas),
    }

    # Save detailed epoch data
    detail_file = Path(run_save_dir) / "epoch_detail.json"
    with open(detail_file, 'w') as f:
        json.dump({
            'method': method,
            'seed': seed,
            'batch_size': batch_size,
            'lr': lr,
            'model': MODEL_NAME,
            'dataset': DATASET,
            'epochs': TOTAL_EPOCHS,
            'best_acc': best_acc,
            'alignment_snr_full': full_snr,
            'alignment_snr_late': late_snr,
            'total_wd_budget': total_wd_budget,
            'epoch_accuracies': epoch_accuracies,
            'per_epoch_alpha_stats': per_epoch_alpha_stats,
            'elapsed_sec': elapsed,
        }, f, indent=2)

    return result


def write_progress(task_id, results_dir, run_idx, total_runs, metric=None):
    """Write progress file for system monitor."""
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        'task_id': task_id,
        'epoch': run_idx,
        'total_epochs': total_runs,
        'step': run_idx,
        'total_steps': total_runs,
        'loss': None,
        'metric': metric or {},
        'updated_at': datetime.now().isoformat(),
    }))


def mark_task_done(task_id, results_dir, status="success", summary=""):
    """Write DONE marker file for system monitor."""
    pid_file = Path(results_dir) / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        'task_id': task_id,
        'status': status,
        'summary': summary,
        'final_progress': final_progress,
        'timestamp': datetime.now().isoformat(),
    }))


def update_gpu_progress(results_dir, task_id, planned_min, actual_min,
                        start_time, end_time, config_snapshot, status="completed"):
    """Update gpu_progress.json after task completion."""
    gpu_progress_file = Path(results_dir).parent / "gpu_progress.json"
    try:
        if gpu_progress_file.exists():
            with open(gpu_progress_file) as f:
                gp = json.load(f)
        else:
            gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

        if status == "completed":
            if task_id not in gp.get("completed", []):
                gp.setdefault("completed", []).append(task_id)
        elif status == "failed":
            if task_id not in gp.get("failed", []):
                gp.setdefault("failed", []).append(task_id)

        # Remove from running
        gp.setdefault("running", {})
        if task_id in gp["running"]:
            del gp["running"][task_id]

        # Record timing
        gp.setdefault("timings", {})[task_id] = {
            "planned_min": planned_min,
            "actual_min": actual_min,
            "start_time": start_time,
            "end_time": end_time,
            "config_snapshot": config_snapshot,
        }

        with open(gpu_progress_file, "w") as f:
            json.dump(gp, f, indent=2)

    except Exception as e:
        print(f"Warning: Failed to update gpu_progress.json: {e}")


def main():
    total_runs = len(BATCH_SIZES) * len(METHODS) * len(SEEDS)  # 5 * 3 * 3 = 45

    print("=" * 90)
    print("FULL EXPERIMENT: batchsize_sweep — Phase 2b")
    print(f"  Model: {MODEL_NAME} on {DATASET}")
    print(f"  Epochs: {TOTAL_EPOCHS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Batch sizes: {BATCH_SIZES}")
    print(f"  Methods: {', '.join(METHODS)}")
    print(f"  Total runs: {total_runs}")
    print(f"  LR scaling: {LR_BASE} * (bs / {LR_REF_BS})")
    print(f"  Diagnostic interval: every {DIAG_INTERVAL} batches")
    print("=" * 90, flush=True)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Write PID file
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        mem = torch.cuda.get_device_properties(0)
        total_mem = getattr(mem, 'total_mem', None) or getattr(mem, 'total_memory', 0)
        print(f"Memory: {total_mem / 1e9:.1f} GB", flush=True)

    # Enable cudnn benchmark for faster training (since input size is fixed)
    torch.backends.cudnn.benchmark = True

    all_results = []
    start_time = datetime.now()
    overall_start = time.time()
    run_idx = 0
    n_failed = 0

    # Iterate: batch_size -> method -> seed
    for bs in BATCH_SIZES:
        lr = LR_BASE * (bs / LR_REF_BS)
        print(f"\n{'#' * 80}")
        print(f"# Batch size: {bs}, LR: {lr:.4f}")
        print(f"{'#' * 80}", flush=True)

        for method in METHODS:
            for seed in SEEDS:
                run_idx += 1
                method_safe = method.replace("-", "_")
                config_label = f"{method_safe}_bs{bs}_seed{seed}"

                print(f"\n{'=' * 70}")
                print(f"[{run_idx}/{total_runs}] {config_label} (LR={lr:.4f})")
                print(f"{'=' * 70}", flush=True)

                # Check if this run already completed (resume support)
                run_detail = SAVE_DIR / config_label / "epoch_detail.json"
                if run_detail.exists():
                    try:
                        with open(run_detail) as f:
                            existing = json.load(f)
                        if existing.get('epochs') == TOTAL_EPOCHS and existing.get('best_acc', 0) > 0:
                            print(f"  SKIP (already completed): Best={existing['best_acc']:.2f}%", flush=True)
                            result = {
                                'method': method,
                                'seed': seed,
                                'batch_size': bs,
                                'lr': lr,
                                'best_acc': existing['best_acc'],
                                'final_test_acc': existing.get('epoch_accuracies', [{}])[-1].get('test_acc', 0),
                                'final_train_acc': existing.get('epoch_accuracies', [{}])[-1].get('train_acc', 0),
                                'generalization_gap': (existing.get('epoch_accuracies', [{}])[-1].get('train_acc', 0) -
                                                       existing.get('epoch_accuracies', [{}])[-1].get('test_acc', 0)),
                                'alignment_snr_full': existing.get('alignment_snr_full', float('nan')),
                                'alignment_snr_late': existing.get('alignment_snr_late', float('nan')),
                                'total_wd_budget': existing.get('total_wd_budget', 0),
                                'udwdc_v2_cumulative_budget': existing.get('udwdc_v2_cumulative_budget'),
                                'elapsed_sec': existing.get('elapsed_sec', 0),
                                'epochs': TOTAL_EPOCHS,
                                'resumed': True,
                            }
                            all_results.append(result)
                            write_progress(TASK_ID, str(RESULTS_DIR), run_idx, total_runs, {
                                'config': config_label, 'status': 'skipped (completed)',
                                'best_acc': result['best_acc'],
                            })
                            continue
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass  # Corrupted file, re-run

                try:
                    result = run_single(method, seed, bs, lr, device, run_idx, total_runs)
                    all_results.append(result)

                    extra = ""
                    if method == "UDWDC-v2" and result.get('udwdc_v2_cumulative_budget') is not None:
                        extra = f" | WD={result['udwdc_v2_cumulative_budget']:.4e}"

                    print(f"  DONE: {config_label} | Best: {result['best_acc']:.2f}% | "
                          f"Final: {result['final_test_acc']:.2f}% | "
                          f"SNR: {result['alignment_snr_full']:.4f} | "
                          f"Time: {result['elapsed_sec']:.1f}s{extra}", flush=True)

                    write_progress(TASK_ID, str(RESULTS_DIR), run_idx, total_runs, {
                        'config': config_label,
                        'best_acc': result['best_acc'],
                        'snr': result['alignment_snr_full'],
                    })

                except Exception as e:
                    print(f"  FAILED: {config_label} — {e}", flush=True)
                    traceback.print_exc()
                    all_results.append({
                        'method': method, 'seed': seed, 'batch_size': bs,
                        'status': 'failed', 'error': str(e),
                    })
                    n_failed += 1
                    write_progress(TASK_ID, str(RESULTS_DIR), run_idx, total_runs, {
                        'config': config_label, 'status': 'failed', 'error': str(e),
                    })

                # Free GPU memory between runs
                torch.cuda.empty_cache()
                gc.collect()

    total_elapsed = time.time() - overall_start
    end_time = datetime.now()

    # ── Aggregate Results ──
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90, flush=True)

    # Group by (method, batch_size), compute mean+/-std across seeds
    config_stats = {}
    for bs in BATCH_SIZES:
        for method in METHODS:
            key = f"{method}_bs{bs}"
            runs = [r for r in all_results
                    if r.get('method') == method and r.get('batch_size') == bs
                    and 'best_acc' in r]
            if not runs:
                continue

            best_accs = [r['best_acc'] for r in runs]
            final_accs = [r['final_test_acc'] for r in runs]
            gen_gaps = [r['generalization_gap'] for r in runs]
            snrs_full = [r['alignment_snr_full'] for r in runs
                         if not math.isnan(r.get('alignment_snr_full', float('nan')))]
            snrs_late = [r['alignment_snr_late'] for r in runs
                         if not math.isnan(r.get('alignment_snr_late', float('nan')))]
            wd_budgets = [r.get('total_wd_budget', 0) for r in runs]

            config_stats[key] = {
                'method': method,
                'batch_size': bs,
                'lr': LR_BASE * (bs / LR_REF_BS),
                'best_acc_mean': float(np.mean(best_accs)),
                'best_acc_std': float(np.std(best_accs)),
                'final_acc_mean': float(np.mean(final_accs)),
                'final_acc_std': float(np.std(final_accs)),
                'gen_gap_mean': float(np.mean(gen_gaps)),
                'gen_gap_std': float(np.std(gen_gaps)),
                'snr_full_mean': float(np.mean(snrs_full)) if snrs_full else float('nan'),
                'snr_full_std': float(np.std(snrs_full)) if snrs_full else float('nan'),
                'snr_late_mean': float(np.mean(snrs_late)) if snrs_late else float('nan'),
                'snr_late_std': float(np.std(snrs_late)) if snrs_late else float('nan'),
                'wd_budget_mean': float(np.mean(wd_budgets)),
                'wd_budget_std': float(np.std(wd_budgets)),
                'n_seeds': len(runs),
            }

    # Print summary table
    print(f"\n{'Config':<22} {'Best Acc':>14} {'Final Acc':>14} {'SNR (full)':>14} {'SNR (late)':>14} {'WD Budget':>14}")
    print("-" * 100)
    for bs in BATCH_SIZES:
        for method in METHODS:
            key = f"{method}_bs{bs}"
            if key not in config_stats:
                print(f"{key:<22} {'FAILED':>14}")
                continue
            s = config_stats[key]
            snr_f = f"{s['snr_full_mean']:.4f}" if not math.isnan(s['snr_full_mean']) else "NaN"
            snr_l = f"{s['snr_late_mean']:.4f}" if not math.isnan(s['snr_late_mean']) else "NaN"
            print(f"{key:<22} "
                  f"{s['best_acc_mean']:>6.2f}+/-{s['best_acc_std']:.2f} "
                  f"{s['final_acc_mean']:>6.2f}+/-{s['final_acc_std']:.2f} "
                  f"{snr_f:>14} "
                  f"{snr_l:>14} "
                  f"{s['wd_budget_mean']:>10.2f}+/-{s['wd_budget_std']:.1f}")
    print("-" * 100, flush=True)

    # ── Accuracy deltas vs FixedWD ──
    print("\nAccuracy Deltas vs FixedWD (mean across seeds):")
    print(f"{'BS':>6} {'CWD-Fixed':>14} {'UDWDC-v2-Fixed':>16}")
    accuracy_deltas = {}
    for bs in BATCH_SIZES:
        fixed_key = f"FixedWD_bs{bs}"
        fixed_acc = config_stats.get(fixed_key, {}).get('best_acc_mean', 0)
        deltas = {}
        for method in ["CWD", "UDWDC-v2"]:
            key = f"{method}_bs{bs}"
            if key in config_stats:
                deltas[method] = config_stats[key]['best_acc_mean'] - fixed_acc
            else:
                deltas[method] = float('nan')

        cwd_d = deltas.get('CWD', float('nan'))
        udwdc_d = deltas.get('UDWDC-v2', float('nan'))
        print(f"{bs:>6} {cwd_d:>+14.2f} {udwdc_d:>+16.2f}")
        accuracy_deltas[str(bs)] = deltas

    # ── H3 Hypothesis Check ──
    print("\n" + "=" * 90)
    print("H3 HYPOTHESIS CHECK")
    print("=" * 90, flush=True)

    h3_result = "INCONCLUSIVE"
    bs64_cwd = config_stats.get("CWD_bs64", {}).get('best_acc_mean', 0)
    bs64_udwdc = config_stats.get("UDWDC-v2_bs64", {}).get('best_acc_mean', 0)
    bs64_fixed = config_stats.get("FixedWD_bs64", {}).get('best_acc_mean', 0)

    if bs64_cwd > 0 and bs64_udwdc > 0:
        if bs64_udwdc > bs64_cwd:
            h3_result = "FALSIFIED"
            print(f"H3 FALSIFIED: UDWDC-v2 ({bs64_udwdc:.2f}%) > CWD ({bs64_cwd:.2f}%) at bs=64")
        else:
            bs1024_cwd = config_stats.get("CWD_bs1024", {}).get('best_acc_mean', 0)
            bs1024_udwdc = config_stats.get("UDWDC-v2_bs1024", {}).get('best_acc_mean', 0)
            if bs1024_udwdc >= bs1024_cwd and bs64_cwd >= bs64_udwdc:
                h3_result = "SUPPORTED"
                print(f"H3 SUPPORTED: CWD ({bs64_cwd:.2f}%) >= UDWDC-v2 ({bs64_udwdc:.2f}%) at bs=64, "
                      f"but UDWDC-v2 ({bs1024_udwdc:.2f}%) >= CWD ({bs1024_cwd:.2f}%) at bs=1024")
            else:
                h3_result = "PARTIAL"
                print(f"H3 PARTIAL: CWD ({bs64_cwd:.2f}%) >= UDWDC-v2 ({bs64_udwdc:.2f}%) at bs=64, "
                      f"UDWDC-v2 ({bs1024_udwdc:.2f}%) vs CWD ({bs1024_cwd:.2f}%) at bs=1024")
    else:
        print("H3: Insufficient data to evaluate")

    # SNR trends
    print("\nAlignment SNR trends (full training):")
    for method in METHODS:
        snr_values = []
        for bs in BATCH_SIZES:
            key = f"{method}_bs{bs}"
            s = config_stats.get(key, {})
            snr_val = s.get('snr_full_mean', float('nan'))
            snr_values.append((bs, snr_val))
        print(f"  {method}: {' | '.join(f'bs={bs}:{snr:.4f}' for bs, snr in snr_values)}")

    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)", flush=True)
    print(f"Runs completed: {total_runs - n_failed}/{total_runs}", flush=True)

    # ── Save Summary JSON ──
    summary = {
        'task_id': TASK_ID,
        'mode': 'full',
        'model': MODEL_NAME,
        'dataset': DATASET,
        'epochs': TOTAL_EPOCHS,
        'seeds': SEEDS,
        'batch_sizes': BATCH_SIZES,
        'methods': METHODS,
        'lr_base': LR_BASE,
        'lr_ref_bs': LR_REF_BS,
        'momentum': MOMENTUM,
        'weight_decay': WEIGHT_DECAY,
        'diag_interval': DIAG_INTERVAL,
        'total_runs': total_runs,
        'n_completed': total_runs - n_failed,
        'n_failed': n_failed,
        'total_elapsed_sec': total_elapsed,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'h3_result': h3_result,
        'config_stats': config_stats,
        'accuracy_deltas': accuracy_deltas,
        'all_results': [],
    }

    for r in all_results:
        summary['all_results'].append({k: v for k, v in r.items()
                                       if k != 'epoch_history'})

    summary_file = SAVE_DIR / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_file}", flush=True)

    # ── Save Summary Markdown ──
    md_lines = [
        "# Phase 2b: Full Batch-Size Sweep Results (H3)",
        "",
        f"**Task ID**: {TASK_ID}",
        f"**Model**: {MODEL_NAME}",
        f"**Dataset**: {DATASET}",
        f"**Epochs**: {TOTAL_EPOCHS}",
        f"**Seeds**: {', '.join(str(s) for s in SEEDS)}",
        f"**Batch sizes**: {', '.join(str(bs) for bs in BATCH_SIZES)}",
        f"**Methods**: {', '.join(METHODS)}",
        f"**Total runs**: {total_runs} ({total_runs - n_failed} completed)",
        f"**Total time**: {total_elapsed/3600:.2f} hours",
        f"**H3 result**: {h3_result}",
        "",
        "## Method Comparison (mean +/- std across 3 seeds)",
        "",
        "| BS | Method | LR | Best Acc (%) | Final Acc (%) | Gen Gap (%) | SNR (full) | SNR (late) | WD Budget |",
        "|----|--------|----|-----------:|-------------:|----------:|----------:|----------:|---------:|",
    ]

    for bs in BATCH_SIZES:
        for method in METHODS:
            key = f"{method}_bs{bs}"
            if key not in config_stats:
                md_lines.append(f"| {bs} | {method} | - | FAILED | - | - | - | - | - |")
                continue
            s = config_stats[key]
            snr_f = f"{s['snr_full_mean']:.4f}" if not math.isnan(s['snr_full_mean']) else "NaN"
            snr_l = f"{s['snr_late_mean']:.4f}" if not math.isnan(s['snr_late_mean']) else "NaN"
            md_lines.append(
                f"| {bs} | {method} | {s['lr']:.4f} | "
                f"{s['best_acc_mean']:.2f}+/-{s['best_acc_std']:.2f} | "
                f"{s['final_acc_mean']:.2f}+/-{s['final_acc_std']:.2f} | "
                f"{s['gen_gap_mean']:.2f}+/-{s['gen_gap_std']:.2f} | "
                f"{snr_f} | {snr_l} | "
                f"{s['wd_budget_mean']:.2f}+/-{s['wd_budget_std']:.1f} |"
            )

    md_lines += [
        "",
        "## Accuracy Deltas vs FixedWD",
        "",
        "| BS | CWD - Fixed | UDWDC-v2 - Fixed |",
        "|----|------------|-----------------|",
    ]
    for bs in BATCH_SIZES:
        deltas = accuracy_deltas.get(str(bs), {})
        cwd_d = deltas.get('CWD', float('nan'))
        udwdc_d = deltas.get('UDWDC-v2', float('nan'))
        md_lines.append(f"| {bs} | {cwd_d:+.2f} | {udwdc_d:+.2f} |")

    md_lines += [
        "",
        "## H3 Hypothesis Assessment",
        "",
        f"**Result: {h3_result}**",
        "",
        "H3 predicts that binary masking (CWD) is preferred at small batch sizes (b<=256) ",
        "while continuous modulation (UDWDC-v2) provides marginal improvement at large batch sizes (b>=1024).",
        "",
        f"- At bs=64: CWD={bs64_cwd:.2f}%, UDWDC-v2={bs64_udwdc:.2f}%, FixedWD={bs64_fixed:.2f}%",
    ]

    # Add SNR trend analysis
    md_lines += [
        "",
        "## Alignment SNR Trends",
        "",
        "| BS | FixedWD SNR | CWD SNR | UDWDC-v2 SNR |",
        "|----|-----------|---------|-------------|",
    ]
    for bs in BATCH_SIZES:
        row_vals = []
        for method in METHODS:
            key = f"{method}_bs{bs}"
            s = config_stats.get(key, {})
            snr = s.get('snr_full_mean', float('nan'))
            row_vals.append(f"{snr:.4f}" if not math.isnan(snr) else "NaN")
        md_lines.append(f"| {bs} | {' | '.join(row_vals)} |")

    md_file = SAVE_DIR / "summary.md"
    md_file.write_text("\n".join(md_lines))
    print(f"Saved: {md_file}", flush=True)

    # ── Write DONE marker ──
    status = "success" if n_failed == 0 else "partial"
    mark_task_done(
        TASK_ID, str(RESULTS_DIR), status=status,
        summary=(f"Phase 2b batchsize sweep: {total_runs - n_failed}/{total_runs} runs. "
                 f"H3={h3_result}. VGG-16-BN CIFAR-100, 200 epochs, 5 batch sizes, "
                 f"3 methods, 3 seeds. Time: {total_elapsed/3600:.2f}h.")
    )
    print("Wrote DONE marker", flush=True)

    # ── Update GPU Progress ──
    update_gpu_progress(
        str(RESULTS_DIR), TASK_ID,
        planned_min=60,
        actual_min=int(total_elapsed / 60),
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        config_snapshot={
            'model': MODEL_NAME,
            'batch_size': f"{BATCH_SIZES}",
            'dataset': DATASET,
            'epochs': TOTAL_EPOCHS,
            'n_methods': len(METHODS),
            'n_seeds': len(SEEDS),
            'n_batch_sizes': len(BATCH_SIZES),
            'total_runs': total_runs,
            'gpu_model': 'RTX PRO 6000 Blackwell',
            'gpu_count': 1,
        },
        status=status,
    )
    print("Updated gpu_progress.json", flush=True)

    return n_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
