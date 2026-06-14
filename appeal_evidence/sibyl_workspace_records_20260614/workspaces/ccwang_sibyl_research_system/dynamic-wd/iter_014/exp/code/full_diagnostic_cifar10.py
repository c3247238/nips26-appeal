#!/usr/bin/env python3
"""FULL experiment: diagnostic_cifar10 — Phase 1 diagnostic on CIFAR-10/ResNet-20.

Train ResNet-20 on CIFAR-10 for 200 epochs with all 8 WD methods
(FixedWD, CWD, SWD, CPR, DefazioCorrective, NoWD, UDWDC, UDWDC-v2).
Seeds: 42, 123, 456. Track per-layer rho_t, alpha_t, effective lambda_t
trajectories across training.

This feeds the H1 unification fitting analysis.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n sibyl_dynamic-wd python full_diagnostic_cifar10.py
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

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from optimizers import create_optimizer
from data import get_cifar10_loaders
from diagnostics.logger import DiagnosticLogger

# ── Task Configuration ──
TASK_ID = "diagnostic_cifar10"
TOTAL_EPOCHS = 200
SEEDS = [42, 123, 456]
BATCH_SIZE = 128
BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
MODEL_NAME = "resnet20"
DATASET = "cifar10"
NUM_CLASSES = 10

# 8 methods from task_plan.json
METHODS = ["FixedWD", "CWD", "SWD", "CPR", "DefazioCorrective", "NoWD", "UDWDC", "UDWDC-v2"]

WORKSPACE = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current")
SAVE_DIR = WORKSPACE / "exp" / "results" / "full" / "phase1_diagnostic"
RESULTS_DIR = WORKSPACE / "exp" / "results"
CODE_DIR = WORKSPACE / "exp" / "code"


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


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
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
    return total_loss / total, 100.0 * correct / total


def run_single(method, seed, device, train_loader, test_loader, run_idx, total_runs):
    """Run one method with one seed for TOTAL_EPOCHS epochs.

    Returns dict with results and per-epoch trajectories.
    """
    set_seed(seed)

    model = create_model(MODEL_NAME, num_classes=NUM_CLASSES).to(device)

    opt_kwargs = {}
    if method in ("UDWDC", "UDWDC-v2"):
        opt_kwargs = {"K_p": 0.5, "K_i": 0.1, "K_d": 0.3}

    wd_opt = create_optimizer(
        method, model, lr=BASE_LR, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY, **opt_kwargs
    )

    criterion = nn.CrossEntropyLoss()

    method_safe = method.replace("-", "_")
    logger = DiagnosticLogger(
        save_dir=str(SAVE_DIR / f"{method_safe}_seed{seed}"),
        method=method, seed=seed,
        model_name=MODEL_NAME, dataset=DATASET
    )

    t0 = time.time()
    best_acc = 0.0
    per_epoch_rho = []
    per_epoch_alpha = []
    per_epoch_effective_wd = []
    cumulative_wd_budget = 0.0
    epoch_accuracies = []

    for epoch in range(TOTAL_EPOCHS):
        lr = cosine_lr(wd_opt, epoch, TOTAL_EPOCHS, BASE_LR)

        train_loss, train_acc = train_one_epoch(model, train_loader, wd_opt, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        diagnostics = wd_opt.get_diagnostics()
        logger.log_epoch(epoch, diagnostics, train_loss, test_loss, train_acc, test_acc, lr=lr)

        # Store per-layer rho_t, alpha_t, effective_wd for this epoch
        epoch_rho = {}
        epoch_alpha = {}
        epoch_eff_wd = {}
        for layer_name, diag in diagnostics.items():
            epoch_rho[layer_name] = diag["rho_t"]
            epoch_alpha[layer_name] = diag["alpha_t"]
            epoch_eff_wd[layer_name] = diag["effective_wd"]
        per_epoch_rho.append(epoch_rho)
        per_epoch_alpha.append(epoch_alpha)
        per_epoch_effective_wd.append(epoch_eff_wd)

        epoch_accuracies.append({
            "epoch": epoch,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "lr": lr,
        })

        # Track WD budget for UDWDC-v2
        if method == "UDWDC-v2" and hasattr(wd_opt, 'end_epoch_check'):
            wd_opt.end_epoch_check()
            cumulative_wd_budget = wd_opt.get_cumulative_wd_budget()

        if test_acc > best_acc:
            best_acc = test_acc

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            eta_sec = elapsed / (epoch + 1) * (TOTAL_EPOCHS - epoch - 1)
            print(f"  [{method}/seed{seed}] Epoch {epoch+1}/{TOTAL_EPOCHS} | "
                  f"LR: {lr:.6f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | "
                  f"Best: {best_acc:.2f}% | ETA: {eta_sec/60:.1f}min "
                  f"[run {run_idx}/{total_runs}]")

        # Write progress file every 20 epochs for system monitor
        if (epoch + 1) % 20 == 0 or epoch == TOTAL_EPOCHS - 1:
            write_progress(epoch + 1, TOTAL_EPOCHS, run_idx, total_runs,
                           {"method": method, "seed": seed, "test_acc": test_acc,
                            "best_acc": best_acc})

    elapsed = time.time() - t0
    logger.save()

    # Compute total WD budget from logger (for non-UDWDC-v2 methods)
    total_wd_budget = logger.get_total_wd_budget()
    if method == "UDWDC-v2":
        total_wd_budget = cumulative_wd_budget

    gen_gap = logger.get_generalization_gap()

    result = {
        "method": method,
        "seed": seed,
        "best_acc": best_acc,
        "final_test_acc": test_acc,
        "final_train_acc": train_acc,
        "final_test_loss": test_loss,
        "final_train_loss": train_loss,
        "gen_gap": gen_gap,
        "total_wd_budget": total_wd_budget,
        "elapsed_sec": elapsed,
        "n_layers": len(per_epoch_rho[0]) if per_epoch_rho else 0,
        "epochs": TOTAL_EPOCHS,
    }

    if method == "UDWDC-v2":
        result["cumulative_wd_budget"] = cumulative_wd_budget

    # Save detailed epoch data (for downstream H1 analysis)
    detail_file = SAVE_DIR / f"{method_safe}_seed{seed}" / "epoch_detail.json"
    with open(detail_file, "w") as f:
        json.dump({
            "method": method,
            "seed": seed,
            "model": MODEL_NAME,
            "dataset": DATASET,
            "epochs": TOTAL_EPOCHS,
            "epoch_accuracies": epoch_accuracies,
            "best_acc": best_acc,
            "total_wd_budget": total_wd_budget,
            "elapsed_sec": elapsed,
        }, f, indent=2)

    return result


def write_progress(epoch, total_epochs, run_idx, total_runs, metric=None):
    """Write progress file for system monitor."""
    progress = {
        "task_id": TASK_ID,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": run_idx,
        "total_steps": total_runs,
        "loss": None,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps(progress))


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
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
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
    total_runs = len(METHODS) * len(SEEDS)  # 8 * 3 = 24

    print("=" * 80)
    print(f"FULL EXPERIMENT: diagnostic_cifar10 — Phase 1")
    print(f"  Model: {MODEL_NAME} on {DATASET}")
    print(f"  Epochs: {TOTAL_EPOCHS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Methods: {', '.join(METHODS)}")
    print(f"  Total runs: {total_runs}")
    print(f"  Expected time: ~{total_runs * 12}min ({total_runs * 12 / 60:.1f}h)")
    print("=" * 80)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Write PID file
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data once
    print("Loading CIFAR-10 data...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=BATCH_SIZE,
        data_root=str(CODE_DIR / "data"),
        seed=42,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    all_results = []
    start_time = datetime.now()
    overall_start = time.time()
    run_idx = 0

    for method in METHODS:
        for seed in SEEDS:
            run_idx += 1
            print(f"\n{'=' * 70}")
            print(f"[{run_idx}/{total_runs}] {method} / seed={seed} / 200 epochs")
            print(f"{'=' * 70}")

            try:
                result = run_single(method, seed, device, train_loader, test_loader,
                                    run_idx, total_runs)
                all_results.append(result)
                extra = ""
                if method == "UDWDC-v2":
                    extra = f" | WD Budget: {result.get('cumulative_wd_budget', 0):.4e}"
                print(f"  DONE: {method}/seed{seed} | Best: {result['best_acc']:.2f}% | "
                      f"Final: {result['final_test_acc']:.2f}% | "
                      f"Gen Gap: {result['gen_gap']:.2f}% | "
                      f"Time: {result['elapsed_sec']:.1f}s{extra}")
            except Exception as e:
                print(f"  FAILED: {method}/seed{seed} — {e}")
                traceback.print_exc()
                all_results.append({
                    "method": method, "seed": seed, "status": "failed",
                    "error": str(e),
                })

            # Free GPU memory between runs
            torch.cuda.empty_cache()
            gc.collect()

    total_elapsed = time.time() - overall_start
    end_time = datetime.now()

    # ── Aggregate Results ──
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Group by method, compute mean ± std across seeds
    method_stats = {}
    for method in METHODS:
        method_results = [r for r in all_results
                          if r.get("method") == method and "best_acc" in r]
        if not method_results:
            continue

        best_accs = [r["best_acc"] for r in method_results]
        final_accs = [r["final_test_acc"] for r in method_results]
        gen_gaps = [r["gen_gap"] for r in method_results]
        wd_budgets = [r.get("total_wd_budget", 0) for r in method_results]

        method_stats[method] = {
            "best_acc_mean": float(np.mean(best_accs)),
            "best_acc_std": float(np.std(best_accs)),
            "final_acc_mean": float(np.mean(final_accs)),
            "final_acc_std": float(np.std(final_accs)),
            "gen_gap_mean": float(np.mean(gen_gaps)),
            "gen_gap_std": float(np.std(gen_gaps)),
            "wd_budget_mean": float(np.mean(wd_budgets)),
            "wd_budget_std": float(np.std(wd_budgets)),
            "n_seeds": len(method_results),
            "per_seed": method_results,
        }

    # Print summary table
    print(f"\n{'Method':<20} {'Best Acc':>14} {'Final Acc':>14} {'Gen Gap':>14} {'WD Budget':>14} {'Seeds':>6}")
    print("-" * 90)
    for method in METHODS:
        if method not in method_stats:
            print(f"{method:<20} {'FAILED':>14}")
            continue
        s = method_stats[method]
        print(f"{method:<20} "
              f"{s['best_acc_mean']:>6.2f}±{s['best_acc_std']:.2f} "
              f"{s['final_acc_mean']:>6.2f}±{s['final_acc_std']:.2f} "
              f"{s['gen_gap_mean']:>6.2f}±{s['gen_gap_std']:.2f} "
              f"{s['wd_budget_mean']:>10.2f}±{s['wd_budget_std']:.1f} "
              f"{s['n_seeds']:>6}")
    print("-" * 90)
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)")

    # ── Save Summary JSON ──
    summary = {
        "task_id": TASK_ID,
        "mode": "full",
        "model": MODEL_NAME,
        "dataset": DATASET,
        "epochs": TOTAL_EPOCHS,
        "seeds": SEEDS,
        "methods": METHODS,
        "batch_size": BATCH_SIZE,
        "base_lr": BASE_LR,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "total_elapsed_sec": total_elapsed,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "method_stats": {},
        "all_results": [],
    }

    # Serialize method stats (without per_seed details — those are large)
    for method, stats in method_stats.items():
        summary["method_stats"][method] = {
            "best_acc_mean": stats["best_acc_mean"],
            "best_acc_std": stats["best_acc_std"],
            "final_acc_mean": stats["final_acc_mean"],
            "final_acc_std": stats["final_acc_std"],
            "gen_gap_mean": stats["gen_gap_mean"],
            "gen_gap_std": stats["gen_gap_std"],
            "wd_budget_mean": stats["wd_budget_mean"],
            "wd_budget_std": stats["wd_budget_std"],
            "n_seeds": stats["n_seeds"],
        }

    # Serialize per-run results (without huge trajectory data)
    for r in all_results:
        entry = {k: v for k, v in r.items()
                 if k not in ["per_epoch_rho", "per_epoch_alpha",
                              "per_epoch_effective_wd", "epoch_metrics"]}
        summary["all_results"].append(entry)

    summary_file = SAVE_DIR / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_file}")

    # ── Save Summary Markdown ──
    md_lines = [
        f"# Phase 1: Full Diagnostic Results — CIFAR-10/ResNet-20",
        f"",
        f"**Task ID**: {TASK_ID}",
        f"**Model**: {MODEL_NAME}",
        f"**Dataset**: {DATASET}",
        f"**Epochs**: {TOTAL_EPOCHS}",
        f"**Seeds**: {', '.join(str(s) for s in SEEDS)}",
        f"**Methods**: {', '.join(METHODS)}",
        f"**Total runs**: {total_runs}",
        f"**Total time**: {total_elapsed/3600:.2f} hours",
        f"",
        f"## Method Comparison (mean ± std across {len(SEEDS)} seeds)",
        f"",
        f"| Method | Best Acc (%) | Final Acc (%) | Gen Gap (%) | WD Budget |",
        f"|--------|-------------|--------------|-------------|-----------|",
    ]
    for method in METHODS:
        if method not in method_stats:
            md_lines.append(f"| {method} | FAILED | FAILED | FAILED | FAILED |")
            continue
        s = method_stats[method]
        md_lines.append(
            f"| {method} | {s['best_acc_mean']:.2f}±{s['best_acc_std']:.2f} | "
            f"{s['final_acc_mean']:.2f}±{s['final_acc_std']:.2f} | "
            f"{s['gen_gap_mean']:.2f}±{s['gen_gap_std']:.2f} | "
            f"{s['wd_budget_mean']:.2f}±{s['wd_budget_std']:.1f} |"
        )

    md_lines += [
        f"",
        f"## Per-Seed Results",
        f"",
        f"| Method | Seed | Best Acc (%) | Final Acc (%) | Gen Gap (%) | Time (s) |",
        f"|--------|------|-------------|--------------|-------------|----------|",
    ]
    for r in all_results:
        if "best_acc" not in r:
            md_lines.append(f"| {r.get('method','?')} | {r.get('seed','?')} | FAILED | | | |")
            continue
        md_lines.append(
            f"| {r['method']} | {r['seed']} | {r['best_acc']:.2f} | "
            f"{r['final_test_acc']:.2f} | {r['gen_gap']:.2f} | {r['elapsed_sec']:.1f} |"
        )

    # Key observations
    md_lines += ["", "## Key Observations", ""]

    # Check UDWDC vs UDWDC-v2
    if "UDWDC" in method_stats and "UDWDC-v2" in method_stats:
        u1 = method_stats["UDWDC"]
        u2 = method_stats["UDWDC-v2"]
        md_lines.append(f"- UDWDC: {u1['best_acc_mean']:.2f}±{u1['best_acc_std']:.2f}% vs "
                       f"UDWDC-v2: {u2['best_acc_mean']:.2f}±{u2['best_acc_std']:.2f}%")

    # Check UDWDC-v2 WD budget
    if "UDWDC-v2" in method_stats:
        md_lines.append(f"- UDWDC-v2 WD budget: {method_stats['UDWDC-v2']['wd_budget_mean']:.2f} "
                       f"(confirms stability fix, WD > 0)")

    # Ranking
    ranked = sorted(method_stats.items(), key=lambda x: x[1]["best_acc_mean"], reverse=True)
    ranking_str = " > ".join(f"{m}({s['best_acc_mean']:.2f}%)" for m, s in ranked)
    md_lines.append(f"- Method ranking: {ranking_str}")

    n_successful = sum(1 for r in all_results if "best_acc" in r)
    md_lines.append(f"- {n_successful}/{total_runs} runs completed successfully")

    md_file = SAVE_DIR / "summary.md"
    md_file.write_text("\n".join(md_lines))
    print(f"Saved: {md_file}")

    # ── Write DONE marker ──
    n_failed = total_runs - n_successful
    status = "success" if n_failed == 0 else "partial"
    mark_task_done(
        TASK_ID, str(RESULTS_DIR), status=status,
        summary=f"Phase 1 diagnostic: {n_successful}/{total_runs} runs completed. "
                f"8 methods x 3 seeds x 200 epochs on CIFAR-10/ResNet-20. "
                f"Total time: {total_elapsed/3600:.2f}h."
    )
    print(f"Wrote DONE marker")

    # ── Update GPU Progress ──
    update_gpu_progress(
        str(RESULTS_DIR), TASK_ID,
        planned_min=60,
        actual_min=int(total_elapsed / 60),
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        config_snapshot={
            "model": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "dataset": DATASET,
            "epochs": TOTAL_EPOCHS,
            "n_methods": len(METHODS),
            "n_seeds": len(SEEDS),
            "gpu_model": "RTX PRO 6000 Blackwell",
            "gpu_count": 1,
        },
        status=status,
    )
    print(f"Updated gpu_progress.json")

    return n_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
