#!/usr/bin/env python3
"""Pilot for diagnostic_cifar10: Train ResNet-20 on CIFAR-10 for 10 epochs
with all 7 WD methods, tracking per-layer rho_t trajectories.

PILOT pass criteria:
1. All 7 methods complete 10 epochs without error
2. rho_t trajectories are non-trivial (std > 0.01)
3. FixedWD per-layer rho_t CV < 0.3 at epoch 10 (trending toward convergence)

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n sibyl_dynamic-wd python pilot_diagnostic_cifar10.py
"""

import json
import os
import sys
import time
import math
import traceback
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


TASK_ID = "diagnostic_cifar10"
PILOT_EPOCHS = 10
PILOT_SEED = 42
BATCH_SIZE = 128
BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
MODEL_NAME = "resnet20"
DATASET = "cifar10"
NUM_CLASSES = 10

METHODS = ["FixedWD", "CWD", "SWD", "CPR", "DefazioCorrective", "NoWD", "UDWDC"]

WORKSPACE = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current")
SAVE_DIR = WORKSPACE / "exp" / "results" / "pilots" / "diagnostic_cifar10"
RESULTS_DIR = WORKSPACE / "exp" / "results"


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_lr(optimizer_wrapper, epoch, total_epochs, base_lr):
    lr = base_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    optimizer_wrapper.set_lr(lr)
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


def run_method(method, device, train_loader, test_loader):
    """Run one method for PILOT_EPOCHS and return results + diagnostics."""
    set_seed(PILOT_SEED)

    model = create_model(MODEL_NAME, num_classes=NUM_CLASSES).to(device)

    opt_kwargs = {}
    if method == "UDWDC":
        opt_kwargs = {"K_p": 0.5, "K_i": 0.1, "K_d": 0.3}

    wd_opt = create_optimizer(
        method, model, lr=BASE_LR, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY, **opt_kwargs
    )

    criterion = nn.CrossEntropyLoss()
    logger = DiagnosticLogger(
        save_dir=str(SAVE_DIR / method),
        method=method, seed=PILOT_SEED,
        model_name=MODEL_NAME, dataset=DATASET
    )

    t0 = time.time()
    best_acc = 0.0
    per_epoch_rho = []  # list of dicts: epoch -> {layer: rho_t}

    for epoch in range(PILOT_EPOCHS):
        # Use 200 as total_epochs for LR schedule (matching full run schedule)
        lr = cosine_lr(wd_opt, epoch, 200, BASE_LR)

        train_loss, train_acc = train_one_epoch(model, train_loader, wd_opt, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        diagnostics = wd_opt.get_diagnostics()
        logger.log_epoch(epoch, diagnostics, train_loss, test_loss, train_acc, test_acc, lr=lr)

        # Store per-layer rho_t for this epoch
        epoch_rho = {}
        for layer_name, diag in diagnostics.items():
            epoch_rho[layer_name] = diag["rho_t"]
        per_epoch_rho.append(epoch_rho)

        if test_acc > best_acc:
            best_acc = test_acc

        print(f"  [{method}] Epoch {epoch+1}/{PILOT_EPOCHS} | LR: {lr:.6f} | "
              f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Loss: {train_loss:.4f}")

    elapsed = time.time() - t0
    logger.save()

    return {
        "method": method,
        "best_acc": best_acc,
        "final_test_acc": test_acc,
        "final_train_acc": train_acc,
        "elapsed_sec": elapsed,
        "per_epoch_rho": per_epoch_rho,
        "epoch_metrics": logger.epoch_metrics,
        "n_layers": len(per_epoch_rho[0]) if per_epoch_rho else 0,
    }


def check_pilot_criteria(all_results):
    """Check PILOT pass criteria and return structured results."""
    checks = {}

    # Check 1: All 7 methods complete 10 epochs without error
    completed_methods = [r["method"] for r in all_results if r is not None]
    checks["all_methods_complete"] = {
        "passed": len(completed_methods) == len(METHODS),
        "detail": f"{len(completed_methods)}/{len(METHODS)} methods completed",
        "completed": completed_methods,
    }

    # Check 2: rho_t trajectories are non-trivial (std > 0.01)
    rho_nontrivial = {}
    for result in all_results:
        if result is None:
            continue
        method = result["method"]
        # Collect all rho_t values across layers and epochs
        all_rho_values = []
        for epoch_rho in result["per_epoch_rho"]:
            for layer, rho in epoch_rho.items():
                all_rho_values.append(rho)

        std = float(np.std(all_rho_values)) if all_rho_values else 0.0
        rho_nontrivial[method] = {"std": std, "passed": std > 0.01}

    all_nontrivial = all(v["passed"] for v in rho_nontrivial.values())
    checks["rho_t_nontrivial"] = {
        "passed": all_nontrivial,
        "detail": rho_nontrivial,
    }

    # Check 3: FixedWD per-layer rho_t CV < 0.3 at epoch 10
    fixed_wd_result = next((r for r in all_results if r is not None and r["method"] == "FixedWD"), None)
    if fixed_wd_result:
        last_epoch_rho = fixed_wd_result["per_epoch_rho"][-1]
        rho_values = list(last_epoch_rho.values())
        # Filter out zero/near-zero values (bias/BN layers)
        rho_wd_layers = [v for v in rho_values if v > 0.001]

        if rho_wd_layers:
            mean_rho = float(np.mean(rho_wd_layers))
            std_rho = float(np.std(rho_wd_layers))
            cv = std_rho / (mean_rho + 1e-10)
            checks["fixedwd_cv_convergence"] = {
                "passed": cv < 0.3,
                "cv": cv,
                "mean_rho": mean_rho,
                "std_rho": std_rho,
                "n_wd_layers": len(rho_wd_layers),
                "detail": f"CV={cv:.4f} (threshold 0.3), mean_rho={mean_rho:.4f}, std_rho={std_rho:.4f}",
            }
        else:
            checks["fixedwd_cv_convergence"] = {
                "passed": False,
                "detail": "No WD layers found",
            }
    else:
        checks["fixedwd_cv_convergence"] = {
            "passed": False,
            "detail": "FixedWD did not complete",
        }

    overall = all(c["passed"] for c in checks.values())
    return overall, checks


def write_progress(epoch, total, method_idx, total_methods, metric=None):
    """Write progress file for system monitor."""
    progress = {
        "task_id": TASK_ID,
        "epoch": epoch,
        "total_epochs": total,
        "step": method_idx,
        "total_steps": total_methods,
        "loss": None,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps(progress))


def main():
    print("=" * 70)
    print(f"PILOT: diagnostic_cifar10 — ResNet-20 on CIFAR-10, {PILOT_EPOCHS} epochs, 7 methods")
    print(f"Device: cuda:0, Seed: {PILOT_SEED}")
    print("=" * 70)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Write PID file
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data once (full dataset, no subset for pilot — dataset is small enough)
    print("Loading CIFAR-10 data...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=BATCH_SIZE,
        data_root=str(WORKSPACE / "exp" / "code" / "data"),
        seed=PILOT_SEED,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    all_results = []
    overall_start = time.time()

    for idx, method in enumerate(METHODS):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(METHODS)}] Running: {method}")
        print(f"{'='*60}")

        try:
            result = run_method(method, device, train_loader, test_loader)
            all_results.append(result)
            print(f"  DONE: {method} | Best: {result['best_acc']:.2f}% | "
                  f"Final: {result['final_test_acc']:.2f}% | Time: {result['elapsed_sec']:.1f}s")
        except Exception as e:
            print(f"  FAILED: {method} — {e}")
            traceback.print_exc()
            all_results.append(None)

        write_progress(
            epoch=PILOT_EPOCHS,
            total=PILOT_EPOCHS * len(METHODS),
            method_idx=idx + 1,
            total_methods=len(METHODS),
            metric={"method": method, "completed": all_results[-1] is not None},
        )

    total_elapsed = time.time() - overall_start

    # Evaluate pilot criteria
    print("\n" + "=" * 70)
    print("PILOT CRITERIA EVALUATION")
    print("=" * 70)

    overall_pass, checks = check_pilot_criteria([r for r in all_results if r is not None])

    for check_name, check_result in checks.items():
        status = "PASS" if check_result["passed"] else "FAIL"
        detail = check_result.get("detail", "")
        if isinstance(detail, dict):
            detail_str = "; ".join(f"{k}: std={v['std']:.4f} {'PASS' if v['passed'] else 'FAIL'}"
                                   for k, v in detail.items())
        else:
            detail_str = str(detail)
        print(f"  [{status}] {check_name}: {detail_str}")

    # Summary table
    print("\n" + "-" * 70)
    print(f"{'Method':<20} {'Best Acc':>10} {'Final Acc':>10} {'Train Acc':>10} {'Time (s)':>10}")
    print("-" * 70)
    for result in all_results:
        if result is None:
            continue
        print(f"{result['method']:<20} {result['best_acc']:>10.2f} {result['final_test_acc']:>10.2f} "
              f"{result['final_train_acc']:>10.2f} {result['elapsed_sec']:>10.1f}")
    print("-" * 70)
    print(f"Total time: {total_elapsed:.1f}s")

    recommendation = "GO" if overall_pass else "NO_GO"
    print(f"\nOverall: {recommendation}")

    # Save pilot summary JSON
    pilot_summary = {
        "overall_recommendation": recommendation,
        "task_id": TASK_ID,
        "mode": "pilot",
        "model": MODEL_NAME,
        "dataset": DATASET,
        "epochs": PILOT_EPOCHS,
        "seed": PILOT_SEED,
        "checks": {},
        "per_method": {},
        "total_elapsed_sec": total_elapsed,
    }

    # Serialize checks (make JSON-safe)
    for check_name, check_result in checks.items():
        safe_check = {"passed": check_result["passed"]}
        detail = check_result.get("detail", "")
        if isinstance(detail, dict):
            safe_check["detail"] = {k: {"std": float(v["std"]), "passed": v["passed"]} for k, v in detail.items()}
        else:
            safe_check["detail"] = str(detail)
        for k, v in check_result.items():
            if k not in ["passed", "detail"] and isinstance(v, (int, float, str, bool)):
                safe_check[k] = v
        pilot_summary["checks"][check_name] = safe_check

    for result in all_results:
        if result is None:
            continue
        pilot_summary["per_method"][result["method"]] = {
            "best_acc": result["best_acc"],
            "final_test_acc": result["final_test_acc"],
            "final_train_acc": result["final_train_acc"],
            "elapsed_sec": result["elapsed_sec"],
            "n_layers": result["n_layers"],
            "mean_rho_trajectory": [
                float(np.mean(list(epoch_rho.values())))
                for epoch_rho in result["per_epoch_rho"]
            ],
        }

    summary_file = SAVE_DIR / "pilot_summary.json"
    with open(summary_file, "w") as f:
        json.dump(pilot_summary, f, indent=2)
    print(f"\nSaved pilot summary: {summary_file}")

    # Write pilot_summary.md
    md_lines = [
        f"# Pilot Summary: diagnostic_cifar10",
        f"",
        f"**Recommendation**: {recommendation}",
        f"**Model**: {MODEL_NAME} on {DATASET}",
        f"**Epochs**: {PILOT_EPOCHS} (pilot, out of 200 full)",
        f"**Seed**: {PILOT_SEED}",
        f"**Total time**: {total_elapsed:.1f}s",
        f"",
        f"## Checks",
    ]
    for check_name, check_result in checks.items():
        status = "PASS" if check_result["passed"] else "FAIL"
        md_lines.append(f"- **[{status}]** {check_name}")

    md_lines += [
        f"",
        f"## Per-Method Results",
        f"",
        f"| Method | Best Acc (%) | Final Acc (%) | Train Acc (%) | Time (s) |",
        f"|--------|-------------|--------------|--------------|----------|",
    ]
    for result in all_results:
        if result is None:
            continue
        md_lines.append(
            f"| {result['method']} | {result['best_acc']:.2f} | "
            f"{result['final_test_acc']:.2f} | {result['final_train_acc']:.2f} | "
            f"{result['elapsed_sec']:.1f} |"
        )

    md_file = SAVE_DIR / "pilot_summary.md"
    md_file.write_text("\n".join(md_lines))
    print(f"Saved pilot summary MD: {md_file}")

    # Write DONE marker
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    done_file = RESULTS_DIR / f"{TASK_ID}_DONE"
    done_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": "success" if overall_pass else "partial",
        "summary": f"Pilot {recommendation}: {len([r for r in all_results if r is not None])}/{len(METHODS)} methods completed. "
                   + "; ".join(f"{k}: {'PASS' if v['passed'] else 'FAIL'}" for k, v in checks.items()),
        "final_progress": {
            "recommendation": recommendation,
            "methods_completed": len([r for r in all_results if r is not None]),
            "total_methods": len(METHODS),
        },
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"Wrote DONE marker: {done_file}")

    return overall_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
