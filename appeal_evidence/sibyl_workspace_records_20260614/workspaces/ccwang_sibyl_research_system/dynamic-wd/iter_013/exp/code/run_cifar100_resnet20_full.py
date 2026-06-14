#!/usr/bin/env python3
"""
Batch runner for cifar100_resnet20_full task.

Runs all 7 WD methods x 3 seeds = 21 training runs on CIFAR-100/ResNet-20,
200 epochs each. Results saved to exp/results/full/cifar100_resnet20/.

Updates gpu_progress.json with timing info upon completion.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Configuration
WORKSPACE = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current")
CODE_DIR = WORKSPACE / "exp" / "code"
RESULTS_BASE = WORKSPACE / "exp" / "results" / "full" / "cifar100_resnet20"
GPU_PROGRESS_FILE = WORKSPACE / "exp" / "gpu_progress.json"
TASK_ID = "cifar100_resnet20_full"

# Methods and seeds
METHODS = ["NoWD", "FixedWD", "SWD", "CWD", "CPR", "CAWD", "EqWD"]
SEEDS = [42, 123, 456]

# Training hyperparameters
EPOCHS = 200
BATCH_SIZE = 128
LR = 0.1
WD_LAMBDA = 5e-4
EQWD_BETA = 1.0
EQWD_EMA_ALPHA = 0.9
DIAG_INTERVAL = 10
CHECKPOINT_EPOCHS = "100,200"


def get_run_id(method: str, seed: int) -> str:
    return f"{method}_seed{seed}"


def build_cmd(method: str, seed: int, output_dir: str) -> list:
    """Build the training command."""
    run_id = get_run_id(method, seed)
    python_exe = "/home/ccwang/sibyl-research-system/.venv/bin/python3"
    cmd = [
        python_exe, str(CODE_DIR / "train.py"),
        "--dataset", "cifar100",
        "--arch", "resnet20",
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--lr_schedule", "cosine",
        "--wd_method", method,
        "--wd_lambda", str(WD_LAMBDA),
        "--seed", str(seed),
        "--diag_interval", str(DIAG_INTERVAL),
        "--output_dir", output_dir,
        "--task_id", run_id,
        "--save_checkpoints", CHECKPOINT_EPOCHS,
        "--data_dir", str(CODE_DIR / "data"),
        "--num_workers", "4",
    ]
    if method == "EqWD":
        cmd.extend(["--eqwd_beta", str(EQWD_BETA)])
        cmd.extend(["--eqwd_ema_alpha", str(EQWD_EMA_ALPHA)])
    return cmd


def write_progress(completed: int, total: int, current_method: str, current_seed: int,
                   results_so_far: dict, start_time: float):
    """Write progress file for system monitor."""
    elapsed = time.time() - start_time
    progress_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    data = {
        "task_id": TASK_ID,
        "epoch": completed,
        "total_epochs": total,
        "step": completed,
        "total_steps": total,
        "loss": None,
        "metric": {
            "runs_completed": completed,
            "runs_total": total,
            "current_method": current_method,
            "current_seed": current_seed,
            "elapsed_minutes": round(elapsed / 60, 1),
        },
        "updated_at": datetime.now().isoformat(),
    }
    progress_file.write_text(json.dumps(data))


def write_done(status: str, summary: str, all_results: dict):
    """Write DONE marker."""
    # Clean up PID file
    pid_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()

    progress_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass

    marker = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
    data = {
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    marker.write_text(json.dumps(data, indent=2))


def update_gpu_progress(all_results: dict, start_time: float, end_time: float, status: str):
    """Update gpu_progress.json with timing and completion info."""
    gp = {}
    if GPU_PROGRESS_FILE.exists():
        try:
            gp = json.loads(GPU_PROGRESS_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            pass

    completed = gp.get("completed", [])
    failed = gp.get("failed", [])
    running = gp.get("running", {})
    timings = gp.get("timings", {})

    # Remove from running
    if TASK_ID in running:
        del running[TASK_ID]

    elapsed_min = round((end_time - start_time) / 60)

    if status == "success":
        if TASK_ID not in completed:
            completed.append(TASK_ID)
    else:
        if TASK_ID not in failed:
            failed.append(TASK_ID)

    # Best accuracies per method
    method_summary = {}
    for method in METHODS:
        accs = []
        for seed in SEEDS:
            run_id = get_run_id(method, seed)
            if run_id in all_results:
                accs.append(all_results[run_id].get("best_test_top1", 0))
        if accs:
            method_summary[method] = {
                "mean": round(sum(accs) / len(accs), 2),
                "values": accs,
            }

    timings[TASK_ID] = {
        "planned_min": 60,
        "actual_min": elapsed_min,
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "config_snapshot": {
            "model": "ResNet-20",
            "dataset": "CIFAR-100",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "methods": len(METHODS),
            "seeds": len(SEEDS),
            "total_runs": len(METHODS) * len(SEEDS),
            "gpu_model": "RTX PRO 6000 Blackwell",
            "gpu_count": 1,
            "method_results": method_summary,
        },
    }

    gp["completed"] = completed
    gp["failed"] = failed
    gp["running"] = running
    gp["timings"] = timings
    GPU_PROGRESS_FILE.write_text(json.dumps(gp, indent=2))


def main():
    start_time = time.time()

    # Write PID
    pid_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    # Create results directory
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)

    total_runs = len(METHODS) * len(SEEDS)
    all_results = {}
    completed_runs = 0
    failed_runs = []

    print(f"=" * 70)
    print(f"CIFAR-100 / ResNet-20 Full Experiment")
    print(f"Methods: {METHODS}")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {total_runs}")
    print(f"Epochs per run: {EPOCHS}")
    print(f"=" * 70)

    for method in METHODS:
        for seed in SEEDS:
            run_id = get_run_id(method, seed)
            run_dir = str(RESULTS_BASE / run_id)
            os.makedirs(run_dir, exist_ok=True)

            # Check if already completed
            results_file = Path(run_dir) / f"{run_id}_results.json"
            if results_file.exists():
                try:
                    existing = json.loads(results_file.read_text())
                    if existing.get("epochs") == EPOCHS and existing.get("best_test_top1", 0) > 0:
                        print(f"\n[SKIP] {run_id} already completed: {existing['best_test_top1']:.2f}%")
                        all_results[run_id] = existing
                        completed_runs += 1
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass

            print(f"\n{'=' * 70}")
            print(f"[RUN {completed_runs + 1}/{total_runs}] {run_id}")
            print(f"  Method: {method}, Seed: {seed}")
            print(f"{'=' * 70}")

            write_progress(completed_runs, total_runs, method, seed, all_results, start_time)

            cmd = build_cmd(method, seed, run_dir)
            run_start = time.time()

            try:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = "7"
                result = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=3600,  # 1 hour timeout per run
                )

                if result.returncode != 0:
                    print(f"[FAIL] {run_id}: {result.stderr[-500:]}")
                    failed_runs.append(run_id)
                    # Save error log
                    error_log = Path(run_dir) / f"{run_id}_error.log"
                    error_log.write_text(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
                else:
                    # Load results
                    if results_file.exists():
                        run_results = json.loads(results_file.read_text())
                        all_results[run_id] = run_results
                        run_elapsed = time.time() - run_start
                        print(f"[OK] {run_id}: Best={run_results['best_test_top1']:.2f}%, "
                              f"Time={run_elapsed/60:.1f}min")
                    else:
                        print(f"[WARN] {run_id}: completed but no results file found")
                        failed_runs.append(run_id)

            except subprocess.TimeoutExpired:
                print(f"[TIMEOUT] {run_id}: exceeded 60 min limit")
                failed_runs.append(run_id)
            except Exception as e:
                print(f"[ERROR] {run_id}: {str(e)}")
                failed_runs.append(run_id)

            completed_runs += 1

    end_time = time.time()
    total_elapsed = end_time - start_time

    # Generate summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Successful: {len(all_results)}/{total_runs}")
    if failed_runs:
        print(f"Failed: {failed_runs}")

    # Per-method summary
    summary_table = {}
    print(f"\n{'Method':<10} {'Seed 42':>10} {'Seed 123':>10} {'Seed 456':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 62)
    for method in METHODS:
        accs = []
        row = {"seeds": {}}
        for seed in SEEDS:
            run_id = get_run_id(method, seed)
            if run_id in all_results:
                acc = all_results[run_id]["best_test_top1"]
                accs.append(acc)
                row["seeds"][str(seed)] = acc
            else:
                row["seeds"][str(seed)] = None

        if accs:
            mean_acc = sum(accs) / len(accs)
            if len(accs) > 1:
                std_acc = (sum((a - mean_acc)**2 for a in accs) / (len(accs) - 1))**0.5
            else:
                std_acc = 0.0
            row["mean"] = round(mean_acc, 2)
            row["std"] = round(std_acc, 2)
            row["n_seeds"] = len(accs)

            seed_strs = []
            for seed in SEEDS:
                run_id = get_run_id(method, seed)
                if run_id in all_results:
                    seed_strs.append(f"{all_results[run_id]['best_test_top1']:>10.2f}")
                else:
                    seed_strs.append(f"{'FAIL':>10}")
            print(f"{method:<10} {''.join(seed_strs)} {mean_acc:>10.2f} {std_acc:>10.2f}")
        else:
            row["mean"] = None
            row["std"] = None
            row["n_seeds"] = 0
            print(f"{method:<10} {'FAIL':>10} {'FAIL':>10} {'FAIL':>10} {'N/A':>10} {'N/A':>10}")

        summary_table[method] = row

    # Save aggregated summary
    summary = {
        "task_id": TASK_ID,
        "dataset": "cifar100",
        "arch": "resnet20",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "wd_lambda": WD_LAMBDA,
        "methods": METHODS,
        "seeds": SEEDS,
        "total_runs": total_runs,
        "successful_runs": len(all_results),
        "failed_runs": failed_runs,
        "elapsed_minutes": round(total_elapsed / 60, 1),
        "results_table": summary_table,
        "all_results": {k: {
            "best_test_top1": v.get("best_test_top1"),
            "final_test_top1": v.get("final_test_top1"),
            "final_train_loss": v.get("final_train_loss"),
            "final_train_acc": v.get("final_train_acc"),
            "elapsed_minutes": v.get("elapsed_minutes"),
        } for k, v in all_results.items()},
    }
    summary_path = RESULTS_BASE / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {summary_path}")

    # Also write summary.md
    md_lines = [
        "# CIFAR-100 / ResNet-20 Full Experiment Results\n",
        f"**Total time**: {total_elapsed/60:.1f} minutes\n",
        f"**Successful runs**: {len(all_results)}/{total_runs}\n",
        f"**Epochs**: {EPOCHS}, **Batch size**: {BATCH_SIZE}, **LR**: {LR}\n\n",
        "## Results Table (Test Accuracy %)\n\n",
        "| Method | Seed 42 | Seed 123 | Seed 456 | Mean | Std |\n",
        "|--------|---------|----------|----------|------|-----|\n",
    ]
    for method in METHODS:
        row = summary_table[method]
        seed_vals = []
        for seed in SEEDS:
            v = row["seeds"].get(str(seed))
            seed_vals.append(f"{v:.2f}" if v is not None else "FAIL")
        mean_str = f"{row['mean']:.2f}" if row["mean"] is not None else "N/A"
        std_str = f"{row['std']:.2f}" if row["std"] is not None else "N/A"
        md_lines.append(f"| {method} | {seed_vals[0]} | {seed_vals[1]} | {seed_vals[2]} | {mean_str} | {std_str} |\n")

    md_path = RESULTS_BASE / "summary.md"
    md_path.write_text("".join(md_lines))

    # Determine status
    status = "success" if len(failed_runs) == 0 else "partial"
    summary_msg = f"{len(all_results)}/{total_runs} runs completed in {total_elapsed/60:.1f}min"
    if all_results:
        # Check pilot pass criteria: EqWD mean >= FixedWD mean - 0.5%
        eqwd_accs = [all_results[get_run_id("EqWD", s)]["best_test_top1"]
                     for s in SEEDS if get_run_id("EqWD", s) in all_results]
        fixed_accs = [all_results[get_run_id("FixedWD", s)]["best_test_top1"]
                      for s in SEEDS if get_run_id("FixedWD", s) in all_results]
        if eqwd_accs and fixed_accs:
            eqwd_mean = sum(eqwd_accs) / len(eqwd_accs)
            fixed_mean = sum(fixed_accs) / len(fixed_accs)
            delta = eqwd_mean - fixed_mean
            summary_msg += f"; EqWD={eqwd_mean:.2f}% vs Fixed={fixed_mean:.2f}% (delta={delta:+.2f}%)"
            if delta >= -0.5:
                summary_msg += " [PASS]"
            else:
                summary_msg += " [FAIL: EqWD too low]"

    write_done(status, summary_msg, summary)
    update_gpu_progress(all_results, start_time, end_time, status)
    print(f"\n[DONE] {summary_msg}")


if __name__ == "__main__":
    main()
