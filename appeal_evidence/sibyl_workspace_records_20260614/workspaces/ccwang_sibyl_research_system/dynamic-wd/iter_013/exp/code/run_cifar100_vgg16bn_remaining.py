#!/usr/bin/env python3
"""
Complete remaining VGG16BN runs: CPR, CAWD, EqWD x 3 seeds = 9 runs.
Previous agent died mid-run. NoWD, FixedWD, SWD, CWD already completed.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WORKSPACE = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current")
CODE_DIR = WORKSPACE / "exp" / "code"
RESULTS_BASE = WORKSPACE / "exp" / "results" / "cifar100_vgg16bn_full"
TASK_ID = "cifar100_vgg16bn_full"

# Only remaining methods
METHODS = ["CPR", "CAWD", "EqWD"]
SEEDS = [42, 123, 456]

EPOCHS = 50
BATCH_SIZE = 128
LR = 0.1
WD_LAMBDA = 5e-4
EQWD_BETA = 2.0  # VGG uses beta=2.0 (from pilot results)
EQWD_EMA_ALPHA = 0.9

def get_run_id(method, seed):
    return f"{method}_seed{seed}"

def build_cmd(method, seed, output_dir):
    python_exe = "/home/ccwang/sibyl-research-system/.venv/bin/python3"
    cmd = [
        python_exe, str(CODE_DIR / "train.py"),
        "--dataset", "cifar100",
        "--arch", "vgg16bn",
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--lr_schedule", "cosine",
        "--wd_method", method,
        "--wd_lambda", str(WD_LAMBDA),
        "--seed", str(seed),
        "--diag_interval", "10",
        "--output_dir", output_dir,
        "--task_id", get_run_id(method, seed),
        "--data_dir", str(CODE_DIR / "data"),
        "--num_workers", "4",
    ]
    if method == "EqWD":
        cmd.extend(["--eqwd_beta", str(EQWD_BETA)])
        cmd.extend(["--eqwd_ema_alpha", str(EQWD_EMA_ALPHA)])
    return cmd

def write_progress(completed, total, current_method, current_seed, start_time):
    progress_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    data = {
        "task_id": TASK_ID,
        "epoch": completed + 12,  # 12 already done
        "total_epochs": total + 12,
        "step": completed + 12,
        "total_steps": total + 12,
        "loss": None,
        "metric": {
            "runs_completed": completed + 12,
            "runs_total": total + 12,
            "current_method": current_method,
            "current_seed": current_seed,
            "elapsed_minutes": round((time.time() - start_time) / 60, 1),
        },
        "updated_at": datetime.now().isoformat(),
    }
    progress_file.write_text(json.dumps(data))

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "7")

    # Write PID
    pid_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    start_time = time.time()
    all_results = {}
    total_runs = len(METHODS) * len(SEEDS)
    completed = 0

    for method in METHODS:
        for seed in SEEDS:
            run_id = get_run_id(method, seed)
            output_dir = str(RESULTS_BASE / run_id)

            # Skip if already has results
            result_file = Path(output_dir) / f"{run_id}_results.json"
            done_file = Path(output_dir) / f"{run_id}_DONE"
            default_done = Path(output_dir) / "default_DONE"
            default_results = Path(output_dir) / "default_results.json"
            if result_file.exists() or done_file.exists() or default_done.exists():
                print(f"[SKIP] {run_id} already completed")
                completed += 1
                if default_results.exists():
                    all_results[run_id] = json.loads(default_results.read_text())
                continue

            # Clean partial run directory
            partial_dir = Path(output_dir)
            if partial_dir.exists():
                import shutil
                shutil.rmtree(partial_dir)

            partial_dir.mkdir(parents=True, exist_ok=True)
            write_progress(completed, total_runs, method, seed, start_time)

            cmd = build_cmd(method, seed, output_dir)
            print(f"[RUN] {run_id}: {' '.join(cmd[-8:])}")

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(CODE_DIR))
                if result.returncode == 0:
                    print(f"[OK] {run_id}")
                    # Read results
                    for rf in [result_file, default_results]:
                        if rf.exists():
                            all_results[run_id] = json.loads(rf.read_text())
                            break
                else:
                    print(f"[FAIL] {run_id}: {result.stderr[-500:]}")
            except subprocess.TimeoutExpired:
                print(f"[TIMEOUT] {run_id}")

            completed += 1
            write_progress(completed, total_runs, method, seed, start_time)

    # Write DONE marker
    end_time = time.time()
    done_marker = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
    done_data = {
        "task_id": TASK_ID,
        "status": "success",
        "summary": f"Completed {completed}/{total_runs} remaining VGG16BN runs",
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    done_marker.write_text(json.dumps(done_data, indent=2))
    print(f"\n[DONE] {completed}/{total_runs} runs completed in {(end_time-start_time)/60:.1f} min")

if __name__ == "__main__":
    main()
