#!/usr/bin/env python3
"""
ImageNet ResNet-50 full experiment: 7 methods × 3 seeds = 21 runs.
Two GPU parallel mode with AMP (45 epochs, batch_size=256).
Split: GPU_A runs methods[0:4], GPU_B runs methods[4:7].
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
RESULTS_DIR = WORKSPACE / "exp" / "results"
PYTHON = "/home/ccwang/sibyl-research-system/.venv/bin/python3"
TRAIN_SCRIPT = str(CODE_DIR / "train.py")
TASK_ID = "imagenet_resnet50_full"
IMAGENET_DIR = "/home/ccwang/dataset/imagenet-1k"

ALL_METHODS = ["NoWD", "FixedWD", "SWD", "CWD", "CPR", "CAWD", "EqWD"]
SEEDS = [42, 123, 456]
EPOCHS = 45
BATCH_SIZE = 256
LR = 0.1
WD_LAMBDA = 1e-4
EQWD_BETA = 1.0
EQWD_EMA_ALPHA = 0.9


def build_cmd(method, seed, output_dir):
    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--dataset", "imagenet",
        "--arch", "resnet50",
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--lr_schedule", "cosine",  # Cosine annealing (better for shorter schedules)
        "--wd_method", method,
        "--wd_lambda", str(WD_LAMBDA),
        "--seed", str(seed),
        "--diag_interval", "200",
        "--output_dir", output_dir,
        "--task_id", f"{method}_seed{seed}",
        "--imagenet_dir", IMAGENET_DIR,
        "--num_workers", "8",
        "--amp",
    ]
    if method == "EqWD":
        cmd.extend(["--eqwd_beta", str(EQWD_BETA)])
        cmd.extend(["--eqwd_ema_alpha", str(EQWD_EMA_ALPHA)])
    return cmd


def write_progress(completed, total, current_method, current_seed, start_time, group=""):
    progress = {
        "task_id": TASK_ID,
        "epoch": completed, "total_epochs": total,
        "step": completed, "total_steps": total, "loss": None,
        "metric": {
            "runs_completed": completed, "runs_total": total,
            "current_method": current_method, "current_seed": current_seed,
            "elapsed_minutes": round((time.time() - start_time) / 60, 1),
            "group": group,
        },
        "updated_at": datetime.now().isoformat(),
    }
    suffix = f"_{group}" if group else ""
    (RESULTS_DIR / f"{TASK_ID}{suffix}_PROGRESS.json").write_text(json.dumps(progress))


def run_group(methods, gpu_id, group_name):
    """Run a group of methods on a specific GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[{group_name}] GPU {gpu_id}: {methods}", flush=True)

    task_dir = RESULTS_DIR / "full" / "imagenet_resnet50"
    task_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    all_results = {}
    total_runs = len(methods) * len(SEEDS)
    completed = 0

    for method in methods:
        for seed in SEEDS:
            run_id = f"{method}_seed{seed}"
            output_dir = str(task_dir / run_id)

            result_files = [
                Path(output_dir) / f"{run_id}_results.json",
                Path(output_dir) / "default_results.json",
            ]
            if any(f.exists() for f in result_files):
                print(f"[SKIP] {run_id}", flush=True)
                for rf in result_files:
                    if rf.exists():
                        all_results[run_id] = json.loads(rf.read_text())
                completed += 1
                continue

            od = Path(output_dir)
            if od.exists():
                import shutil
                shutil.rmtree(od)
            od.mkdir(parents=True, exist_ok=True)

            write_progress(completed, total_runs, method, seed, start_time, group_name)

            cmd = build_cmd(method, seed, output_dir)
            print(f"[{group_name} {completed+1}/{total_runs}] {run_id}...", flush=True)

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=259200,
                    cwd=str(CODE_DIR),
                )
                if result.returncode == 0:
                    for rf in result_files:
                        if rf.exists():
                            all_results[run_id] = json.loads(rf.read_text())
                    acc = all_results.get(run_id, {}).get("best_test_top1", "?")
                    print(f"  OK (top1={acc})", flush=True)
                else:
                    print(f"  FAIL: {result.stderr[-500:]}", flush=True)
                    err_log = Path(output_dir) / "error.log"
                    err_log.write_text(result.stderr[-2000:])
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT", flush=True)

            completed += 1
            write_progress(completed, total_runs, method, seed, start_time, group_name)

    return all_results


def main():
    # Support running as a specific group
    group = sys.argv[1] if len(sys.argv) > 1 else "all"

    task_dir = RESULTS_DIR / "full" / "imagenet_resnet50"
    task_dir.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

    if group == "A":
        methods = ALL_METHODS[:4]  # NoWD, FixedWD, SWD, CWD
        gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "2"))
        all_results = run_group(methods, gpu_id, "A")
    elif group == "B":
        methods = ALL_METHODS[4:]  # CPR, CAWD, EqWD
        gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "7"))
        all_results = run_group(methods, gpu_id, "B")
    else:
        # Sequential all
        gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "2"))
        all_results = run_group(ALL_METHODS, gpu_id, "all")

    # Only write DONE when we're the "all" runner or we can check both groups
    if group == "all":
        _write_done(all_results)
    else:
        # Write partial done marker
        (RESULTS_DIR / f"{TASK_ID}_{group}_DONE").write_text(
            json.dumps({"group": group, "results": all_results,
                        "timestamp": datetime.now().isoformat()}, indent=2))
        # Check if both groups done
        if (RESULTS_DIR / f"{TASK_ID}_A_DONE").exists() and \
           (RESULTS_DIR / f"{TASK_ID}_B_DONE").exists():
            a = json.loads((RESULTS_DIR / f"{TASK_ID}_A_DONE").read_text())
            b = json.loads((RESULTS_DIR / f"{TASK_ID}_B_DONE").read_text())
            merged = {**a.get("results", {}), **b.get("results", {})}
            _write_done(merged)


def _write_done(all_results):
    total_runs = len(ALL_METHODS) * len(SEEDS)
    done = {
        "task_id": TASK_ID, "status": "success",
        "summary": f"{len(all_results)}/{total_runs} runs",
        "results": all_results, "timestamp": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps(done, indent=2))
    print(f"\n[DONE] ImageNet ResNet-50: {len(all_results)}/{total_runs}")


if __name__ == "__main__":
    main()
