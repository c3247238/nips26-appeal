#!/usr/bin/env python3
"""
Run all ablation experiments sequentially on a single GPU.

1. ablation_beta: EqWD with beta in {0.1, 0.5, 1.0, 2.0, 5.0}, CIFAR-100/ResNet-20, seed 42
2. ablation_ema: EqWD with EMA alpha in {0.8, 0.9, 0.95, 0.99}, CIFAR-100/ResNet-20, seed 42
3. ablation_layertype: EqWD uniform vs layer-aware, CIFAR-100/VGG-16-BN, 3 seeds
4. nobn_ablation: All 7 methods, CIFAR-100/VGG-16 (NO BN), 3 seeds
5. control experiments: phase_schedule, gradnorm_only, noise_injection
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

WORKSPACE = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current")
CODE_DIR = WORKSPACE / "exp" / "code"
RESULTS_DIR = WORKSPACE / "exp" / "results"
PYTHON = "/home/ccwang/sibyl-research-system/.venv/bin/python3"
TRAIN_SCRIPT = str(CODE_DIR / "train.py")

def run_experiment(task_id, runs, description):
    """Run a set of training runs for one ablation task."""
    task_dir = RESULTS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Write PID
    (RESULTS_DIR / f"{task_id}.pid").write_text(str(os.getpid()))

    results = {}
    total = len(runs)
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"[{task_id}] {description}")
    print(f"  Total runs: {total}")
    print(f"{'='*60}")

    for i, (run_id, cmd_args) in enumerate(runs):
        output_dir = str(task_dir / run_id)

        # Skip if already done
        done_files = [
            Path(output_dir) / f"{run_id}_DONE",
            Path(output_dir) / "default_DONE",
        ]
        result_files = [
            Path(output_dir) / f"{run_id}_results.json",
            Path(output_dir) / "default_results.json",
        ]
        if any(f.exists() for f in done_files + result_files):
            print(f"  [SKIP] {run_id}")
            for rf in result_files:
                if rf.exists():
                    results[run_id] = json.loads(rf.read_text())
            continue

        # Clean partial
        od = Path(output_dir)
        if od.exists():
            import shutil
            shutil.rmtree(od)
        od.mkdir(parents=True, exist_ok=True)

        # Write progress
        progress = {
            "task_id": task_id, "epoch": i, "total_epochs": total,
            "step": i, "total_steps": total, "loss": None,
            "metric": {"runs_completed": i, "runs_total": total,
                       "current_run": run_id,
                       "elapsed_minutes": round((time.time() - start_time) / 60, 1)},
            "updated_at": datetime.now().isoformat(),
        }
        (RESULTS_DIR / f"{task_id}_PROGRESS.json").write_text(json.dumps(progress))

        cmd = [PYTHON, TRAIN_SCRIPT] + cmd_args + ["--output_dir", output_dir, "--task_id", run_id]
        print(f"  [{i+1}/{total}] {run_id}...", end=" ", flush=True)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(CODE_DIR))
            if result.returncode == 0:
                for rf in result_files:
                    if rf.exists():
                        results[run_id] = json.loads(rf.read_text())
                acc = results.get(run_id, {}).get("best_test_top1", "?")
                print(f"OK (top1={acc})")
            else:
                print(f"FAIL")
                err_log = Path(output_dir) / "error.log"
                err_log.write_text(result.stderr[-2000:])
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT")

    # Write DONE
    elapsed = (time.time() - start_time) / 60
    done = {
        "task_id": task_id, "status": "success",
        "summary": f"{len(results)}/{total} runs in {elapsed:.1f}min",
        "results": results, "timestamp": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{task_id}_DONE").write_text(json.dumps(done, indent=2))
    print(f"  [{task_id}] Done: {len(results)}/{total} in {elapsed:.1f}min")
    return results

def base_cifar100_resnet20_args(seed=42):
    return ["--dataset", "cifar100", "--arch", "resnet20",
            "--epochs", "200", "--batch_size", "128",
            "--lr", "0.1", "--lr_schedule", "cosine",
            "--wd_method", "EqWD", "--wd_lambda", "5e-4",
            "--seed", str(seed), "--diag_interval", "10",
            "--data_dir", str(CODE_DIR / "data"), "--num_workers", "4"]

def base_cifar100_vgg16bn_args(seed=42):
    return ["--dataset", "cifar100", "--arch", "vgg16bn",
            "--epochs", "50", "--batch_size", "128",
            "--lr", "0.1", "--lr_schedule", "cosine",
            "--wd_lambda", "5e-4",
            "--seed", str(seed), "--diag_interval", "10",
            "--data_dir", str(CODE_DIR / "data"), "--num_workers", "4"]

def base_cifar100_vgg16_nobn_args(seed=42):
    return ["--dataset", "cifar100", "--arch", "vgg16",
            "--epochs", "50", "--batch_size", "128",
            "--lr", "0.1", "--lr_schedule", "cosine",
            "--wd_lambda", "5e-4",
            "--seed", str(seed), "--diag_interval", "10",
            "--data_dir", str(CODE_DIR / "data"), "--num_workers", "4"]

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "7")
    master_start = time.time()

    # 1. Ablation: beta sensitivity
    beta_runs = []
    for beta in [0.1, 0.5, 1.0, 2.0, 5.0]:
        run_id = f"beta_{beta}"
        args = base_cifar100_resnet20_args(42) + ["--eqwd_beta", str(beta), "--eqwd_ema_alpha", "0.9"]
        beta_runs.append((run_id, args))
    run_experiment("ablation_beta", beta_runs, "EqWD beta sensitivity on CIFAR-100/ResNet-20")

    # 2. Ablation: EMA alpha sensitivity
    ema_runs = []
    for alpha in [0.8, 0.9, 0.95, 0.99]:
        run_id = f"ema_{alpha}"
        args = base_cifar100_resnet20_args(42) + ["--eqwd_beta", "1.0", "--eqwd_ema_alpha", str(alpha)]
        ema_runs.append((run_id, args))
    run_experiment("ablation_ema", ema_runs, "EqWD EMA alpha sensitivity on CIFAR-100/ResNet-20")

    # 3. No-BN ablation: All 7 methods on VGG-16 without BN
    nobn_runs = []
    methods = ["NoWD", "FixedWD", "SWD", "CWD", "CPR", "CAWD", "EqWD"]
    for method in methods:
        for seed in [42, 123, 456]:
            run_id = f"{method}_seed{seed}"
            args = base_cifar100_vgg16_nobn_args(seed) + ["--wd_method", method]
            if method == "EqWD":
                args += ["--eqwd_beta", "2.0", "--eqwd_ema_alpha", "0.9"]
            nobn_runs.append((run_id, args))
    run_experiment("nobn_ablation", nobn_runs, "All 7 methods on CIFAR-100/VGG-16 (NO BN)")

    # 4. Control: phase schedule (fixed WD vs EqWD with same total regularization budget)
    control_runs = []
    for seed in [42, 123, 456]:
        # EqWD with phase scheduling: high WD early, low WD late
        run_id = f"phase_eqwd_seed{seed}"
        args = base_cifar100_resnet20_args(seed) + ["--eqwd_beta", "1.0", "--eqwd_ema_alpha", "0.9"]
        control_runs.append((run_id, args))
    run_experiment("control_phase_schedule", control_runs, "Phase schedule control on CIFAR-100/ResNet-20")

    elapsed = (time.time() - master_start) / 60
    print(f"\n{'='*60}")
    print(f"All ablation experiments complete in {elapsed:.1f} min")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
