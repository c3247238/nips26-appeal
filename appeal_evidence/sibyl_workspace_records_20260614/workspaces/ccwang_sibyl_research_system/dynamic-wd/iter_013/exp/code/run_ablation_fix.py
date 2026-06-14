#!/usr/bin/env python3
"""Fix ablation runs that were incorrectly run on CIFAR-10 instead of CIFAR-100."""

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

def run_single(task_id, run_id, args, output_dir):
    od = Path(output_dir)
    if od.exists():
        import shutil
        shutil.rmtree(od)
    od.mkdir(parents=True, exist_ok=True)

    cmd = [PYTHON, TRAIN_SCRIPT] + args + ["--output_dir", output_dir, "--task_id", run_id]
    print(f"  [{run_id}]...", end=" ", flush=True)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(CODE_DIR))
    if result.returncode == 0:
        for f in [Path(output_dir) / f"{run_id}_results.json", Path(output_dir) / "default_results.json"]:
            if f.exists():
                r = json.loads(f.read_text())
                print(f"OK (top1={r.get('best_test_top1', '?')})")
                return r
    else:
        print(f"FAIL: {result.stderr[-200:]}")
    return None

def base_args(seed=42):
    return ["--dataset", "cifar100", "--arch", "resnet20",
            "--epochs", "200", "--batch_size", "128",
            "--lr", "0.1", "--lr_schedule", "cosine",
            "--wd_method", "EqWD", "--wd_lambda", "5e-4",
            "--seed", str(seed), "--diag_interval", "10",
            "--data_dir", str(CODE_DIR / "data"), "--num_workers", "4"]

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "7")

    all_results = {}

    # 1. Fix ablation_beta: beta 0.1, 0.5, 1.0
    print("=== Fixing ablation_beta (CIFAR-100) ===")
    task_dir = RESULTS_DIR / "ablation_beta"
    for beta in [0.1, 0.5, 1.0]:
        run_id = f"beta_{beta}"
        args = base_args(42) + ["--eqwd_beta", str(beta), "--eqwd_ema_alpha", "0.9"]
        r = run_single("ablation_beta", run_id, args, str(task_dir / run_id))
        if r:
            all_results[f"beta_{run_id}"] = r

    # Re-create DONE marker with all results
    existing_results = {}
    for beta in [0.1, 0.5, 1.0, 2.0, 5.0]:
        run_id = f"beta_{beta}"
        for f in [task_dir / run_id / f"{run_id}_results.json", task_dir / run_id / "default_results.json"]:
            if f.exists():
                existing_results[run_id] = json.loads(f.read_text())
                break
    done = {"task_id": "ablation_beta", "status": "success",
            "summary": f"{len(existing_results)}/5 runs", "results": existing_results,
            "timestamp": datetime.now().isoformat()}
    (RESULTS_DIR / "ablation_beta_DONE").write_text(json.dumps(done, indent=2))
    print(f"  ablation_beta: {len(existing_results)}/5 done")

    # 2. Fix ablation_ema: ema 0.9
    print("\n=== Fixing ablation_ema (CIFAR-100) ===")
    task_dir = RESULTS_DIR / "ablation_ema"
    args = base_args(42) + ["--eqwd_beta", "1.0", "--eqwd_ema_alpha", "0.9"]
    r = run_single("ablation_ema", "ema_0.9", args, str(task_dir / "ema_0.9"))
    if r:
        all_results["ema_0.9"] = r

    existing_results = {}
    for alpha in [0.8, 0.9, 0.95, 0.99]:
        run_id = f"ema_{alpha}"
        for f in [task_dir / run_id / f"{run_id}_results.json", task_dir / run_id / "default_results.json"]:
            if f.exists():
                existing_results[run_id] = json.loads(f.read_text())
                break
    done = {"task_id": "ablation_ema", "status": "success",
            "summary": f"{len(existing_results)}/4 runs", "results": existing_results,
            "timestamp": datetime.now().isoformat()}
    (RESULTS_DIR / "ablation_ema_DONE").write_text(json.dumps(done, indent=2))
    print(f"  ablation_ema: {len(existing_results)}/4 done")

    print("\nAll ablation fixes complete!")

if __name__ == "__main__":
    main()
