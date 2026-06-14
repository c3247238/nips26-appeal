#!/usr/bin/env python3
"""Ablation: Layer-type-aware vs uniform EqWD on CIFAR-100/VGG-16-BN, 3 seeds."""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

WORKSPACE = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current")
CODE_DIR = WORKSPACE / "exp" / "code"
RESULTS_DIR = WORKSPACE / "exp" / "results"
PYTHON = "/home/ccwang/sibyl-research-system/.venv/bin/python3"
TRAIN_SCRIPT = str(CODE_DIR / "train.py")
TASK_ID = "ablation_layertype"

def base_args(seed=42):
    return ["--dataset", "cifar100", "--arch", "vgg16bn",
            "--epochs", "50", "--batch_size", "128",
            "--lr", "0.1", "--lr_schedule", "cosine",
            "--wd_method", "EqWD", "--wd_lambda", "5e-4",
            "--eqwd_beta", "2.0", "--eqwd_ema_alpha", "0.9",
            "--seed", str(seed), "--diag_interval", "10",
            "--data_dir", str(CODE_DIR / "data"), "--num_workers", "4"]

def run_single(run_id, args, output_dir):
    od = Path(output_dir)
    result_files = [od / f"{run_id}_results.json", od / "default_results.json"]
    if any(f.exists() for f in result_files):
        print(f"  [SKIP] {run_id}")
        for f in result_files:
            if f.exists():
                return json.loads(f.read_text())
        return None

    if od.exists():
        import shutil
        shutil.rmtree(od)
    od.mkdir(parents=True, exist_ok=True)

    cmd = [PYTHON, TRAIN_SCRIPT] + args + ["--output_dir", output_dir, "--task_id", run_id]
    print(f"  [{run_id}]...", end=" ", flush=True)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=str(CODE_DIR))
    if result.returncode == 0:
        for f in result_files:
            if f.exists():
                r = json.loads(f.read_text())
                print(f"OK (top1={r.get('best_test_top1', '?')})")
                return r
    else:
        print(f"FAIL: {result.stderr[-200:]}")
    return None

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "7")
    task_dir = RESULTS_DIR / TASK_ID
    task_dir.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

    all_results = {}

    # Mode 1: Uniform EqWD (default)
    print("=== Uniform EqWD on CIFAR-100/VGG-16-BN ===")
    for seed in [42, 123, 456]:
        run_id = f"uniform_seed{seed}"
        args = base_args(seed)
        r = run_single(run_id, args, str(task_dir / run_id))
        if r:
            all_results[run_id] = r

    # Mode 2: Layer-type-aware EqWD
    print("\n=== Layer-type-aware EqWD on CIFAR-100/VGG-16-BN ===")
    for seed in [42, 123, 456]:
        run_id = f"layeraware_seed{seed}"
        args = base_args(seed) + ["--eqwd_layer_aware"]
        r = run_single(run_id, args, str(task_dir / run_id))
        if r:
            all_results[run_id] = r

    # Write DONE
    done = {
        "task_id": TASK_ID, "status": "success",
        "summary": f"{len(all_results)}/6 runs",
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps(done, indent=2))
    print(f"\n[DONE] {len(all_results)}/6 runs")

if __name__ == "__main__":
    main()
