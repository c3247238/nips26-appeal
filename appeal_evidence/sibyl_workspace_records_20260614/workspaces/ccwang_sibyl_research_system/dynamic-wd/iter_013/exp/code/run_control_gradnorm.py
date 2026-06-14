#!/usr/bin/env python3
"""Control experiment: EqWD (ratio-based) vs GradNorm-only scheduling.

Tests H4: Does the gradient-to-weight RATIO provide value beyond gradient norm alone?

Compares:
1. EqWD: phi = 1 + beta * |r_t - r*|/r* where r_t = ||g||/||w||
2. SWD: phi = ||g|| / EMA(||g||) (gradient norm only, no weight info)
3. FixedWD: constant lambda (baseline)

CIFAR-100, ResNet-20, 3 seeds each.
"""

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
TASK_ID = "control_gradnorm_only"

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

def base_args(method, seed=42):
    args = ["--dataset", "cifar100", "--arch", "resnet20",
            "--epochs", "200", "--batch_size", "128",
            "--lr", "0.1", "--lr_schedule", "cosine",
            "--wd_method", method, "--wd_lambda", "5e-4",
            "--seed", str(seed), "--diag_interval", "10",
            "--data_dir", str(CODE_DIR / "data"), "--num_workers", "4"]
    if method == "EqWD":
        args.extend(["--eqwd_beta", "1.0", "--eqwd_ema_alpha", "0.9"])
    return args

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "4")
    task_dir = RESULTS_DIR / TASK_ID
    task_dir.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

    all_results = {}
    methods = ["FixedWD", "SWD", "EqWD"]

    for method in methods:
        print(f"\n=== {method} on CIFAR-100/ResNet-20 ===")
        for seed in [42, 123, 456]:
            run_id = f"{method}_seed{seed}"
            args = base_args(method, seed)
            r = run_single(run_id, args, str(task_dir / run_id))
            if r:
                all_results[run_id] = r

    # Analysis
    import numpy as np
    print("\n=== Control: GradNorm vs Ratio ===")
    for method in methods:
        accs = [all_results.get(f"{method}_seed{s}", {}).get("best_test_top1", 0)
                for s in [42, 123, 456]]
        print(f"  {method}: {np.mean(accs):.2f} ± {np.std(accs):.2f}%  {accs}")

    done = {
        "task_id": TASK_ID, "status": "success",
        "summary": f"{len(all_results)}/9 runs",
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps(done, indent=2))
    print(f"\n[DONE] {len(all_results)}/9 runs")

if __name__ == "__main__":
    main()
