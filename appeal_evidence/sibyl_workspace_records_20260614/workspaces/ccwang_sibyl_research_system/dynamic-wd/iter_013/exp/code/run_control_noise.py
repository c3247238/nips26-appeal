#!/usr/bin/env python3
"""Control experiment: Real alignment vs noise injection.

Tests H5: Does CAWD's alignment signal contain real information, or would random
noise scheduling perform similarly?

Compares:
1. CAWD (sigma=0): Real alignment signal
2. CAWD (sigma=0.5): Alignment + noise
3. CAWD (sigma=1.0): Alignment + heavy noise (dominates signal)
4. FixedWD: Baseline

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
TASK_ID = "control_noise_injection"

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

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400, cwd=str(CODE_DIR))
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
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "4")
    task_dir = RESULTS_DIR / TASK_ID
    task_dir.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

    all_results = {}
    configs = [
        ("FixedWD", 0.0, "baseline"),
        ("CAWD", 0.0, "real_alignment"),
        ("CAWD", 0.5, "noisy_alignment"),
        ("CAWD", 1.0, "heavy_noise"),
    ]

    for method, sigma, label in configs:
        print(f"\n=== {label} ({method}, sigma={sigma}) ===")
        for seed in [42, 123, 456]:
            run_id = f"{label}_seed{seed}"
            args = ["--dataset", "cifar100", "--arch", "resnet20",
                    "--epochs", "200", "--batch_size", "128",
                    "--lr", "0.1", "--lr_schedule", "cosine",
                    "--wd_method", method, "--wd_lambda", "5e-4",
                    "--seed", str(seed), "--diag_interval", "10",
                    "--data_dir", str(CODE_DIR / "data"), "--num_workers", "4"]
            if sigma > 0:
                args.extend(["--cawd_noise_sigma", str(sigma)])
            r = run_single(run_id, args, str(task_dir / run_id))
            if r:
                all_results[run_id] = r

    # Analysis
    import numpy as np
    print("\n=== Control: Noise Injection ===")
    for _, _, label in configs:
        accs = [all_results.get(f"{label}_seed{s}", {}).get("best_test_top1", 0)
                for s in [42, 123, 456]]
        print(f"  {label}: {np.mean(accs):.2f} ± {np.std(accs):.2f}%  {accs}")

    done = {
        "task_id": TASK_ID, "status": "success",
        "summary": f"{len(all_results)}/12 runs",
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps(done, indent=2))
    print(f"\n[DONE] {len(all_results)}/12 runs")

if __name__ == "__main__":
    main()
