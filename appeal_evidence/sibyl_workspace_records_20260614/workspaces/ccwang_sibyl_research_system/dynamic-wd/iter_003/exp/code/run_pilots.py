#!/usr/bin/env python3
"""
Run all pilot experiments for the Unified Dynamic WD Framework.

Wave 1: AdamW constant baseline (1 GPU)
Wave 2: CWD, SWD, Cosine-WD in parallel (3 GPUs)
Wave 3: Metrics validation + Soft CWD sweep (1 GPU)
"""

import json
import subprocess
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

CODE_DIR = Path(__file__).parent
EXP_DIR = CODE_DIR.parent
RESULTS_DIR = EXP_DIR / "results" / "pilots"


def run_experiment(name, args, gpu_id):
    """Run a single experiment as a subprocess."""
    output_dir = str(RESULTS_DIR / name)
    cmd = [
        sys.executable, str(CODE_DIR / "train_unified.py"),
        "--output_dir", output_dir,
        "--gpu_id", str(gpu_id),
    ] + args

    print(f"[{name}] Starting on GPU {gpu_id}: {' '.join(cmd[-10:])}")
    start = time.time()

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    )

    elapsed = time.time() - start
    success = result.returncode == 0

    if not success:
        print(f"[{name}] FAILED after {elapsed:.0f}s")
        print(f"  stderr: {result.stderr[-500:]}")
        # Save error log
        err_path = Path(output_dir)
        err_path.mkdir(parents=True, exist_ok=True)
        (err_path / "error.log").write_text(result.stderr)
    else:
        print(f"[{name}] Done in {elapsed:.0f}s")

    return name, success, elapsed


def check_kill_criteria(results):
    """Check pilot kill criteria. Returns (pass, messages)."""
    messages = []
    all_pass = True

    # 1. AdamW baseline > 91%
    baseline_path = RESULTS_DIR / "pilot_baseline" / "summary.json"
    if baseline_path.exists():
        summary = json.loads(baseline_path.read_text())
        acc = summary.get('best_test_acc', 0)
        if acc < 91.0:
            messages.append(f"KILL: AdamW baseline {acc:.2f}% < 91% — codebase bug")
            all_pass = False
        else:
            messages.append(f"OK: AdamW baseline {acc:.2f}% >= 91%")
    else:
        messages.append("KILL: AdamW baseline summary not found")
        all_pass = False

    # 2. CWD mask ratio in [0.1, 0.9]
    cwd_path = RESULTS_DIR / "pilot_cwd" / "epoch_metrics.jsonl"
    if cwd_path.exists():
        lines = cwd_path.read_text().strip().split('\n')
        if lines:
            last = json.loads(lines[-1])
            mask_ratio = last.get('mask_ratio', last.get('avg_mask_ratio', 0.5))
            if mask_ratio < 0.1 or mask_ratio > 0.9:
                messages.append(f"KILL: CWD mask ratio {mask_ratio:.3f} outside [0.1, 0.9]")
                all_pass = False
            else:
                messages.append(f"OK: CWD mask ratio {mask_ratio:.3f}")

    # 3. CSI differentiation (check from metrics)
    method_csi = {}
    for name in ["pilot_baseline", "pilot_cwd", "pilot_swd", "pilot_coswd"]:
        ep_path = RESULTS_DIR / name / "epoch_metrics.jsonl"
        if ep_path.exists():
            lines = ep_path.read_text().strip().split('\n')
            if lines:
                last = json.loads(lines[-1])
                method_csi[name] = last.get('csi', 0.0)

    if len(method_csi) >= 3:
        vals = list(method_csi.values())
        mean_csi = sum(vals) / len(vals)
        if mean_csi > 1e-6:
            cv = (sum((v - mean_csi)**2 for v in vals) / len(vals))**0.5 / mean_csi
        else:
            cv = 0.0
        if cv < 0.05:
            messages.append(f"WARNING: CSI CV={cv:.4f} < 5% — metric may be degenerate")
        else:
            messages.append(f"OK: CSI CV={cv:.4f} — methods differentiated")

    return all_pass, messages


def main():
    """Execute all pilot waves."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Available GPUs (pick less busy ones: 1, 4 are free, 0, 3 have low usage)
    available_gpus = [1, 4, 0, 3]

    print("=" * 60)
    print("Unified Dynamic WD Framework — Pilot Experiments")
    print("=" * 60)

    all_results = {}

    # ──── Wave 1: Baseline ────
    print("\n▶ Wave 1: AdamW constant baseline")
    name, success, elapsed = run_experiment(
        "pilot_baseline",
        ["--arch", "resnet20", "--dataset", "cifar10",
         "--wd_method", "constant", "--epochs", "100",
         "--lr", "1e-3", "--wd", "5e-4", "--seed", "42",
         "--lr_schedule", "cosine"],
        gpu_id=available_gpus[0]
    )
    all_results[name] = {"success": success, "time": elapsed}

    if not success:
        print("FATAL: Baseline failed. Aborting pilots.")
        sys.exit(1)

    # ──── Wave 2: CWD, SWD, Cosine in parallel ────
    print("\n▶ Wave 2: CWD + SWD + Cosine WD (parallel)")
    wave2_tasks = [
        ("pilot_cwd", ["--arch", "resnet20", "--dataset", "cifar10",
                        "--wd_method", "cwd_hard", "--epochs", "100",
                        "--lr", "1e-3", "--wd", "5e-4", "--seed", "42",
                        "--lr_schedule", "cosine"],
         available_gpus[1]),
        ("pilot_swd", ["--arch", "resnet20", "--dataset", "cifar10",
                        "--wd_method", "swd", "--epochs", "100",
                        "--lr", "1e-3", "--wd", "5e-4", "--seed", "42",
                        "--swd_sensitivity", "1.0", "--lr_schedule", "cosine"],
         available_gpus[2]),
        ("pilot_coswd", ["--arch", "resnet20", "--dataset", "cifar10",
                          "--wd_method", "cosine_schedule", "--epochs", "100",
                          "--lr", "1e-3", "--wd", "5e-4", "--wd_min", "0",
                          "--seed", "42", "--lr_schedule", "cosine"],
         available_gpus[3]),
    ]

    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_experiment, name, args, gpu): name
            for name, args, gpu in wave2_tasks
        }
        for future in as_completed(futures):
            name, success, elapsed = future.result()
            all_results[name] = {"success": success, "time": elapsed}

    # ──── Wave 3: Soft CWD sweep ────
    print("\n▶ Wave 3: Soft CWD sweep (beta=[10, 50, 100, 500, 1000])")
    for beta in [10, 50, 100, 500, 1000]:
        name, success, elapsed = run_experiment(
            f"pilot_soft_cwd_beta{beta}",
            ["--arch", "resnet20", "--dataset", "cifar10",
             "--wd_method", "cwd_soft", "--epochs", "100",
             "--lr", "1e-3", "--wd", "5e-4", "--seed", "42",
             "--cwd_beta", str(beta), "--lr_schedule", "cosine"],
            gpu_id=available_gpus[0]
        )
        all_results[name] = {"success": success, "time": elapsed}

    # ──── Kill criteria check ────
    print("\n" + "=" * 60)
    print("Kill Criteria Check")
    print("=" * 60)
    all_pass, messages = check_kill_criteria(all_results)
    for msg in messages:
        print(f"  {msg}")

    # ──── Summary ────
    print("\n" + "=" * 60)
    print("Pilot Summary")
    print("=" * 60)

    # Collect results
    pilot_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kill_criteria_pass": all_pass,
        "kill_criteria_messages": messages,
        "experiments": {},
    }

    for name in ["pilot_baseline", "pilot_cwd", "pilot_swd", "pilot_coswd"]:
        summary_path = RESULTS_DIR / name / "summary.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            pilot_summary["experiments"][name] = {
                "best_test_acc": s.get("best_test_acc"),
                "final_csi": s.get("final_csi"),
                "final_ais": s.get("final_ais"),
                "final_bem": s.get("final_bem"),
                "time_sec": s.get("total_time_sec"),
            }
            print(f"  {name:20s}: acc={s.get('best_test_acc', 0):.2f}%  "
                  f"CSI={s.get('final_csi', 0):.4f}  "
                  f"AIS={s.get('final_ais', 0):.4f}  "
                  f"BEM={s.get('final_bem', 0):.4f}")

    # Soft CWD results
    print("\n  Soft CWD convergence to hard CWD:")
    hard_cwd_path = RESULTS_DIR / "pilot_cwd" / "summary.json"
    hard_acc = 0
    if hard_cwd_path.exists():
        hard_acc = json.loads(hard_cwd_path.read_text()).get('best_test_acc', 0)

    for beta in [10, 50, 100, 500, 1000]:
        sp = RESULTS_DIR / f"pilot_soft_cwd_beta{beta}" / "summary.json"
        if sp.exists():
            s = json.loads(sp.read_text())
            soft_acc = s.get('best_test_acc', 0)
            diff = abs(soft_acc - hard_acc)
            pilot_summary["experiments"][f"soft_cwd_beta{beta}"] = {
                "best_test_acc": soft_acc,
                "diff_from_hard": round(diff, 4),
            }
            status = "OK" if diff < 0.2 else "WARN"
            print(f"    beta={beta:4d}: acc={soft_acc:.2f}% (diff={diff:.2f}%) [{status}]")

    # Save overall pilot summary
    with open(RESULTS_DIR / "pilot_summary.json", 'w') as f:
        json.dump(pilot_summary, f, indent=2)

    if all_pass:
        print("\n✓ All kill criteria PASSED. Ready for full experiments.")
    else:
        print("\n✗ Some kill criteria FAILED. Review before proceeding.")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
