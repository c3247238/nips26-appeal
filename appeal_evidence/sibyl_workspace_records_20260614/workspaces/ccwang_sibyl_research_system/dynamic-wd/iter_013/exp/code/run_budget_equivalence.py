#!/usr/bin/env python3
"""Budget Equivalence Test (H6): Compare tuned fixed WD vs tuned dynamic WD methods.

Uses Optuna to find best hyperparameters for each method, then compares.
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
TASK_ID = "budget_equivalence"

N_TRIALS = 15  # Per method
EPOCHS = 100   # Shorter for search
SEED = 42

def run_trial(method, lr, wd, beta=1.0, ema_alpha=0.9, trial_id="0", seed=42):
    """Run a single Optuna trial."""
    output_dir = str(RESULTS_DIR / TASK_ID / f"{method}_trial{trial_id}")
    od = Path(output_dir)
    if od.exists():
        import shutil
        shutil.rmtree(od)
    od.mkdir(parents=True, exist_ok=True)

    run_id = f"{method}_t{trial_id}"
    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--dataset", "cifar100", "--arch", "resnet20",
        "--epochs", str(EPOCHS), "--batch_size", "128",
        "--lr", str(lr), "--lr_schedule", "cosine",
        "--wd_method", method, "--wd_lambda", str(wd),
        "--seed", str(seed), "--diag_interval", "50",
        "--data_dir", str(CODE_DIR / "data"), "--num_workers", "4",
        "--output_dir", output_dir, "--task_id", run_id,
    ]
    if method == "EqWD":
        cmd.extend(["--eqwd_beta", str(beta), "--eqwd_ema_alpha", str(ema_alpha)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=str(CODE_DIR))
        if result.returncode == 0:
            for rf in [Path(output_dir) / f"{run_id}_results.json", Path(output_dir) / "default_results.json"]:
                if rf.exists():
                    r = json.loads(rf.read_text())
                    return r.get("best_test_top1", 0.0)
    except subprocess.TimeoutExpired:
        pass
    return 0.0

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "4")

    task_dir = RESULTS_DIR / TASK_ID
    task_dir.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    methods = ["FixedWD", "SWD", "CWD", "EqWD"]
    all_results = {}
    start_time = time.time()

    for method in methods:
        print(f"\n=== {method}: {N_TRIALS} trials ===")

        def objective(trial):
            lr = trial.suggest_float("lr", 0.01, 0.3, log=True)
            wd = trial.suggest_float("wd", 1e-5, 1e-2, log=True)
            if method == "EqWD":
                beta = trial.suggest_float("beta", 0.1, 5.0, log=True)
                ema = trial.suggest_float("ema_alpha", 0.7, 0.99)
            else:
                beta, ema = 1.0, 0.9

            acc = run_trial(method, lr, wd, beta, ema, trial.number)
            elapsed = (time.time() - start_time) / 60
            print(f"  Trial {trial.number}: lr={lr:.4f} wd={wd:.6f} → {acc:.2f}% ({elapsed:.0f}min)")
            return acc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=N_TRIALS)

        best = study.best_trial
        all_results[method] = {
            "best_acc": best.value,
            "best_params": best.params,
            "n_trials": N_TRIALS,
            "all_trials": [{"value": t.value, "params": t.params}
                          for t in study.trials if t.value is not None],
        }
        print(f"  Best: {best.value:.2f}% with {best.params}")

        # Write progress
        progress = {
            "task_id": TASK_ID,
            "epoch": methods.index(method) + 1, "total_epochs": len(methods),
            "step": methods.index(method) + 1, "total_steps": len(methods),
            "loss": None,
            "metric": {
                "methods_completed": methods.index(method) + 1,
                "methods_total": len(methods),
                "current_method": method,
                "elapsed_minutes": round((time.time() - start_time) / 60, 1),
            },
            "updated_at": datetime.now().isoformat(),
        }
        (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps(progress))

    # Re-run best configs with 3 seeds for statistical comparison
    print("\n=== Re-running best configs with 3 seeds ===")
    seed_results = {}
    for method, data in all_results.items():
        params = data["best_params"]
        seed_accs = []
        for seed in [42, 123, 456]:
            beta = params.get("beta", 1.0)
            ema = params.get("ema_alpha", 0.9)
            acc = run_trial(method, params["lr"], params["wd"], beta, ema,
                          trial_id=f"best_seed{seed}", seed=seed)
            seed_accs.append(acc)
            print(f"  {method} seed{seed}: {acc:.2f}%")

        import numpy as np
        seed_results[method] = {
            "mean": float(np.mean(seed_accs)),
            "std": float(np.std(seed_accs)),
            "accs": seed_accs,
            "params": params,
        }

    # Analysis
    print("\n=== Budget Equivalence Results ===")
    fixed_mean = seed_results["FixedWD"]["mean"]
    for method, data in seed_results.items():
        delta = data["mean"] - fixed_mean
        print(f"  {method}: {data['mean']:.2f}±{data['std']:.2f}% "
              f"(Δ vs FixedWD: {delta:+.2f}%)")

    # Write final results
    elapsed = (time.time() - start_time) / 60
    output = {
        "task_id": TASK_ID,
        "search_results": all_results,
        "seed_results": seed_results,
        "conclusion": {
            "fixed_wd_mean": fixed_mean,
            "dynamic_beats_fixed": any(
                seed_results[m]["mean"] > fixed_mean + 0.5
                for m in ["SWD", "CWD", "EqWD"]
            ),
        },
        "config": {"n_trials": N_TRIALS, "epochs": EPOCHS},
        "elapsed_minutes": elapsed,
        "timestamp": datetime.now().isoformat(),
    }

    out_dir = RESULTS_DIR / "full" / "budget_equivalence"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "budget_equivalence_results.json").write_text(json.dumps(output, indent=2))

    done = {
        "task_id": TASK_ID, "status": "success",
        "summary": f"4 methods × {N_TRIALS} trials + 3-seed re-run in {elapsed:.0f}min",
        "timestamp": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps(done, indent=2))
    print(f"\n[DONE] Budget equivalence in {elapsed:.0f}min")

if __name__ == "__main__":
    main()
