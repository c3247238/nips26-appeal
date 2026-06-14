#!/usr/bin/env python3
"""Re-run budget equivalence best configs with correct seeds."""

import json
import os
import time
from datetime import datetime
from pathlib import Path

WORKSPACE = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
TASK_ID = "budget_equivalence"

# Import fixed run_trial
os.chdir(str(WORKSPACE / "exp" / "code"))
from run_budget_equivalence import run_trial

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "4")

# Load existing search results
existing = json.loads((RESULTS_DIR / "full" / "budget_equivalence" / "budget_equivalence_results.json").read_text())
all_results = existing["search_results"]

print("=== Re-running best configs with correct seeds ===")
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
print("\n=== Budget Equivalence Results (Fixed Seeds) ===")
fixed_mean = seed_results["FixedWD"]["mean"]
for method, data in seed_results.items():
    delta = data["mean"] - fixed_mean
    print(f"  {method}: {data['mean']:.2f}±{data['std']:.2f}% "
          f"(Δ vs FixedWD: {delta:+.2f}%)")

# Update results file
existing["seed_results"] = seed_results
existing["conclusion"] = {
    "fixed_wd_mean": fixed_mean,
    "dynamic_beats_fixed": any(
        seed_results[m]["mean"] > fixed_mean + 0.5
        for m in ["SWD", "CWD", "EqWD"]
    ),
}
existing["timestamp"] = datetime.now().isoformat()
(RESULTS_DIR / "full" / "budget_equivalence" / "budget_equivalence_results.json").write_text(
    json.dumps(existing, indent=2))

# Update DONE marker
done = {
    "task_id": TASK_ID, "status": "success",
    "summary": f"4 methods × 15 trials + 3-seed re-run (fixed)",
    "timestamp": datetime.now().isoformat(),
}
(RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps(done, indent=2))
print("\n[DONE] Budget equivalence re-evaluation complete")
