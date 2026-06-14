"""
IGSD Split Runner: runs one seed shard of the igsd_pareto_corrected FULL sweep.
Usage: CUDA_VISIBLE_DEVICES=X python igsd_split_runner.py --seed 42

Each shard writes to igsd_pareto/partial_seed_<N>.json.
After all shards finish, run igsd_merge.py to combine.
"""
import os, sys, json, time, random, gc, argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
CODE_DIR  = WORKSPACE / "exp" / "code"
RESULTS_DIR = WORKSPACE / "exp" / "results" / "igsd_pareto"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

# Import from the main IGSD FULL script
from igsd_pareto_corrected_full import (
    IGSDGenerator, evaluate_gsm8k, evaluate_math500,
    load_gsm8k_data, load_math500_data,
    TAU_VALUES, T_DRAFT_VALUES, T_FULL, GEN_LENGTH, BLOCK_LENGTH,
    BASELINE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    seed = args.seed
    task_id = f"igsd_shard_s{seed}"
    start_time = datetime.now()

    print(f"[{task_id}] Starting seed={seed} at {start_time.isoformat()}", flush=True)
    (RESULTS_DIR / f"{task_id}.pid").write_text(str(os.getpid()))

    # Load model
    device = "cuda:0"
    print(f"[{task_id}] Loading model...", flush=True)
    generator = IGSDGenerator(device=device)
    print(f"[{task_id}] Model loaded.", flush=True)

    # Load data
    gsm8k_data = load_gsm8k_data()
    math500_data = load_math500_data()
    print(f"[{task_id}] GSM8K={len(gsm8k_data)}, MATH500={len(math500_data)}", flush=True)

    # Check partial results
    partial_path = RESULTS_DIR / f"igsd_partial_seed_{seed}.json"
    shard_results = {}
    if partial_path.exists():
        try:
            shard_results = json.loads(partial_path.read_text())
            print(f"[{task_id}] Resuming from partial ({len(shard_results)} configs done)", flush=True)
        except:
            shard_results = {}

    total_configs = len(TAU_VALUES) * len(T_DRAFT_VALUES)
    completed = 0

    for tau in TAU_VALUES:
        for t_draft in T_DRAFT_VALUES:
            config_key = f"tau{tau}_td{t_draft}"
            if config_key in shard_results:
                print(f"  [skip] {config_key} already done", flush=True)
                completed += 1
                continue

            print(f"\n[{task_id}] === {config_key} (seed={seed}) ===", flush=True)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            config_result = {"tau": tau, "t_draft": t_draft, "seed": seed}

            # GSM8K
            try:
                gsm_r = evaluate_gsm8k(generator, gsm8k_data, tau, t_draft, seed)
                config_result["gsm8k"] = gsm_r
                gsm_speedup = gsm_r["avg_tps"] / BASELINE["gsm8k"]["avg_tps"] if BASELINE["gsm8k"]["avg_tps"] > 0 else 0
                gsm_acc_ret = gsm_r["exact_match"] / BASELINE["gsm8k"]["exact_match"] if BASELINE["gsm8k"]["exact_match"] > 0 else 0
                print(f"  GSM8K: acc={gsm_r['exact_match']:.3f}, tps={gsm_r['avg_tps']:.1f}, speedup={gsm_speedup:.2f}x", flush=True)
            except Exception as e:
                print(f"  GSM8K ERROR: {e}", flush=True)
                config_result["gsm8k"] = {"error": str(e)[:200]}
            gc.collect(); torch.cuda.empty_cache()

            # MATH500
            try:
                math_r = evaluate_math500(generator, math500_data, tau, t_draft, seed)
                config_result["math500"] = math_r
                print(f"  MATH500: acc={math_r['exact_match']:.3f}, tps={math_r['avg_tps']:.1f}", flush=True)
            except Exception as e:
                print(f"  MATH500 ERROR: {e}", flush=True)
                config_result["math500"] = {"error": str(e)[:200]}
            gc.collect(); torch.cuda.empty_cache()

            shard_results[config_key] = config_result
            partial_path.write_text(json.dumps(shard_results, indent=2))
            completed += 1

            # Progress
            (RESULTS_DIR / f"{task_id}_PROGRESS.json").write_text(json.dumps({
                "task_id": task_id, "step": completed, "total_steps": total_configs,
                "updated_at": datetime.now().isoformat(),
                "metric": {"tau": tau, "t_draft": t_draft, "seed": seed}
            }))
            print(f"  [checkpoint] {config_key} saved ({completed}/{total_configs})", flush=True)

    elapsed_min = (datetime.now() - start_time).total_seconds() / 60
    pid_f = RESULTS_DIR / f"{task_id}.pid"
    if pid_f.exists(): pid_f.unlink()
    (RESULTS_DIR / f"{task_id}_DONE").write_text(json.dumps({
        "task_id": task_id, "status": "success", "seed": seed,
        "elapsed_minutes": round(elapsed_min, 1),
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"[{task_id}] Done in {elapsed_min:.1f} min", flush=True)


if __name__ == "__main__":
    main()
