"""
M1 Split Runner: runs one seed shard of the m1_pareto_corrected sweep.
Usage: CUDA_VISIBLE_DEVICES=X python m1_split_runner.py --seed 42 [--skip-humaneval]

Each shard writes to m1_pareto/partial_seed_<N>.json.
After all shards finish, run m1_merge.py to combine.
"""
import os, sys, json, time, random, gc, re, subprocess, argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE     = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
SHARED        = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared")
MODEL_PATH    = str(SHARED / "checkpoints" / "llada-8b-instruct")
GSM8K_DIR     = str(SHARED / "datasets" / "gsm8k")
MATH500_DIR   = str(SHARED / "datasets" / "math500")
HUMANEVAL_DIR = str(SHARED / "datasets" / "humaneval")
CODE_DIR      = WORKSPACE / "exp" / "code"
RESULTS_DIR   = WORKSPACE / "exp" / "results" / "m1_pareto"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

LLADA_MASK_ID = 126336
BATCH_SIZE_GSM8K = 8      # 2x from original 4
BATCH_SIZE_MATH500 = 4    # 2x from original 2
BATCH_SIZE_HE = 8         # 2x from original 4

BASELINE = {
    "gsm8k":     {"exact_match": 0.7122, "avg_tps": 31.013},
    "math500":   {"exact_match": 0.1107, "avg_tps": 79.221},
    "humaneval": {"pass_at_1":   0.0244, "avg_tps": 97.999},
}

D2CACHE_CHR = {
    0.5: {"gsm8k_chr": 0.933, "math500_chr": 0.871, "theoretical_speedup": 2.27},
    1.0: {"gsm8k_chr": 0.970, "math500_chr": 0.935, "theoretical_speedup": 2.39},
    2.0: {"gsm8k_chr": 0.991, "math500_chr": 0.980, "theoretical_speedup": 2.47},
}

ENTROPY_THRESHOLDS = [0.5, 1.0, 2.0]

# ── Reuse generation + eval functions from main script ────────────────────────
# Import from the main m1 script
sys.path.insert(0, str(CODE_DIR))
from m1_pareto_corrected import (
    batched_generate_m1, eval_gsm8k_m1, eval_math500_m1, eval_humaneval_m1,
    load_gsm8k, load_math500, load_humaneval,
    profile_vram
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--skip-humaneval", action="store_true")
    args = parser.parse_args()

    seed = args.seed
    task_id = f"m1_shard_s{seed}"
    start_time = datetime.now()

    print(f"[{task_id}] Starting seed={seed} at {start_time.isoformat()}", flush=True)
    print(f"[{task_id}] batch_size GSM8K={BATCH_SIZE_GSM8K}, MATH500={BATCH_SIZE_MATH500}", flush=True)

    # Write PID
    (RESULTS_DIR / f"{task_id}.pid").write_text(str(os.getpid()))

    device = "cuda:0"
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # Load model
    print(f"[{task_id}] Loading LLaDA-8B-Instruct...", flush=True)
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    vram = profile_vram(device)
    print(f"[{task_id}] Model loaded. VRAM: {vram.get('vram_used_mb',0)} MB", flush=True)

    # Load datasets
    gsm8k_data = load_gsm8k()
    math500_data = load_math500()
    he_data = load_humaneval() if not args.skip_humaneval else []
    print(f"[{task_id}] GSM8K={len(gsm8k_data)}, MATH500={len(math500_data)}, HE={len(he_data)}", flush=True)

    # Check partial
    partial_path = RESULTS_DIR / f"partial_seed_{seed}.json"
    shard_results = {}
    if partial_path.exists():
        try:
            shard_results = json.loads(partial_path.read_text())
            print(f"[{task_id}] Resuming from partial", flush=True)
        except:
            shard_results = {}

    benchmarks = ["gsm8k", "math500"] + (["humaneval"] if not args.skip_humaneval else [])
    total_steps = len(ENTROPY_THRESHOLDS) * len(benchmarks)
    completed = 0

    for t_idx, threshold in enumerate(ENTROPY_THRESHOLDS):
        tkey = str(threshold)
        if tkey in shard_results:
            print(f"  [skip] threshold={threshold} already done", flush=True)
            completed += len(benchmarks)
            continue

        print(f"\n[{task_id}] === Threshold {threshold} ({t_idx+1}/{len(ENTROPY_THRESHOLDS)}) ===", flush=True)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        seed_results = {}

        # GSM8K
        try:
            gsm_r = eval_gsm8k_m1(model, tokenizer, gsm8k_data, seed, threshold, device,
                                   batch_size=BATCH_SIZE_GSM8K, n_warmup=2, tag=f"_t{threshold}_s{seed}")
            seed_results["gsm8k"] = gsm_r
            completed += 1
            print(f"    GSM8K done: acc={gsm_r['exact_match']:.3f}, speedup={gsm_r['speedup']:.2f}x", flush=True)
        except Exception as e:
            print(f"    GSM8K ERROR: {e}", flush=True)
            seed_results["gsm8k"] = {"error": str(e)[:200], "exact_match": 0, "avg_tps": 0, "speedup": 0, "accuracy_retention": 0}
            completed += 1
        gc.collect(); torch.cuda.empty_cache()

        # MATH500
        try:
            math_r = eval_math500_m1(model, tokenizer, math500_data, seed, threshold, device,
                                      batch_size=BATCH_SIZE_MATH500, n_warmup=2, tag=f"_t{threshold}_s{seed}")
            seed_results["math500"] = math_r
            completed += 1
            print(f"    MATH500 done: acc={math_r['exact_match']:.3f}, speedup={math_r['speedup']:.2f}x", flush=True)
        except Exception as e:
            print(f"    MATH500 ERROR: {e}", flush=True)
            seed_results["math500"] = {"error": str(e)[:200], "exact_match": 0, "avg_tps": 0, "speedup": 0, "accuracy_retention": 0}
            completed += 1
        gc.collect(); torch.cuda.empty_cache()

        # HumanEval (optional)
        if not args.skip_humaneval:
            try:
                he_r = eval_humaneval_m1(model, tokenizer, he_data, seed, threshold, device,
                                          batch_size=BATCH_SIZE_HE, n_warmup=2, tag=f"_t{threshold}_s{seed}")
                seed_results["humaneval"] = he_r
                completed += 1
                print(f"    HumanEval done: pass@1={he_r['pass_at_1']:.3f}", flush=True)
            except Exception as e:
                print(f"    HumanEval ERROR: {e}", flush=True)
                seed_results["humaneval"] = {"error": str(e)[:200], "pass_at_1": 0, "avg_tps": 0, "speedup": 0, "accuracy_retention": 0}
                completed += 1
            gc.collect(); torch.cuda.empty_cache()

        shard_results[tkey] = seed_results
        partial_path.write_text(json.dumps(shard_results, indent=2))

        # Progress
        (RESULTS_DIR / f"{task_id}_PROGRESS.json").write_text(json.dumps({
            "task_id": task_id, "step": completed, "total_steps": total_steps,
            "updated_at": datetime.now().isoformat(),
            "metric": {"threshold": threshold, "seed": seed, "completed": completed}
        }))
        print(f"  [checkpoint] threshold={threshold} saved ({completed}/{total_steps})", flush=True)

    elapsed_min = (datetime.now() - start_time).total_seconds() / 60

    # Write DONE marker
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
