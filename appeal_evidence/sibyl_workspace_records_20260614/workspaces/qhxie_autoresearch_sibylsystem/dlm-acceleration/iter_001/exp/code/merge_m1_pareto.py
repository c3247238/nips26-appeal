import json, sys, numpy as np
from pathlib import Path
from datetime import datetime

WORKSPACE   = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full_m1"
ENTROPY_THRESHOLDS = [0.5, 1.0, 2.0, 3.0]
SEEDS = [42, 123, 456]
BASELINE = {
    "gsm8k":     {"exact_match": 0.7122, "avg_tps": 31.013},
    "math500":   {"exact_match": 0.1107, "avg_tps": 79.221},
    "humaneval": {"pass_at_1":   0.0244, "avg_tps": 97.999},
    "mbpp":      {"pass_at_1":   0.0000, "avg_tps": 191.586},
}
benchmarks = ["gsm8k", "math500", "humaneval", "mbpp"]
acc_key = {"gsm8k": "exact_match", "math500": "exact_match", "humaneval": "pass_at_1", "mbpp": "pass_at_1"}
weights = {"gsm8k": 1319, "math500": 500, "humaneval": 164, "mbpp": 374}

def load_json(path):
    try: return json.loads(path.read_text()) if path.exists() else {}
    except: return {}

def merge_and_aggregate():
    all_data = {}

    # Base partial has thresholds 0.5 and 1.0
    base = load_json(RESULTS_DIR / "m1_pareto_partial.json")
    all_data.update(base)

    # t2 and t3 wrote full output with per_threshold_per_seed
    for tag, thresh in [("t2", "2.0"), ("t3", "3.0")]:
        full = load_json(RESULTS_DIR / f"m1_pareto_full_{tag}.json")
        if full and "per_threshold_per_seed" in full:
            src = full["per_threshold_per_seed"].get(thresh, {})
            if src:
                all_data[thresh] = src
                print(f"  Loaded threshold={thresh} from m1_pareto_full_{tag}.json")
        # Also try own partial if it exists
        partial = load_json(RESULTS_DIR / f"m1_pareto_partial_{tag}.json")
        if partial.get(thresh):
            all_data[thresh] = partial[thresh]

    print(f"Merged thresholds: {list(all_data.keys())}")
    missing = [str(t) for t in ENTROPY_THRESHOLDS if str(t) not in all_data]
    if missing:
        print(f"ERROR: missing thresholds: {missing}"); sys.exit(1)

    total_w = sum(weights.values())
    pareto_points = []
    best_op_point, best_qas = None, -1.0

    for threshold in ENTROPY_THRESHOLDS:
        thresh_data = all_data.get(str(threshold), {})
        agg = {}
        for bm in benchmarks:
            vals = {k: [] for k in ["tps", "acc", "speedup", "ret", "ch"]}
            for seed in SEEDS:
                sd = thresh_data.get(str(seed), {}).get(bm, {})
                if "error" in sd or not sd: continue
                vals["tps"].append(sd.get("avg_tps", 0))
                vals["acc"].append(sd.get(acc_key[bm], 0))
                vals["speedup"].append(sd.get("speedup", 0))
                vals["ret"].append(sd.get("accuracy_retention", 0))
                vals["ch"].append(sd.get("cache_hit_rate", 0))
            agg[bm] = {
                "avg_tps_mean":       float(np.mean(vals["tps"]))  if vals["tps"] else 0,
                "accuracy_mean":      float(np.mean(vals["acc"]))  if vals["acc"] else 0,
                "speedup_mean":       float(np.mean(vals["speedup"])) if vals["speedup"] else 0,
                "acc_retention_mean": float(np.mean(vals["ret"]))  if vals["ret"] else 0,
                "cache_hit_mean":     float(np.mean(vals["ch"]))   if vals["ch"] else 0,
                "n_seeds": len(vals["tps"]),
            }
        combined_speedup = sum(agg[bm]["speedup_mean"] * weights[bm] for bm in benchmarks) / total_w
        combined_acc_ret = sum(agg[bm]["acc_retention_mean"] * weights[bm] for bm in benchmarks) / total_w
        combined_qas = combined_speedup * combined_acc_ret
        gsm_acc_drop = 1.0 - agg["gsm8k"]["acc_retention_mean"]
        point = {
            "entropy_threshold": threshold, "aggregated": agg,
            "combined_speedup": combined_speedup, "combined_accuracy_retention": combined_acc_ret,
            "combined_qas": combined_qas, "within_2pct_accuracy_budget": gsm_acc_drop <= 0.02,
            "gsm8k_acc_drop": gsm_acc_drop,
        }
        pareto_points.append(point)
        print(f"  threshold={threshold}: speedup={combined_speedup:.3f}x, acc_ret={combined_acc_ret:.3f}, QAS={combined_qas:.3f}")
        if gsm_acc_drop <= 0.02 and combined_qas > best_qas:
            best_qas = combined_qas; best_op_point = point
    if best_op_point is None:
        best_op_point = max(pareto_points, key=lambda p: p["combined_qas"])

    now = datetime.now().isoformat()
    out_path = RESULTS_DIR / "m1_pareto_full.json"
    out_path.write_text(json.dumps({
        "task_id": "full_m1_pareto", "model": "LLaDA-8B-Instruct", "method": "M1-EntropyCache",
        "merged_at": now, "seeds": SEEDS, "entropy_thresholds": ENTROPY_THRESHOLDS,
        "benchmarks": benchmarks, "baseline_reference": BASELINE,
        "pareto_points": pareto_points, "operating_point": best_op_point,
        "per_threshold_per_seed": all_data,
    }, indent=2))
    print(f"\nM1 Pareto merged → {out_path}")
    print(f"Best op: threshold={best_op_point['entropy_threshold']}, speedup={best_op_point['combined_speedup']:.3f}x, QAS={best_op_point['combined_qas']:.3f}")

    gp_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        gp = json.loads(gp_path.read_text()) if gp_path.exists() else {"completed":[],"failed":[],"running":{},"timings":{}}
        if "full_m1_pareto" not in gp["completed"]: gp["completed"].append("full_m1_pareto")
        gp["timings"]["full_m1_pareto"] = {"merged_at": now, "note": "split t2+t3 parallel"}
        gp_path.write_text(json.dumps(gp, indent=2))
        print("gpu_progress.json updated")
    except Exception as e: print(f"WARNING: {e}")

if __name__ == "__main__":
    merge_and_aggregate()
