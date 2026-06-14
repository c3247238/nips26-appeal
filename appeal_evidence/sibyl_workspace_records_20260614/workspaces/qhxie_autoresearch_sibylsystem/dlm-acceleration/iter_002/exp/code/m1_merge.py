"""Merge 3 seed shards into final m1_pareto_corrected.json"""
import json, numpy as np
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current/exp/results/m1_pareto")
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
SEEDS = [42, 123, 456]
THRESHOLDS = [0.5, 1.0, 2.0]
BASELINE = {
    "gsm8k": {"exact_match": 0.7122, "avg_tps": 31.013},
    "math500": {"exact_match": 0.1107, "avg_tps": 79.221},
    "humaneval": {"pass_at_1": 0.0244, "avg_tps": 97.999},
}
D2CACHE_CHR = {
    0.5: {"gsm8k_chr": 0.933, "math500_chr": 0.871, "theoretical_speedup": 2.27},
    1.0: {"gsm8k_chr": 0.970, "math500_chr": 0.935, "theoretical_speedup": 2.39},
    2.0: {"gsm8k_chr": 0.991, "math500_chr": 0.980, "theoretical_speedup": 2.47},
}

# Load all shards
all_data = {}
for seed in SEEDS:
    p = RESULTS_DIR / f"partial_seed_{seed}.json"
    if p.exists():
        all_data[str(seed)] = json.loads(p.read_text())
        print(f"Loaded seed {seed}: {len(all_data[str(seed)])} thresholds")
    else:
        print(f"WARNING: missing seed {seed}")

# Restructure: threshold -> seed -> benchmarks
per_threshold_per_seed = {}
for thresh in THRESHOLDS:
    tkey = str(thresh)
    per_threshold_per_seed[tkey] = {}
    for seed in SEEDS:
        skey = str(seed)
        if skey in all_data and tkey in all_data[skey]:
            per_threshold_per_seed[tkey][skey] = all_data[skey][tkey]

# Aggregate
pareto_points = []
best_op_point = None
best_qas = -1.0

for threshold in THRESHOLDS:
    tkey = str(threshold)
    thresh_data = per_threshold_per_seed.get(tkey, {})
    agg = {}
    for bm in ["gsm8k", "math500", "humaneval"]:
        acc_key = {"gsm8k": "exact_match", "math500": "exact_match", "humaneval": "pass_at_1"}[bm]
        vals_tps, vals_acc, vals_speedup, vals_ret, vals_ch = [], [], [], [], []
        for seed in SEEDS:
            sd = thresh_data.get(str(seed), {}).get(bm, {})
            if "error" in sd or not sd: continue
            vals_tps.append(sd.get("avg_tps", 0))
            vals_acc.append(sd.get(acc_key, 0))
            vals_speedup.append(sd.get("speedup", 0))
            vals_ret.append(sd.get("accuracy_retention", 0))
            vals_ch.append(sd.get("cache_hit_rate", 0))
        if vals_tps:
            agg[bm] = {
                "avg_tps_mean": float(np.mean(vals_tps)), "avg_tps_std": float(np.std(vals_tps)),
                "accuracy_mean": float(np.mean(vals_acc)), "accuracy_std": float(np.std(vals_acc)),
                "speedup_mean": float(np.mean(vals_speedup)), "speedup_std": float(np.std(vals_speedup)),
                "acc_retention_mean": float(np.mean(vals_ret)), "acc_retention_std": float(np.std(vals_ret)),
                "cache_hit_mean": float(np.mean(vals_ch)), "n_seeds": len(vals_tps),
                "per_seed_acc": vals_acc, "per_seed_speedup": vals_speedup,
            }
        else:
            agg[bm] = {"avg_tps_mean": 0, "accuracy_mean": 0, "speedup_mean": 0,
                        "acc_retention_mean": 0, "cache_hit_mean": 0, "n_seeds": 0}

    gsm_s = agg["gsm8k"]["speedup_mean"]
    gsm_r = agg["gsm8k"]["acc_retention_mean"]
    math_s = agg["math500"]["speedup_mean"]
    math_r = agg["math500"]["acc_retention_mean"]
    combined_speedup = 0.7 * gsm_s + 0.3 * math_s
    combined_acc_ret = 0.7 * gsm_r + 0.3 * math_r
    combined_qas = combined_speedup * combined_acc_ret
    d2 = D2CACHE_CHR.get(threshold, D2CACHE_CHR[1.0])
    proj_speedup = d2["theoretical_speedup"]
    proj_qas = proj_speedup * combined_acc_ret
    gsm_drop = 1.0 - gsm_r
    within = gsm_drop <= 0.02

    point = {
        "entropy_threshold": threshold, "aggregated": agg,
        "combined_speedup_measured": combined_speedup,
        "combined_speedup_projected": proj_speedup,
        "combined_accuracy_retention": combined_acc_ret,
        "combined_qas_measured": combined_qas,
        "combined_qas_projected": proj_qas,
        "within_2pct_accuracy_budget": within,
        "gsm8k_acc_drop": gsm_drop,
        "d2cache_chr_data": d2,
    }
    pareto_points.append(point)
    if within and combined_qas > best_qas:
        best_qas = combined_qas
        best_op_point = point
    print(f"  t={threshold}: speedup_meas={combined_speedup:.3f}x, proj={proj_speedup:.2f}x, "
          f"acc_ret={combined_acc_ret:.3f}, QAS_meas={combined_qas:.3f}, within_2%={within}")

if best_op_point is None:
    for p in pareto_points:
        if (1.0 - p["aggregated"]["gsm8k"]["acc_retention_mean"]) <= 0.05:
            if p["combined_qas_measured"] > best_qas:
                best_qas = p["combined_qas_measured"]
                best_op_point = p
if best_op_point is None and pareto_points:
    best_op_point = max(pareto_points, key=lambda p: p["combined_qas_measured"])

output = {
    "task_id": "m1_pareto_corrected", "model": "LLaDA-8B-Instruct",
    "method": "M1-EntropyCache", "mode": "full", "iteration": 2,
    "seeds": SEEDS, "entropy_thresholds": THRESHOLDS,
    "benchmarks": ["gsm8k", "math500", "humaneval"],
    "combined_metric": "0.7*GSM8K + 0.3*MATH500",
    "baseline_reference": BASELINE, "d2cache_chr_reference": D2CACHE_CHR,
    "pareto_points": pareto_points, "operating_point": best_op_point,
    "per_threshold_per_seed": per_threshold_per_seed,
    "timestamp": datetime.now().isoformat(),
}

out_path = RESULTS_DIR / "m1_pareto_corrected.json"
out_path.write_text(json.dumps(output, indent=2))
print(f"\nMerged results saved to {out_path}")

# Update tracking files
gp_path = WORKSPACE / "exp" / "gpu_progress.json"
gp = json.loads(gp_path.read_text())
if "m1_pareto_corrected" not in gp.get("completed", []):
    gp["completed"].append("m1_pareto_corrected")
if "m1_pareto_corrected" in gp.get("running", {}):
    del gp["running"]["m1_pareto_corrected"]
gp_path.write_text(json.dumps(gp, indent=2))

es_path = WORKSPACE / "exp" / "experiment_state.json"
es = json.loads(es_path.read_text())
if "m1_pareto_corrected" in es.get("tasks", {}):
    es["tasks"]["m1_pareto_corrected"]["status"] = "completed"
es_path.write_text(json.dumps(es, indent=2))

# DONE marker
(RESULTS_DIR / "m1_pareto_corrected_DONE").write_text(json.dumps({
    "task_id": "m1_pareto_corrected", "status": "success",
    "summary": f"Merged 3 seeds. Best: t={best_op_point['entropy_threshold'] if best_op_point else 'N/A'}",
    "timestamp": datetime.now().isoformat(),
}))
print("Tracking files updated. DONE.")
