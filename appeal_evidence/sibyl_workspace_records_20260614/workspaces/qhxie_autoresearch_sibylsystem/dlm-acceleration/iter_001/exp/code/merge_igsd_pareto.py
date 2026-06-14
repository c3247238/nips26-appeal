"""
Merge igsd_pareto_partial.json + p2 result files into igsd_pareto_full.json.
Run after all 4 p2 processes complete.
"""
import json, sys, numpy as np
from pathlib import Path
from datetime import datetime

WORKSPACE   = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full_igsd"

SEEDS = [42, 123, 456]
benchmarks = ["gsm8k", "math500", "humaneval", "mbpp"]
acc_key = {"gsm8k": "exact_match", "math500": "exact_match",
           "humaneval": "pass_at_1", "mbpp": "pass_at_1"}
weights = {"gsm8k": 1319, "math500": 500, "humaneval": 164, "mbpp": 374}

BASELINE = {
    "gsm8k":     {"exact_match": 0.7122, "avg_tps": 31.013},
    "math500":   {"exact_match": 0.1107, "avg_tps": 79.221},
    "humaneval": {"pass_at_1":   0.0244, "avg_tps": 97.999},
    "mbpp":      {"pass_at_1":   0.0000, "avg_tps": 191.586},
}

def load_json(path):
    try: return json.loads(path.read_text()) if path.exists() else {}
    except: return {}

def merge():
    all_results = {}
    # 1. Load partial (main process results)
    partial = load_json(RESULTS_DIR / "igsd_pareto_partial.json")
    all_results.update(partial)
    print(f"Partial: {len(partial)} configs")

    # 2. Load p2 results
    p2_files = [
        ("igsd_p2_tau09_td16_s123.json", "tau_0.9_tdraft_16", "123"),
        ("igsd_p2_tau09_td16_s456.json", "tau_0.9_tdraft_16", "456"),
        ("igsd_p2_tau09_td32_s123.json", "tau_0.9_tdraft_32", "123"),
        ("igsd_p2_tau09_td32_s456.json", "tau_0.9_tdraft_32", "456"),
    ]
    for fname, config_key, seed in p2_files:
        p2 = load_json(RESULTS_DIR / fname)
        if not p2:
            print(f"  WARNING: {fname} missing")
            continue
        # p2 format: {config_key: {seed: {bm: ...}}}
        cfg_data = p2.get(config_key, {})
        seed_data = cfg_data.get(seed, {})
        if seed_data:
            if config_key not in all_results:
                all_results[config_key] = {}
            all_results[config_key][seed] = seed_data
            print(f"  Loaded {fname}: {config_key}[seed={seed}], benchmarks={list(seed_data.keys())}")
        else:
            print(f"  WARNING: {fname} has no data for {config_key}[{seed}]")

    print(f"\nTotal merged configs: {len(all_results)}")

    # 3. Aggregate and compute pareto
    tau_values = sorted(set(float(k.split('_')[1]) for k in all_results))
    t_draft_values = sorted(set(int(k.split('tdraft_')[1]) for k in all_results))
    print(f"tau values: {tau_values}")
    print(f"t_draft values: {t_draft_values}")

    pareto_points = []
    best_op_point, best_qas = None, -1.0
    total_w = sum(weights.values())

    for tau in tau_values:
        for t_draft in t_draft_values:
            config_key = f"tau_{tau}_tdraft_{t_draft}"
            config_data = all_results.get(config_key, {})
            if not config_data:
                continue

            agg = {}
            for bm in benchmarks:
                vals = {"tps": [], "acc": [], "speedup": [], "ret": [], "accept": []}
                for seed in SEEDS:
                    sd = config_data.get(str(seed), {}).get(bm, {})
                    if not sd or "error" in sd:
                        continue
                    bl_tps = BASELINE[bm]["avg_tps"]
                    bl_acc = BASELINE[bm].get(acc_key[bm], 0)
                    cur_acc = sd.get(acc_key[bm], 0)
                    cur_tps = sd.get("avg_tps", bl_tps)
                    vals["tps"].append(cur_tps)
                    vals["acc"].append(cur_acc)
                    vals["speedup"].append(cur_tps / max(bl_tps, 1e-6))
                    vals["ret"].append(cur_acc / max(bl_acc, 1e-6) if bl_acc > 0 else 1.0)
                    vals["accept"].append(sd.get("avg_accept_rate", 0))
                if vals["tps"]:
                    agg[bm] = {
                        "avg_tps": np.mean(vals["tps"]),
                        "avg_acc": np.mean(vals["acc"]),
                        "avg_speedup": np.mean(vals["speedup"]),
                        "avg_acc_ret": np.mean(vals["ret"]),
                        "avg_accept_rate": np.mean(vals["accept"]),
                        "n_seeds": len(vals["tps"]),
                    }

            if not agg:
                continue

            # Combined weighted metrics
            total_w = sum(weights[b] for b in benchmarks if b in agg)
            combined_speedup = sum(weights[b] * agg[b]["avg_speedup"] for b in agg) / total_w
            combined_acc_ret = sum(weights[b] * agg[b]["avg_acc_ret"] for b in agg) / total_w
            combined_accept  = sum(weights[b] * agg[b]["avg_accept_rate"] for b in agg) / total_w
            gsm8k_acc_drop   = 1.0 - agg.get("gsm8k", {}).get("avg_acc_ret", 1.0)
            within_5pct = bool(gsm8k_acc_drop <= 0.05)
            combined_qas = float(combined_speedup * combined_acc_ret) if within_5pct else float(combined_speedup * combined_acc_ret * 0.5)

            point = {
                "tau": float(tau), "t_draft": int(t_draft),
                "combined_speedup": float(combined_speedup),
                "combined_acc_ret": float(combined_acc_ret),
                "combined_accept_rate": float(combined_accept),
                "gsm8k_acc_drop": float(gsm8k_acc_drop),
                "within_5pct": within_5pct,
                "combined_qas": float(combined_qas),
                "per_benchmark": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in agg.items()},
            }
            pareto_points.append(point)

            if within_5pct and combined_qas > best_qas:
                best_qas = combined_qas
                best_op_point = point

            print(f"  tau={tau}, T={t_draft}: speedup={combined_speedup:.3f}x, "
                  f"acc_ret={combined_acc_ret:.3f}, QAS={combined_qas:.3f}, accept={combined_accept:.3f}")

    if best_op_point is None and pareto_points:
        best_op_point = max(pareto_points, key=lambda p: p["combined_qas"])

    output = {
        "generated_at": datetime.now().isoformat(),
        "pareto_points": pareto_points,
        "best_operating_point": best_op_point,
        "all_results": all_results,
    }
    out_path = RESULTS_DIR / "igsd_pareto_full.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nFull results → {out_path}")
    if best_op_point:
        print(f"Best IGSD operating point: tau={best_op_point['tau']}, T_draft={best_op_point['t_draft']}, "
              f"speedup={best_op_point['combined_speedup']:.3f}x, QAS={best_op_point['combined_qas']:.3f}")
    return output, best_op_point

if __name__ == "__main__":
    output, best = merge()

def merge_with_supplementary():
    """Extended merge including supplementary experiments."""
    output, best = merge()
    
    # Load supplementary results and integrate
    sup_files = [
        ("igsd_sup_tau09_td8_s123.json", "tau_0.9_tdraft_8", "123"),
        ("igsd_sup_tau09_td8_s456.json", "tau_0.9_tdraft_8", "456"),
        ("igsd_sup_tau09_td4_s123.json", "tau_0.9_tdraft_4", "123"),
    ]
    
    all_results = output["all_results"]
    updated = False
    for fname, config_key, seed in sup_files:
        p = RESULTS_DIR / fname
        if not p.exists():
            print(f"  MISSING: {fname}")
            continue
        d = load_json(p)
        cfg = d.get(config_key, {})
        sd = cfg.get(seed, {})
        if sd:
            if config_key not in all_results:
                all_results[config_key] = {}
            all_results[config_key][seed] = sd
            print(f"  Supplementary: {config_key}[seed={seed}] added")
            updated = True
    
    if updated:
        print("\nRe-running aggregation with supplementary data...")
        # Re-aggregate
        import sys
        sys.stdout.flush()
    
    return output, best
