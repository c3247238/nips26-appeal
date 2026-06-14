"""
Aggregate full-scale results across seeds for all tasks.
Computes mean +/- std accuracy, cross-seed McNemar test, and summary tables.
"""
import json, os
from pathlib import Path
import numpy as np

RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results/full")
SEEDS = [42, 123, 456]


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def aggregate_task(task_prefix, benchmark_name, metric_key="accuracy"):
    """Aggregate results across 3 seeds for a given task."""
    seed_data = {}
    for seed in SEEDS:
        fname = RESULTS_DIR / f"{task_prefix}_s{seed}.json"
        data = load_json(fname)
        if data:
            seed_data[seed] = data
        else:
            print(f"  WARNING: Missing {fname}")

    if not seed_data:
        return None

    # Get methods from first available seed
    first = list(seed_data.values())[0]
    methods = first.get("methods", [])

    agg = {"benchmark": benchmark_name, "seeds_found": list(seed_data.keys()),
           "n_samples": first.get("n_samples", 0), "methods": {}}

    for method in methods:
        values = []
        for seed, data in seed_data.items():
            summaries = data.get("summaries", {})
            if method in summaries:
                val = summaries[method].get(metric_key, summaries[method].get("pass_at_1", 0))
                values.append(val)

        if values:
            agg["methods"][method] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "per_seed": {seed: v for seed, v in zip(seed_data.keys(), values)},
                "n_seeds": len(values),
            }

    # Cross-seed McNemar (paired across problems)
    if len(seed_data) >= 2 and "vanilla" in methods:
        mcnemar_agg = {}
        for method in methods:
            if method == "vanilla":
                continue
            # Aggregate per-problem correctness across seeds
            improvements = 0
            regressions = 0
            for seed, data in seed_data.items():
                v_results = data.get("results", {}).get("vanilla", [])
                m_results = data.get("results", {}).get(method, [])
                if not v_results or not m_results:
                    continue
                for vr, mr in zip(v_results, m_results):
                    v_correct = vr.get("is_correct", vr.get("passed", False))
                    m_correct = mr.get("is_correct", mr.get("passed", False))
                    if not v_correct and m_correct:
                        improvements += 1
                    elif v_correct and not m_correct:
                        regressions += 1

            if improvements + regressions > 0:
                chi2 = (abs(improvements - regressions) - 1)**2 / (improvements + regressions)
                from scipy import stats
                p_value = 1 - stats.chi2.cdf(chi2, df=1)
                mcnemar_agg[f"vanilla_vs_{method}"] = {
                    "improvements": improvements,
                    "regressions": regressions,
                    "chi2": float(chi2),
                    "p_value": float(p_value),
                    "significant_p005": p_value < 0.05,
                }
        agg["mcnemar_pooled"] = mcnemar_agg

    return agg


def print_table(agg, metric_name="Accuracy"):
    if not agg:
        return
    print(f"\n  {agg['benchmark']} (N={agg['n_samples']}, seeds={agg['seeds_found']})")
    print(f"  {'Method':<18} {metric_name:>12} {'Std':>8}")
    print(f"  {'-'*40}")
    for method, stats in agg["methods"].items():
        print(f"  {method:<18} {stats['mean']:>11.1%} {stats['std']:>7.1%}")

    if "mcnemar_pooled" in agg:
        print(f"\n  McNemar Tests (pooled across seeds):")
        for test_name, result in agg["mcnemar_pooled"].items():
            sig = "*" if result.get("significant_p005") else ""
            print(f"    {test_name}: p={result['p_value']:.4f}{sig} "
                  f"(+{result['improvements']}/-{result['regressions']})")


def main():
    print("=" * 70)
    print("  Full-Scale Results Aggregation")
    print("=" * 70)

    all_agg = {}

    # Task 5a: Countdown
    agg = aggregate_task("task5a_countdown", "Countdown-500", "accuracy")
    if agg:
        all_agg["countdown"] = agg
        print_table(agg)

    # Task 5b: GSM8K
    agg = aggregate_task("task5b_gsm8k", "GSM8K-1319", "accuracy")
    if agg:
        all_agg["gsm8k"] = agg
        print_table(agg)

    # Task 5c: MBPP
    agg = aggregate_task("task5c_mbpp", "MBPP-Sanitized", "pass_at_1")
    if agg:
        all_agg["mbpp"] = agg
        print_table(agg, "Pass@1")

    # Save aggregated results
    out_file = RESULTS_DIR / "aggregated_results.json"
    with open(out_file, "w") as f:
        json.dump(all_agg, f, indent=2, ensure_ascii=False)
    print(f"\n  Aggregated results saved to {out_file}")

    # Summary markdown
    md_lines = ["# Full-Scale DTA Results\n"]
    for bench_name, agg in all_agg.items():
        md_lines.append(f"\n## {agg['benchmark']}\n")
        md_lines.append(f"| Method | Mean | Std | Per-Seed |")
        md_lines.append(f"|--------|------|-----|----------|")
        for method, stats in agg["methods"].items():
            per_seed_str = ", ".join(f"s{s}={v:.1%}" for s, v in stats["per_seed"].items())
            md_lines.append(f"| {method} | {stats['mean']:.1%} | {stats['std']:.1%} | {per_seed_str} |")

    summary_file = RESULTS_DIR / "summary.md"
    with open(summary_file, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  Summary written to {summary_file}")


if __name__ == "__main__":
    main()
