"""
Trade-off Analysis: Does reducing absorption improve downstream performance?

Uses existing experimental data from iterations 1-2.
Analyzes absorption rate vs steering/probing performance.
"""

import json
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, kruskal


def load_data():
    """Load existing experimental results."""
    with open("exp/results/full/correlation_report_full.json") as f:
        report = json.load(f)

# Layer 8 detailed data (optional)
    layer8_abs = {}

# Layer 4 not available separately
    layer4_abs = {}

    return report, layer8_abs, layer4_abs


def categorize_by_absorption(absorption_rates, thresholds=(0.05, 0.10)):
    """Categorize features by absorption level."""
    low = {k: v for k, v in absorption_rates.items() if v <= thresholds[0]}
    med = {k: v for k, v in absorption_rates.items() if thresholds[0] < v <= thresholds[1]}
    high = {k: v for k, v in absorption_rates.items() if v > thresholds[1]}
    return low, med, high


def compare_groups(performance_by_letter, group_low, group_med, group_high, metric_name):
    """Compare performance across absorption groups."""
    low_perf = [performance_by_letter.get(k, 0) for k in group_low]
    med_perf = [performance_by_letter.get(k, 0) for k in group_med]
    high_perf = [performance_by_letter.get(k, 0) for k in group_high]

    # Remove zeros (missing data)
    low_perf = [p for p in low_perf if p > 0]
    med_perf = [p for p in med_perf if p > 0]
    high_perf = [p for p in high_perf if p > 0]

    if not low_perf or not high_perf:
        return None

    # Mann-Whitney U test (non-parametric)
    try:
        stat, pval = mannwhitneyu(low_perf, high_perf, alternative="greater")
    except:
        stat, pval = 0, 1.0

    # Kruskal-Wallis for all three groups
    try:
        kw_stat, kw_pval = kruskal(low_perf, med_perf, high_perf)
    except:
        kw_stat, kw_pval = 0, 1.0

    return {
        "metric": metric_name,
        "low_mean": np.mean(low_perf) if low_perf else 0,
        "low_std": np.std(low_perf) if low_perf else 0,
        "low_n": len(low_perf),
        "med_mean": np.mean(med_perf) if med_perf else 0,
        "med_std": np.std(med_perf) if med_perf else 0,
        "med_n": len(med_perf),
        "high_mean": np.mean(high_perf) if high_perf else 0,
        "high_std": np.std(high_perf) if high_perf else 0,
        "high_n": len(high_perf),
        "mann_whitney_stat": float(stat),
        "mann_whitney_p": float(pval),
        "kruskal_stat": float(kw_stat),
        "kruskal_p": float(kw_pval),
    }


def correlate_absorption_performance(absorption_rates, performance):
    """Spearman correlation between absorption and performance."""
    common = set(absorption_rates.keys()) & set(performance.keys())
    abs_vals = [absorption_rates[k] for k in common]
    perf_vals = [performance[k] for k in common]

    if len(abs_vals) < 3:
        return {"rho": 0, "p_value": 1.0, "n": len(abs_vals)}

    rho, pval = spearmanr(abs_vals, perf_vals)
    return {"rho": float(rho), "p_value": float(pval), "n": len(abs_vals)}


def main():
    print("=" * 60)
    print("Trade-off Analysis: Absorption vs Downstream Performance")
    print("=" * 60)

    report, layer8_abs, layer4_abs = load_data()

    results = {}

    for layer_name, layer_key in [("Layer 4", "layer_4"), ("Layer 8", "layer_8")]:
        print(f"\n{'='*50}")
        print(layer_name)
        print(f"{'='*50}")

        lr = report["layer_results"][layer_key]
        ar = lr["absorption_rates"]

        # Get performance metrics
        steering = lr.get("steering_success_at_50", {})
        probing = lr.get("probing_f1_k5", {})

        # Categorize
        low, med, high = categorize_by_absorption(ar)
        print(f"LOW absorption: {len(low)} features")
        print(f"MED absorption: {len(med)} features")
        print(f"HIGH absorption: {len(high)} features")

        # Compare steering
        if steering:
            steer_comp = compare_groups(steering, low, med, high, "steering_success")
            if steer_comp:
                print(f"\n  Steering Success:")
                print(f"    LOW:  {steer_comp['low_mean']:.3f} (n={steer_comp['low_n']})")
                print(f"    MED:  {steer_comp['med_mean']:.3f} (n={steer_comp['med_n']})")
                print(f"    HIGH: {steer_comp['high_mean']:.3f} (n={steer_comp['high_n']})")
                print(f"    Mann-Whitney p: {steer_comp['mann_whitney_p']:.4f}")
                print(f"    Kruskal-Wallis p: {steer_comp['kruskal_p']:.4f}")

                steer_corr = correlate_absorption_performance(ar, steering)
                print(f"    Spearman rho: {steer_corr['rho']:.3f}, p={steer_corr['p_value']:.4f}")

                results[f"{layer_key}_steering"] = {"comparison": steer_comp, "correlation": steer_corr}

        # Compare probing
        if probing:
            probe_comp = compare_groups(probing, low, med, high, "probing_f1")
            if probe_comp:
                print(f"\n  Probing F1 (k=5):")
                print(f"    LOW:  {probe_comp['low_mean']:.3f} (n={probe_comp['low_n']})")
                print(f"    MED:  {probe_comp['med_mean']:.3f} (n={probe_comp['med_n']})")
                print(f"    HIGH: {probe_comp['high_mean']:.3f} (n={probe_comp['high_n']})")
                print(f"    Mann-Whitney p: {probe_comp['mann_whitney_p']:.4f}")
                print(f"    Kruskal-Wallis p: {probe_comp['kruskal_p']:.4f}")

                probe_corr = correlate_absorption_performance(ar, probing)
                print(f"    Spearman rho: {probe_corr['rho']:.3f}, p={probe_corr['p_value']:.4f}")

                results[f"{layer_key}_probing"] = {"comparison": probe_comp, "correlation": probe_corr}

    # Cross-layer comparison
    print(f"\n{'='*60}")
    print("Cross-Layer Summary")
    print(f"{'='*60}")

    for metric in ["steering", "probing"]:
        l4_key = f"layer_4_{metric}"
        l8_key = f"layer_8_{metric}"

        if l4_key in results and l8_key in results:
            r4 = results[l4_key]["correlation"]
            r8 = results[l8_key]["correlation"]

            print(f"\n{metric.upper()}:")
            print(f"  Layer 4: rho={r4['rho']:.3f}, p={r4['p_value']:.4f}")
            print(f"  Layer 8: rho={r8['rho']:.3f}, p={r8['p_value']:.4f}")

            if r4["rho"] < 0 and r4["p_value"] < 0.05:
                print(f"  -> Layer 4: Lower absorption = Better {metric} (SIGNIFICANT)")
            elif r4["rho"] < 0:
                print(f"  -> Layer 4: Lower absorption = Better {metric} (trend)")

            if r8["rho"] < 0 and r8["p_value"] < 0.05:
                print(f"  -> Layer 8: Lower absorption = Better {metric} (SIGNIFICANT)")
            elif r8["rho"] < 0:
                print(f"  -> Layer 8: Lower absorption = Better {metric} (trend)")

    # Overall verdict
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    significant_findings = []
    for key, val in results.items():
        corr = val["correlation"]
        if corr["rho"] < 0 and corr["p_value"] < 0.05:
            significant_findings.append(key)

    if significant_findings:
        print(f"SIGNIFICANT trade-off found in: {', '.join(significant_findings)}")
        print("Conclusion: REDUCING absorption DOES improve downstream performance")
    else:
        trend_findings = [k for k, v in results.items() if v["correlation"]["rho"] < 0]
        if trend_findings:
            print(f"Trend (not significant) found in: {', '.join(trend_findings)}")
            print("Conclusion: Weak evidence that reducing absorption improves performance")
        else:
            print("No evidence of trade-off. Null result.")

    # Save
    output = {
        "experiment": "tradeoff_analysis",
        "hypothesis": "Lower absorption rates correlate with better downstream performance",
        "results": results,
        "significant_findings": significant_findings,
    }

    with open("exp/results/full/tradeoff_analysis.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: exp/results/full/tradeoff_analysis.json")
    return output


if __name__ == "__main__":
    main()
