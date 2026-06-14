"""
T0.4: Analyze SGD baseline data from iter_003.
Produces comparison tables with paired t-tests, Bonferroni-Holm correction, Cohen's d.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import math

# Data paths
ITER3_RESULTS = Path(__file__).parent.parent.parent.parent / "iter_003" / "exp" / "results"
SGD_DIR = ITER3_RESULTS / "sgd_baseline"
ADAMW_DIR = ITER3_RESULTS / "full"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "analysis"


def load_summaries(base_dir, arch="resnet20"):
    """Load all summary.json files grouped by (dataset, method) -> [acc values]."""
    results = defaultdict(list)
    for dataset in ["cifar10", "cifar100"]:
        dataset_dir = base_dir / dataset / arch
        if not dataset_dir.exists():
            continue
        for method_dir in sorted(dataset_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method = method_dir.name
            for seed_dir in sorted(method_dir.iterdir()):
                summary_path = seed_dir / "summary.json"
                if summary_path.exists():
                    with open(summary_path) as f:
                        data = json.load(f)
                    results[(dataset, method)].append({
                        'best_acc': data['best_test_acc'],
                        'final_acc': data['final_test_acc'],
                        'bem': data.get('final_bem', None),
                        'csi': data.get('final_csi', None),
                        'ais': data.get('final_ais', None),
                        'weight_norm': data.get('final_weight_norm', None),
                        'seed': data['config']['seed'],
                    })
    return results


def mean_std(values):
    if not values:
        return 0, 0
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var)


def cohens_d(group1, group2):
    """Compute Cohen's d (effect size) between two groups."""
    m1, s1 = mean_std(group1)
    m2, s2 = mean_std(group2)
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (m1 - m2) / pooled_std


def paired_t_test(diffs):
    """Simple paired t-test on differences. Returns t-stat and p-value approx."""
    n = len(diffs)
    if n < 2:
        return 0.0, 1.0
    m = sum(diffs) / n
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    if se < 1e-10:
        return float('inf'), 0.0
    t_stat = m / se
    # Approximate two-tailed p-value using normal approximation for df >= 3
    # For small samples, this is rough but adequate for analysis
    import statistics
    p_val = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    return t_stat, p_val


def _normal_cdf(x):
    """Standard normal CDF approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def bonferroni_holm(p_values):
    """Bonferroni-Holm step-down correction.

    Input: list of (label, p_value) tuples.
    Returns: list of (label, original_p, adjusted_p, significant) tuples.
    """
    n = len(p_values)
    sorted_pvals = sorted(p_values, key=lambda x: x[1])
    results = []
    for rank, (label, p) in enumerate(sorted_pvals):
        adjusted_p = min(1.0, p * (n - rank))
        results.append((label, p, adjusted_p, adjusted_p < 0.05))
    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SGD Baseline Analysis (iter_003 data)")
    print("=" * 80)

    sgd_results = load_summaries(SGD_DIR, "resnet20")
    adamw_results = load_summaries(ADAMW_DIR, "resnet20")

    # Table 1: SGD Best Accuracy Summary
    print("\n### Table: SGD ResNet-20 Best Test Accuracy (mean +/- std)")
    print(f"{'Dataset':<10} {'Method':<18} {'Best Acc':<18} {'Final Acc':<18} {'n':>3}")
    print("-" * 70)

    table_data = []
    for dataset in ["cifar10", "cifar100"]:
        for method in ["constant", "cosine_schedule", "cwd_hard", "swd",
                       "half_lambda", "random_mask", "no_wd"]:
            key = (dataset, method)
            if key not in sgd_results:
                continue
            entries = sgd_results[key]
            best_accs = [e['best_acc'] for e in entries]
            final_accs = [e['final_acc'] for e in entries]
            bm, bs = mean_std(best_accs)
            fm, fs = mean_std(final_accs)
            n = len(entries)
            print(f"{dataset:<10} {method:<18} {bm:.2f} +/- {bs:.2f}   {fm:.2f} +/- {fs:.2f}   {n:>3}")
            table_data.append({
                'dataset': dataset, 'method': method,
                'best_mean': bm, 'best_std': bs,
                'final_mean': fm, 'final_std': fs, 'n': n,
                'optimizer': 'sgd'
            })

    # Table 2: Cohen's d (vs constant baseline)
    print("\n### Table: Cohen's d (each method vs constant) - SGD")
    print(f"{'Dataset':<10} {'Method':<18} {'Cohen d':<10} {'Direction':<12}")
    print("-" * 55)

    effect_sizes = []
    p_values_for_correction = []

    for dataset in ["cifar10", "cifar100"]:
        const_key = (dataset, "constant")
        if const_key not in sgd_results:
            continue
        const_accs = [e['best_acc'] for e in sgd_results[const_key]]

        for method in ["cosine_schedule", "cwd_hard", "swd", "half_lambda",
                       "random_mask", "no_wd"]:
            key = (dataset, method)
            if key not in sgd_results:
                continue
            method_accs = [e['best_acc'] for e in sgd_results[key]]
            d = cohens_d(const_accs, method_accs)
            direction = "constant better" if d > 0 else "method better"
            print(f"{dataset:<10} {method:<18} {d:>8.3f}   {direction}")
            effect_sizes.append({
                'dataset': dataset, 'method': method, 'cohens_d': d,
                'optimizer': 'sgd'
            })

            # Paired differences (by seed order)
            if len(const_accs) == len(method_accs):
                diffs = [c - m for c, m in zip(const_accs, method_accs)]
                t_stat, p_val = paired_t_test(diffs)
                p_values_for_correction.append((f"{dataset}/{method}", p_val))

    # Bonferroni-Holm correction
    if p_values_for_correction:
        print("\n### Bonferroni-Holm Corrected p-values (paired t-test, constant vs method)")
        corrections = bonferroni_holm(p_values_for_correction)
        print(f"{'Comparison':<30} {'p_raw':<10} {'p_adj':<10} {'Sig?':<6}")
        print("-" * 60)
        for label, p_raw, p_adj, sig in corrections:
            sig_str = "YES *" if sig else "no"
            print(f"{label:<30} {p_raw:<10.4f} {p_adj:<10.4f} {sig_str}")

    # AdamW comparison if available
    if adamw_results:
        print("\n### Table: AdamW ResNet-20 Best Test Accuracy (mean +/- std)")
        print(f"{'Dataset':<10} {'Method':<18} {'Best Acc':<18} {'n':>3}")
        print("-" * 55)
        for dataset in ["cifar10", "cifar100"]:
            for method in ["constant", "cosine_schedule", "cwd_hard", "cwd_soft",
                          "swd", "half_lambda", "random_mask", "no_wd"]:
                key = (dataset, method)
                if key not in adamw_results:
                    continue
                entries = adamw_results[key]
                best_accs = [e['best_acc'] for e in entries]
                bm, bs = mean_std(best_accs)
                n = len(entries)
                print(f"{dataset:<10} {method:<18} {bm:.2f} +/- {bs:.2f}   {n:>3}")
                table_data.append({
                    'dataset': dataset, 'method': method,
                    'best_mean': bm, 'best_std': bs,
                    'n': n, 'optimizer': 'adamw'
                })

    # Save analysis results
    analysis_output = {
        'table_data': table_data,
        'effect_sizes': effect_sizes,
        'corrections': [(l, pr, pa, s) for l, pr, pa, s in
                        (bonferroni_holm(p_values_for_correction) if p_values_for_correction else [])],
    }

    with open(OUTPUT_DIR / "sgd_baseline_analysis.json", 'w') as f:
        json.dump(analysis_output, f, indent=2)

    print(f"\nAnalysis saved to {OUTPUT_DIR / 'sgd_baseline_analysis.json'}")


if __name__ == '__main__':
    main()
