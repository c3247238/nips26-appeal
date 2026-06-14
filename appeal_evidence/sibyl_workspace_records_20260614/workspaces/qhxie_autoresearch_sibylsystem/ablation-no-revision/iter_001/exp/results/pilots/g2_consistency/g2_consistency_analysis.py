#!/usr/bin/env python3
"""
G2 Consistency Experiment: Activation Energy from Answer Consistency

Hypothesis H2: Activation energy (Ea) estimated from answer consistency
                correlates with problem difficulty (MATH level)

Hypothesis H5: Consistency-derived Ea correlates with actual saturation k₀

This experiment analyzes the G1 saturation data to:
1. Compute consistency trajectories (how often answers match across samples)
2. Estimate Ea from consistency convergence rate
3. Validate against MATH difficulty levels
4. Compare with actual k₀ from saturation fit
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from collections import Counter
import sys

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current")
SEED = 42
SAMPLES = 100
K_VALUES = [1, 2, 4, 8, 16]

np.random.seed(SEED)

def load_g1_results():
    """Load G1 saturation results."""
    results_path = WORKSPACE / "exp/results/pilots/g1_saturation_v2/g1_saturation_v2_results.json"
    with open(results_path) as f:
        return json.load(f)

def extract_answers(k_results):
    """Extract all answers from k_results for a problem."""
    answers = []
    for k in K_VALUES:
        k_str = str(k)
        if k_str in k_results:
            answers.extend(k_results[k_str].get("answers", []))
    return answers

def compute_consistency_at_k(k_results, k):
    """Compute consistency at sample count k.

    Consistency = probability that any two random samples give same answer.
    We approximate by checking what fraction of samples match the most common answer.
    """
    k_str = str(k)
    if k_str not in k_results:
        return 0.0

    answers = k_results[k_str].get("answers", [])
    if not answers:
        return 0.0

    # Count answer frequencies
    answer_counts = Counter(answers)
    most_common_count = answer_counts.most_common(1)[0][1]

    # Consistency = fraction matching most common answer
    return most_common_count / len(answers)

def compute_consistency_trajectory(k_results):
    """Compute consistency at each k value."""
    trajectory = {}
    for k in K_VALUES:
        consistency = compute_consistency_at_k(k_results, k)
        trajectory[k] = consistency
    return trajectory

def estimate_ea_from_consistency(trajectory):
    """Estimate activation energy from consistency convergence.

    Theory: Consistency follows: C_k = C_∞ * (1 - exp(-k * c₀))
    where C_∞ is asymptotic consistency and c₀ relates to Ea.

    Ea is inversely related to c₀ (easier problems converge faster).
    """
    ks = np.array(list(trajectory.keys()))
    cs = np.array([trajectory[k] for k in ks])

    # Handle edge cases
    if cs.min() < 0.1:  # Very low consistency
        return 0.0, 0.0

    try:
        # Fit saturation model: C_k = C_∞ * (1 - exp(-k/c₀))
        def saturation_model(k, c_inf, c0):
            return c_inf * (1 - np.exp(-k / c0))

        popt, _ = curve_fit(
            saturation_model,
            ks, cs,
            p0=[1.0, 2.0],
            bounds=([0.5, 0.1], [1.0, 10.0]),
            maxfev=1000
        )
        c_inf, c0 = popt

        # Ea proxy: inversely proportional to c0
        # Higher c0 means slower convergence = higher Ea
        ea_proxy = 1.0 / c0

        return ea_proxy, c0
    except:
        return 0.0, 0.0

def extract_level(level_str):
    """Extract numeric level from 'Level N' string."""
    if not level_str:
        return 3  # default
    try:
        return int(level_str.replace("Level ", ""))
    except:
        return 3

def analyze_g2_consistency():
    """Main analysis for G2 consistency experiment."""
    print("=" * 60)
    print("G2 Consistency Experiment: Ea from Answer Consistency")
    print("=" * 60)

    # Load G1 results
    g1_results = load_g1_results()
    problems = g1_results.get("problems", [])

    print(f"\nLoaded {len(problems)} problems from G1 saturation data")

    # Analysis data
    problem_analyses = []
    ea_estimates = []
    actual_k0s = []
    levels = []

    for problem in problems[:SAMPLES]:
        idx = problem.get("idx", -1)
        level = extract_level(problem.get("level", "Level 3"))
        k_results = problem.get("k_results", {})

        # Compute consistency trajectory
        trajectory = compute_consistency_trajectory(k_results)

        # Estimate Ea from consistency
        ea_estimate, c0 = estimate_ea_from_consistency(trajectory)

        # Get actual k0 from G1 (if available)
        saturation_fit = problem.get("saturation_fit")
        actual_k0 = saturation_fit.get("k0", 0.0) if saturation_fit else 0.0

        # Compute per-k consistency for output
        per_k_consistency = {}
        for k in K_VALUES:
            k_str = str(k)
            if k_str in k_results:
                answers = k_results[k_str].get("answers", [])
                if answers:
                    answer_counts = Counter(answers)
                    most_common = answer_counts.most_common(1)[0][1]
                    per_k_consistency[f"k={k}"] = {
                        "consistency": most_common / len(answers),
                        "unique_answers": len(answer_counts),
                        "total_samples": len(answers)
                    }

        problem_analyses.append({
            "idx": idx,
            "level": level,
            "level_str": problem.get("level", "Level 3"),
            "problem_type": problem.get("type", "Unknown"),
            "ea_estimate": ea_estimate,
            "c0_estimate": c0,
            "actual_k0": actual_k0,
            "consistency_trajectory": trajectory,
            "per_k_consistency": per_k_consistency
        })

        if ea_estimate > 0:
            ea_estimates.append(ea_estimate)
            levels.append(level)
            if actual_k0 > 0:
                actual_k0s.append(actual_k0)

    # H2: Ea-level correlation
    print("\n" + "-" * 40)
    print("H2: Ea-Level Correlation Analysis")
    print("-" * 40)

    # Group Ea by level
    level_eas = {}
    for pa in problem_analyses:
        level = pa["level"]
        if level not in level_eas:
            level_eas[level] = []
        level_eas[level].append(pa["ea_estimate"])

    level_stats = {}
    for level in sorted(level_eas.keys()):
        eas = [e for e in level_eas[level] if e > 0]
        if eas:
            level_stats[level] = {
                "mean_ea": np.mean(eas),
                "std_ea": np.std(eas),
                "count": len(eas)
            }
            print(f"Level {level}: mean Ea = {np.mean(eas):.3f} (n={len(eas)})")

    # Spearman correlation between Ea and level
    valid_pairs = [(e, l) for e, l in zip(ea_estimates, levels) if e > 0]
    if len(valid_pairs) > 10:
        eas_array = np.array([p[0] for p in valid_pairs])
        levels_array = np.array([p[1] for p in valid_pairs])
        spearman_ea_level, p_ea_level = spearmanr(eas_array, levels_array)
        print(f"\nSpearman(Ea, Level): {spearman_ea_level:.3f} (p={p_ea_level:.4f})")
    else:
        spearman_ea_level = 0.0
        p_ea_level = 1.0

    # H5: Consistency-Ea vs Actual k0 correlation
    print("\n" + "-" * 40)
    print("H5: Consistency-Ea vs Actual k₀ Correlation")
    print("-" * 40)

    valid_k0_pairs = [(e, k) for e, k in zip(ea_estimates, actual_k0s)
                      if e > 0 and k > 0 and k < 10000]  # Filter outliers
    if len(valid_k0_pairs) > 10:
        eas_array = np.array([p[0] for p in valid_k0_pairs])
        k0s_array = np.array([p[1] for p in valid_k0_pairs])
        spearman_ea_k0, p_ea_k0 = spearmanr(eas_array, k0s_array)
        print(f"Spearman(Ea_estimate, k₀_actual): {spearman_ea_k0:.3f} (p={p_ea_k0:.4f})")
        print(f"Valid pairs: {len(valid_k0_pairs)}")
    else:
        spearman_ea_k0 = 0.0
        p_ea_k0 = 1.0
        print(f"Valid pairs: {len(valid_k0_pairs)} (need >10)")

    # Pass criteria
    h2_threshold = 0.4
    h5_threshold = 0.5

    h2_pass = spearman_ea_level > h2_threshold
    h5_pass = spearman_ea_k0 > h5_threshold

    print("\n" + "=" * 60)
    print("HYPOTHESIS EVALUATION")
    print("=" * 60)
    print(f"H2 (Ea-Level Spearman > {h2_threshold}): {'PASS' if h2_pass else 'FAIL'}")
    print(f"  Actual: {spearman_ea_level:.3f}")
    print(f"H5 (Ea-k₀ Spearman > {h5_threshold}): {'PASS' if h5_pass else 'FAIL'}")
    print(f"  Actual: {spearman_ea_k0:.3f}")

    # Overall recommendation
    if h2_pass and h5_pass:
        recommendation = "GO"
        overall_status = "CONFIRMED"
    elif h2_pass or h5_pass:
        recommendation = "PARTIAL"
        overall_status = "PARTIAL"
    else:
        recommendation = "NO_GO"
        overall_status = "NOT_CONFIRMED"

    # Prepare output
    results = {
        "task_id": "g2_consistency",
        "overall_status": overall_status,
        "recommendation": recommendation,
        "h2_evaluation": {
            "hypothesis": "H2: Ea from consistency correlates with MATH level",
            "spearman": spearman_ea_level,
            "p_value": float(p_ea_level),
            "threshold": h2_threshold,
            "pass": h2_pass,
            "level_stats": level_stats
        },
        "h5_evaluation": {
            "hypothesis": "H5: Consistency-derived Ea correlates with actual k₀",
            "spearman": spearman_ea_k0,
            "p_value": float(p_ea_k0),
            "threshold": h5_threshold,
            "pass": h5_pass,
            "valid_pairs": len(valid_k0_pairs) if len(valid_k0_pairs) > 0 else 0
        },
        "problem_analyses": problem_analyses,
        "summary": {
            "total_problems": len(problems),
            "analyzed_problems": len(problem_analyses),
            "valid_ea_estimates": len([e for e in ea_estimates if e > 0]),
            "spearman_ea_level": float(spearman_ea_level),
            "spearman_ea_k0": float(spearman_ea_k0)
        }
    }

    return results

def make_serializable(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def main():
    results = analyze_g2_consistency()

    # Save results
    results_dir = WORKSPACE / "exp/results/pilots/g2_consistency"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Make serializable
    serializable_results = make_serializable(results)

    results_file = results_dir / "g2_consistency_results.json"
    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save summary
    summary_file = results_dir / "g2_consistency_summary.md"
    with open(summary_file, "w") as f:
        f.write("# G2 Consistency Experiment Summary\n\n")
        f.write(f"## Task: g2_consistency\n")
        f.write(f"## Date: {results.get('timestamp', 'N/A')}\n\n")
        f.write("## Overall Status\n")
        f.write(f"**Recommendation**: {results['recommendation']}\n")
        f.write(f"**H2 Status**: {'PASS' if results['h2_evaluation']['pass'] else 'FAIL'}\n")
        f.write(f"**H5 Status**: {'PASS' if results['h5_evaluation']['pass'] else 'FAIL'}\n\n")
        f.write("## H2 Evaluation\n")
        f.write(f"- Spearman(Ea, Level): {results['h2_evaluation']['spearman']:.3f}\n")
        f.write(f"- Threshold: > {results['h2_evaluation']['threshold']}\n")
        f.write(f"- Pass: {results['h2_evaluation']['pass']}\n\n")
        f.write("## H5 Evaluation\n")
        f.write(f"- Spearman(Ea, k₀): {results['h5_evaluation']['spearman']:.3f}\n")
        f.write(f"- Threshold: > {results['h5_evaluation']['threshold']}\n")
        f.write(f"- Pass: {results['h5_evaluation']['pass']}\n")
    print(f"Summary saved to: {summary_file}")

    # Write DONE marker
    done_file = results_dir / "g2_consistency_DONE"
    done_file.write_text(json.dumps({
        "task_id": "g2_consistency",
        "status": "success",
        "recommendation": results['recommendation'],
        "h2_pass": bool(results['h2_evaluation']['pass']),
        "h5_pass": bool(results['h5_evaluation']['pass']),
        "spearman_ea_level": float(results['h2_evaluation']['spearman']),
        "spearman_ea_k0": float(results['h5_evaluation']['spearman']),
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }))

    return results

if __name__ == "__main__":
    main()
