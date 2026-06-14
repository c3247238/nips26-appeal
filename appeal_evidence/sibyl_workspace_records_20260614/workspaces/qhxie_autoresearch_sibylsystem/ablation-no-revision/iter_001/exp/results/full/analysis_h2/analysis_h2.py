#!/usr/bin/env python3
"""
Analysis H2: Ea-difficulty correlation

Compute Spearman correlation between Ea (activation energy from consistency)
and MATH difficulty level using full G1 saturation data (50 problems).

Also evaluates H5: consistency-based Ea vs saturation-derived k0 correlation.

Uses the CORRECT pilot method: fit C_k = c_inf * (1 - exp(-k/c0)) to consistency
trajectory, then Ea = 1/c0.

Input: full_g1_saturation_results.json
Output: analysis_h2.json with correlation statistics
"""

import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from scipy.optimize import curve_fit
import scipy.stats as st

# Paths
RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/exp/results")
INPUT_FILE = RESULTS_DIR / "full/g1_saturation/full_g1_saturation_results.json"
OUTPUT_DIR = RESULTS_DIR / "full/analysis_h2"
OUTPUT_JSON = OUTPUT_DIR / "analysis_h2.json"
OUTPUT_MD = OUTPUT_DIR / "analysis_h2_summary.md"
PID_FILE = OUTPUT_DIR / "analysis_h2.pid"
PROGRESS_FILE = OUTPUT_DIR / "analysis_h2_PROGRESS.json"
DONE_FILE = OUTPUT_DIR / "analysis_h2_DONE"

K_VALUES = [1, 2, 4, 8, 16]


def report_progress(step, total_steps, message=""):
    """Write progress file."""
    progress = {
        "task_id": "analysis_h2",
        "step": step,
        "total_steps": total_steps,
        "message": message,
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(progress))


def mark_done(status, summary, data):
    """Write DONE marker."""
    if PID_FILE.exists():
        PID_FILE.unlink()

    def convert_bools(obj):
        if isinstance(obj, dict):
            return {k: convert_bools(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_bools(v) for v in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    marker = {
        "task_id": "analysis_h2",
        "status": status,
        "summary": summary,
        "data": convert_bools(data),
        "timestamp": datetime.now().isoformat(),
    }
    DONE_FILE.write_text(json.dumps(marker))


def compute_consistency_at_k(k_results, k):
    """Compute consistency at sample count k.

    Consistency = fraction of samples that match the most common answer.
    """
    k_str = str(k)
    if k_str not in k_results:
        return 0.0

    answers = k_results[k_str].get("answers", [])
    if not answers:
        return 0.0

    answer_counts = Counter(answers)
    most_common_count = answer_counts.most_common(1)[0][1]
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

    Fit saturation model: C_k = c_inf * (1 - exp(-k / c0))
    Ea proxy = 1 / c0 (higher c0 = slower convergence = lower Ea)
    """
    ks = np.array(list(trajectory.keys()))
    cs = np.array([trajectory[k] for k in ks])

    if cs.min() < 0.1:
        return 0.0, 0.0, cs[-1]

    try:
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
        ea_proxy = 1.0 / c0
        return ea_proxy, c0, cs[-1]
    except Exception:
        return 0.0, 0.0, cs[-1]


def extract_k0_from_saturation_fit(problem_data):
    """Extract k0 from per-problem saturation fit if available."""
    fit = problem_data.get("saturation_fit")
    if fit and fit.get("fit_success"):
        return fit.get("k0", None)
    return None


def convert_bools(obj):
    if isinstance(obj, dict):
        return {k: convert_bools(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_bools(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    print("Loading G1 saturation data...")
    report_progress(1, 5, "Loading data")
    with open(INPUT_FILE) as f:
        data = json.load(f)

    problems = data["problems"]
    print(f"Loaded {len(problems)} problems")

    # Extract Ea estimates, levels, and k0 values
    print("\nExtracting Ea estimates and difficulty levels...")
    report_progress(2, 5, "Extracting features")

    ea_values = []
    level_values = []
    accuracy_values = []
    k0_values = []
    problem_analyses = []

    for p in problems:
        trajectory = compute_consistency_trajectory(p["k_results"])
        ea_estimate, c0, final_consistency = estimate_ea_from_consistency(trajectory)
        level = p.get("level", "Unknown")

        # Extract numeric level
        level_num = 0
        if isinstance(level, str) and "Level" in level:
            try:
                level_num = int(level.replace("Level ", ""))
            except Exception:
                level_num = 0

        # Single-pass accuracy
        k1 = p["k_results"].get("1", {})
        acc = 1.0 if k1.get("correct", False) else 0.0

        # k0 from saturation fit
        k0 = extract_k0_from_saturation_fit(p)

        problem_analyses.append({
            "idx": p["idx"],
            "level": level,
            "level_num": level_num,
            "ea": float(ea_estimate),
            "c0": float(c0),
            "final_consistency": float(final_consistency),
            "single_pass_correct": acc == 1.0,
            "k0": k0,
        })

        ea_values.append(ea_estimate)
        level_values.append(level_num)
        accuracy_values.append(acc)
        if k0 is not None:
            k0_values.append(k0)

    print(f"  Ea range: [{min(ea_values):.4f}, {max(ea_values):.4f}]")
    print(f"  Level range: [{min(level_values)}, {max(level_values)}]")
    print(f"  Problems with k0: {len(k0_values)}/{len(problems)}")

    # Correlation analysis
    print("\nComputing correlations...")
    report_progress(3, 5, "Computing correlations")

    # H2: Ea vs level
    valid_ea_level = [(e, l) for e, l in zip(ea_values, level_values) if e > 0]
    if len(valid_ea_level) > 10 and len(set([x[1] for x in valid_ea_level])) > 1:
        ea_arr = np.array([x[0] for x in valid_ea_level])
        level_arr = np.array([x[1] for x in valid_ea_level])
        spearman_ea_level, p_ea_level = st.spearmanr(ea_arr, level_arr)
        pearson_ea_level, _ = st.pearsonr(ea_arr, level_arr)
    else:
        spearman_ea_level, p_ea_level = 0, 1
        pearson_ea_level = 0

    # Ea vs accuracy (using all problems)
    if len(set(ea_values)) > 1 and len(set(accuracy_values)) > 1:
        spearman_ea_acc, p_ea_acc = st.spearmanr(ea_values, accuracy_values)
        pearson_ea_acc, _ = st.pearsonr(ea_values, accuracy_values)
    else:
        spearman_ea_acc, p_ea_acc = 0, 1
        pearson_ea_acc = 0

    # Level vs accuracy
    if len(set(level_values)) > 1 and len(set(accuracy_values)) > 1:
        spearman_level_acc, p_level_acc = st.spearmanr(level_values, accuracy_values)
    else:
        spearman_level_acc, p_level_acc = 0, 1

    # H5: Ea vs k0 (consistency-based vs saturation-derived)
    matched_ea = [p["ea"] for p in problem_analyses if p["k0"] is not None and p["ea"] > 0]
    matched_k0 = [p["k0"] for p in problem_analyses if p["k0"] is not None and p["ea"] > 0]
    if len(matched_ea) > 3 and len(set(matched_ea)) > 1 and len(set(matched_k0)) > 1:
        spearman_ea_k0, p_ea_k0 = st.spearmanr(matched_ea, matched_k0)
    else:
        spearman_ea_k0, p_ea_k0 = 0, 1

    print(f"  Spearman(Ea, level): {spearman_ea_level:.4f} (p={p_ea_level:.4f})")
    print(f"  Spearman(Ea, accuracy): {spearman_ea_acc:.4f} (p={p_ea_acc:.4f})")
    print(f"  Spearman(level, accuracy): {spearman_level_acc:.4f} (p={p_level_acc:.4f})")
    print(f"  Spearman(Ea, k0): {spearman_ea_k0:.4f} (p={p_ea_k0:.4f}) [H5]")

    # H2 evaluation
    print("\nH2 Evaluation...")
    report_progress(4, 5, "H2 evaluation")

    h2_threshold = 0.4
    h2_pass = abs(spearman_ea_level) >= h2_threshold

    if h2_pass:
        h2_status = "CONFIRMED"
        h2_notes = f"Ea correlates with MATH difficulty level (Spearman={spearman_ea_level:.4f}, p={p_ea_level:.4f})"
    else:
        h2_status = "FALSIFIED"
        h2_notes = f"Ea does NOT correlate with MATH difficulty level (Spearman={spearman_ea_level:.4f}, p={p_ea_level:.4f})"

    # H5 evaluation
    h5_threshold = 0.5
    h5_pass = abs(spearman_ea_k0) >= h5_threshold

    if h5_pass:
        h5_status = "CONFIRMED"
        h5_notes = f"Consistency-based Ea correlates with saturation-derived k0 (Spearman={spearman_ea_k0:.4f})"
    elif len(matched_ea) > 3:
        h5_status = "FALSIFIED"
        h5_notes = f"Consistency-based Ea does NOT correlate with saturation-derived k0 (Spearman={spearman_ea_k0:.4f})"
    else:
        h5_status = "INSUFFICIENT_DATA"
        h5_notes = f"Too few valid k0 values ({len(matched_ea)}) to evaluate H5"

    print(f"  H2 Status: {h2_status}")
    print(f"  H5 Status: {h5_status}")

    # Level-based statistics
    level_stats = {}
    for level_num in sorted(set(level_values)):
        if level_num == 0:
            continue
        indices = [i for i, lv in enumerate(level_values) if lv == level_num]
        level_ea = [ea_values[i] for i in indices if ea_values[i] > 0]
        level_acc = [accuracy_values[i] for i in indices]

        level_stats[f"Level {level_num}"] = {
            "count": len(indices),
            "mean_ea": float(np.mean(level_ea)) if level_ea else 0.0,
            "std_ea": float(np.std(level_ea)) if level_ea else 0.0,
            "accuracy": float(np.mean(level_acc)),
        }

    # Build output
    output = {
        "task_id": "analysis_h2",
        "n_problems": len(problems),
        "h2_evaluation": {
            "hypothesis": "H2: Activation energy correlates with MATH difficulty level",
            "threshold": h2_threshold,
            "spearman_ea_level": float(spearman_ea_level),
            "p_value_ea_level": float(p_ea_level),
            "pearson_ea_level": float(pearson_ea_level),
            "pass": h2_pass,
            "status": h2_status,
            "notes": h2_notes,
        },
        "h5_evaluation": {
            "hypothesis": "H5: Consistency-based Ea matches saturation-derived k0",
            "threshold": h5_threshold,
            "spearman_ea_k0": float(spearman_ea_k0),
            "p_value_ea_k0": float(p_ea_k0),
            "valid_k0_pairs": len(matched_ea),
            "pass": h5_pass,
            "status": h5_status,
            "notes": h5_notes,
        },
        "correlations": {
            "spearman_ea_accuracy": float(spearman_ea_acc),
            "p_value_ea_accuracy": float(p_ea_acc),
            "pearson_ea_accuracy": float(pearson_ea_acc),
            "spearman_level_accuracy": float(spearman_level_acc),
            "p_value_level_accuracy": float(p_level_acc),
        },
        "level_stats": level_stats,
        "problem_data": problem_analyses,
        "recommendation": "GO" if h2_pass else "NO_GO",
    }

    # Save JSON
    OUTPUT_JSON.write_text(json.dumps(convert_bools(output), indent=2))
    print(f"\nSaved: {OUTPUT_JSON}")

    # Save Markdown summary
    md = f"""# Analysis H2: Ea-Difficulty Correlation

## Task: analysis_h2
## Date: {datetime.now().isoformat()}
## Samples: {len(problems)} problems

## H2 Evaluation

**Hypothesis**: Activation energy correlates with MATH difficulty level

**Status**: {h2_status}

**Key Metrics**:
- Spearman(Ea, level): {spearman_ea_level:.4f} (p={p_ea_level:.4f})
- Pearson(Ea, level): {pearson_ea_level:.4f}
- Threshold: {h2_threshold}

**Pass**: {'YES' if h2_pass else 'NO'}

## H5 Evaluation

**Hypothesis**: Consistency-based Ea matches saturation-derived k0

**Status**: {h5_status}

**Key Metrics**:
- Spearman(Ea, k0): {spearman_ea_k0:.4f} (p={p_ea_k0:.4f})
- Valid k0 pairs: {len(matched_ea)}/{len(problems)}
- Threshold: {h5_threshold}

**Pass**: {'YES' if h5_pass else 'NO'}

## Additional Correlations

| Pair | Spearman | p-value |
|------|----------|---------|
| Ea vs Accuracy | {spearman_ea_acc:.4f} | {p_ea_acc:.4f} |
| Level vs Accuracy | {spearman_level_acc:.4f} | {p_level_acc:.4f} |

## Accuracy by MATH Level

| Level | Count | Mean Ea | Accuracy |
|-------|-------|---------|----------|
"""
    for level_name, stats in level_stats.items():
        md += f"| {level_name} | {stats['count']} | {stats['mean_ea']:.4f} | {stats['accuracy']:.1%} |\n"

    md += f"""
## Interpretation

{h2_notes}

{h5_notes}

## Recommendation

{output['recommendation']}

- **GO**: Ea reliably correlates with difficulty (Spearman > 0.4)
- **NO_GO**: Ea does not correlate with difficulty
"""

    OUTPUT_MD.write_text(md)
    print(f"Saved: {OUTPUT_MD}")

    # Mark done
    mark_done(
        status="success",
        summary=f"H2 {h2_status}: Spearman={spearman_ea_level:.4f} (p={p_ea_level:.4f}), H5 {h5_status}: Spearman={spearman_ea_k0:.4f}",
        data=output
    )

    report_progress(5, 5, "Complete")
    print("\nAnalysis H2 complete.")
    print(f"  H2 Status: {h2_status}")
    print(f"  H5 Status: {h5_status}")
    print(f"  Recommendation: {output['recommendation']}")


if __name__ == "__main__":
    main()
