#!/usr/bin/env python3
"""
Analysis H3: Single-pass threshold evaluation

Evaluate whether Ea (activation energy) from consistency signals can predict
single-pass solveability. Uses full G1 saturation data (50 problems).

Uses the CORRECT pilot method: fit C_k = c_inf * (1 - exp(-k/c0)) to consistency
trajectory, then Ea = 1/c0.

Input: full_g1_saturation_results.json
Output: analysis_h3.json with threshold analysis, ROC metrics, and H3 evaluation
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
OUTPUT_DIR = RESULTS_DIR / "full/analysis_h3"
OUTPUT_JSON = OUTPUT_DIR / "analysis_h3.json"
OUTPUT_MD = OUTPUT_DIR / "analysis_h3_summary.md"
PID_FILE = OUTPUT_DIR / "analysis_h3.pid"
PROGRESS_FILE = OUTPUT_DIR / "analysis_h3_PROGRESS.json"
DONE_FILE = OUTPUT_DIR / "analysis_h3_DONE"

K_VALUES = [1, 2, 4, 8, 16]


def convert_bools(obj):
    if isinstance(obj, dict):
        return {k: convert_bools(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_bools(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def report_progress(step, total_steps, message=""):
    """Write progress file."""
    progress = {
        "task_id": "analysis_h3",
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
    marker = {
        "task_id": "analysis_h3",
        "status": status,
        "summary": summary,
        "data": data,
        "timestamp": datetime.now().isoformat(),
    }
    DONE_FILE.write_text(json.dumps(convert_bools(marker)))


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


def compute_single_pass_accuracy(problem_data):
    """Get single-pass (k=1) accuracy for a problem."""
    k_results = problem_data["k_results"]
    k1 = k_results.get("1", {})
    correct = k1.get("correct", False)
    return 1.0 if correct else 0.0


def find_optimal_threshold(ea_values, accuracy_values):
    """Find optimal Ea threshold that maximizes accuracy difference."""
    if len(ea_values) < 2:
        return None, 0, 0, 0

    # Sort by Ea
    sorted_pairs = sorted(zip(ea_values, accuracy_values))
    sorted_ea = [p[0] for p in sorted_pairs]
    sorted_acc = [p[1] for p in sorted_pairs]

    best_threshold = sorted_ea[0]
    best_delta = 0
    best_low_acc = 0
    best_high_acc = 0

    # Try each unique Ea as threshold
    for i in range(1, len(sorted_ea)):
        threshold = (sorted_ea[i-1] + sorted_ea[i]) / 2

        low_acc = [a for e, a in zip(sorted_ea, sorted_acc) if e <= threshold]
        high_acc = [a for e, a in zip(sorted_ea, sorted_acc) if e > threshold]

        if not low_acc or not high_acc:
            continue

        low_mean = np.mean(low_acc)
        high_mean = np.mean(high_acc)
        delta = low_mean - high_mean

        if delta > best_delta:
            best_delta = delta
            best_threshold = threshold
            best_low_acc = low_mean
            best_high_acc = high_mean

    return best_threshold, best_delta, best_low_acc, best_high_acc


def compute_roc_metrics(ea_values, accuracy_values):
    """Compute ROC-like metrics for Ea as a predictor."""
    if len(set(accuracy_values)) < 2:
        return {"auc": 0.5, "note": "All same label"}

    # Sort by Ea ascending (low Ea = predicted easy = predicted correct)
    sorted_pairs = sorted(zip(ea_values, accuracy_values))
    sorted_ea = [p[0] for p in sorted_pairs]
    sorted_acc = [p[1] for p in sorted_pairs]

    n_pos = sum(accuracy_values)  # Actually correct
    n_neg = len(accuracy_values) - n_pos  # Actually incorrect

    if n_pos == 0 or n_neg == 0:
        return {"auc": 0.5, "note": "All same label"}

    # Compute TPR and FPR at each threshold
    tprs = []
    fprs = []

    for i in range(len(sorted_ea) + 1):
        # Predict positive (correct) for Ea <= threshold
        # i.e., first i elements
        predicted_pos = i
        predicted_neg = len(sorted_ea) - i

        tp = sum(sorted_acc[:i]) if i > 0 else 0
        fp = predicted_pos - tp
        fn = n_pos - tp
        tn = n_neg - fp

        tpr = tp / n_pos if n_pos > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0

        tprs.append(tpr)
        fprs.append(fpr)

    # Compute AUC using trapezoidal rule
    auc = 0
    for i in range(len(fprs) - 1):
        auc += (fprs[i+1] - fprs[i]) * (tprs[i+1] + tprs[i]) / 2

    return {
        "auc": float(auc),
        "tpr": [float(t) for t in tprs],
        "fpr": [float(f) for f in fprs],
    }


def main():
    import os
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    print("Loading G1 saturation data...")
    report_progress(1, 6, "Loading data")
    with open(INPUT_FILE) as f:
        data = json.load(f)

    problems = data["problems"]
    print(f"Loaded {len(problems)} problems")

    # Extract Ea estimates and single-pass accuracy for each problem
    print("\nEstimating Ea and extracting single-pass accuracy...")
    report_progress(2, 6, "Extracting features")

    problem_analyses = []
    ea_values = []
    accuracy_values = []
    level_values = []

    for p in problems:
        trajectory = compute_consistency_trajectory(p["k_results"])
        ea_estimate, c0, final_consistency = estimate_ea_from_consistency(trajectory)
        acc = compute_single_pass_accuracy(p)
        level = p.get("level", "Unknown")

        # Extract numeric level
        level_num = 0
        if isinstance(level, str) and "Level" in level:
            try:
                level_num = int(level.replace("Level ", ""))
            except Exception:
                level_num = 0

        problem_analyses.append({
            "idx": p["idx"],
            "level": level,
            "level_num": level_num,
            "ea": float(ea_estimate),
            "c0": float(c0),
            "final_consistency": float(final_consistency),
            "single_pass_correct": acc == 1.0,
        })

        ea_values.append(ea_estimate)
        accuracy_values.append(acc)
        level_values.append(level_num)

    print(f"  Ea range: [{min(ea_values):.4f}, {max(ea_values):.4f}]")
    print(f"  Overall single-pass accuracy: {np.mean(accuracy_values):.3f}")

    # Find optimal threshold
    print("\nFinding optimal Ea threshold...")
    report_progress(3, 6, "Finding optimal threshold")

    opt_threshold, opt_delta, opt_low_acc, opt_high_acc = find_optimal_threshold(ea_values, accuracy_values)

    print(f"  Optimal threshold: {opt_threshold:.4f}")
    print(f"  Low-Ea accuracy: {opt_low_acc:.3f}")
    print(f"  High-Ea accuracy: {opt_high_acc:.3f}")
    print(f"  Delta: {opt_delta:.3f}")

    # Compute ROC metrics
    print("\nComputing ROC metrics...")
    report_progress(4, 6, "Computing ROC")

    roc = compute_roc_metrics(ea_values, accuracy_values)
    print(f"  AUC: {roc.get('auc', 0):.4f}")

    # Correlation analysis
    print("\nCorrelation analysis...")
    report_progress(5, 6, "Correlation analysis")

    # Ea vs single-pass accuracy
    if len(set(ea_values)) > 1 and len(set(accuracy_values)) > 1:
        spearman_ea_acc, p_ea_acc = st.spearmanr(ea_values, accuracy_values)
        pearson_ea_acc, _ = st.pearsonr(ea_values, accuracy_values)
    else:
        spearman_ea_acc, p_ea_acc = 0, 1
        pearson_ea_acc = 0

    # Ea vs level
    if len(set(ea_values)) > 1 and len(set(level_values)) > 1:
        spearman_ea_level, p_ea_level = st.spearmanr(ea_values, level_values)
    else:
        spearman_ea_level, p_ea_level = 0, 1

    # Level vs accuracy
    if len(set(level_values)) > 1 and len(set(accuracy_values)) > 1:
        spearman_level_acc, p_level_acc = st.spearmanr(level_values, accuracy_values)
    else:
        spearman_level_acc, p_level_acc = 0, 1

    print(f"  Spearman(Ea, accuracy): {spearman_ea_acc:.4f} (p={p_ea_acc:.4f})")
    print(f"  Spearman(Ea, level): {spearman_ea_level:.4f} (p={p_ea_level:.4f})")
    print(f"  Spearman(level, accuracy): {spearman_level_acc:.4f} (p={p_level_acc:.4f})")

    # H3 evaluation
    print("\nH3 Evaluation...")
    report_progress(6, 6, "H3 evaluation")

    # H3: Single-pass accuracy > 75% on low-Ea problems
    h3_threshold = 0.75
    low_ea_acc = opt_low_acc if opt_low_acc is not None else 0
    h3_pass = low_ea_acc >= h3_threshold

    # Also evaluate: is Ea a useful predictor at all?
    # AUC > 0.6 means better than random
    auc = roc.get("auc", 0.5)
    useful_predictor = auc > 0.6

    if h3_pass:
        h3_status = "CONFIRMED"
        h3_notes = f"Low-Ea accuracy ({low_ea_acc:.1%}) meets threshold ({h3_threshold:.0%}). Ea is a useful routing signal."
    elif useful_predictor:
        h3_status = "PARTIAL"
        h3_notes = f"Low-Ea accuracy ({low_ea_acc:.1%}) below threshold ({h3_threshold:.0%}), but Ea has predictive power (AUC={auc:.3f})."
    else:
        h3_status = "FALSIFIED"
        h3_notes = f"Low-Ea accuracy ({low_ea_acc:.1%}) below threshold ({h3_threshold:.0%}) and Ea has no predictive power (AUC={auc:.3f})."

    print(f"  H3 Status: {h3_status}")
    print(f"  Low-Ea accuracy: {low_ea_acc:.3f} (threshold: {h3_threshold})")
    print(f"  AUC: {auc:.3f}")

    # Level-based analysis
    level_stats = {}
    for level_num in sorted(set(level_values)):
        if level_num == 0:
            continue
        indices = [i for i, lv in enumerate(level_values) if lv == level_num]
        level_ea = [ea_values[i] for i in indices]
        level_acc = [accuracy_values[i] for i in indices]

        level_stats[f"Level {level_num}"] = {
            "count": len(indices),
            "mean_ea": float(np.mean(level_ea)),
            "std_ea": float(np.std(level_ea)),
            "accuracy": float(np.mean(level_acc)),
        }

    # Build output
    output = {
        "task_id": "analysis_h3",
        "n_problems": len(problems),
        "h3_evaluation": {
            "hypothesis": "H3: Single-pass accuracy > 75% on low-Ea problems",
            "threshold_used": float(opt_threshold) if opt_threshold else None,
            "low_ea_accuracy": float(opt_low_acc) if opt_low_acc else None,
            "high_ea_accuracy": float(opt_high_acc) if opt_high_acc else None,
            "delta": float(opt_delta) if opt_delta else None,
            "threshold": h3_threshold,
            "pass": h3_pass,
            "status": h3_status,
            "notes": h3_notes,
        },
        "predictor_quality": {
            "auc": float(auc),
            "useful_predictor": useful_predictor,
            "spearman_ea_accuracy": float(spearman_ea_acc),
            "p_value_ea_accuracy": float(p_ea_acc),
            "pearson_ea_accuracy": float(pearson_ea_acc),
        },
        "correlations": {
            "spearman_ea_level": float(spearman_ea_level),
            "p_value_ea_level": float(p_ea_level),
            "spearman_level_accuracy": float(spearman_level_acc),
            "p_value_level_accuracy": float(p_level_acc),
        },
        "level_stats": level_stats,
        "problem_data": problem_analyses,
        "recommendation": "GO" if h3_pass else "REFINE" if useful_predictor else "NO_GO",
    }

    # Save JSON
    OUTPUT_JSON.write_text(json.dumps(convert_bools(output), indent=2))
    print(f"\nSaved: {OUTPUT_JSON}")

    # Save Markdown summary
    md = f"""# Analysis H3: Single-Pass Threshold Evaluation

## Task: analysis_h3
## Date: {datetime.now().isoformat()}
## Samples: {len(problems)} problems

## H3 Evaluation

**Hypothesis**: Single-pass accuracy > 75% on low-Ea problems

**Status**: {h3_status}

**Key Metrics**:
- Optimal Ea threshold: {opt_threshold if opt_threshold is not None else 'N/A'}
- Low-Ea accuracy: {opt_low_acc if opt_low_acc is not None else 'N/A'} (threshold: {h3_threshold})
- High-Ea accuracy: {opt_high_acc if opt_high_acc is not None else 'N/A'}
- Delta: {opt_delta if opt_delta is not None else 'N/A'}

**Pass**: {'YES' if h3_pass else 'NO'}

## Predictor Quality

| Metric | Value |
|--------|-------|
| AUC | {auc:.4f} |
| Spearman(Ea, accuracy) | {spearman_ea_acc:.4f} (p={p_ea_acc:.4f}) |
| Pearson(Ea, accuracy) | {pearson_ea_acc:.4f} |

## Correlations

| Pair | Spearman | p-value |
|------|----------|---------|
| Ea vs Level | {spearman_ea_level:.4f} | {p_ea_level:.4f} |
| Level vs Accuracy | {spearman_level_acc:.4f} | {p_level_acc:.4f} |

## Accuracy by MATH Level

| Level | Count | Mean Ea | Accuracy |
|-------|-------|---------|----------|
"""
    for level_name, stats in level_stats.items():
        md += f"| {level_name} | {stats['count']} | {stats['mean_ea']:.4f} | {stats['accuracy']:.1%} |\n"

    md += f"""
## Interpretation

{h3_notes}

## Recommendation

{output['recommendation']}

- **GO**: Ea reliably predicts single-pass solveability (>75% accuracy for low-Ea)
- **REFINE**: Ea has some predictive power but doesn't meet the 75% threshold
- **NO_GO**: Ea is not a useful routing signal
"""

    OUTPUT_MD.write_text(md)
    print(f"Saved: {OUTPUT_MD}")

    # Mark done
    mark_done(
        status="success",
        summary=f"H3 {h3_status}: low-Ea acc={opt_low_acc:.3f}, AUC={auc:.3f}, Spearman={spearman_ea_acc:.4f}",
        data=output
    )

    print("\nAnalysis H3 complete.")
    print(f"  H3 Status: {h3_status}")
    print(f"  Recommendation: {output['recommendation']}")


if __name__ == "__main__":
    main()
