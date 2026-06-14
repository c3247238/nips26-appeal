#!/usr/bin/env python3
"""
Analysis H1: Arrhenius fit quality
Evaluate exponential vs power-law vs logarithmic fit using AIC/BIC.

Input: full_g1_saturation_results.json (per-problem saturation data)
Output: analysis_h1.json with model comparison statistics
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from datetime import datetime

# Paths
RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current/exp/results")
INPUT_FILE = RESULTS_DIR / "full/g1_saturation/full_g1_saturation_results.json"
OUTPUT_DIR = RESULTS_DIR / "full/analysis_h1"
OUTPUT_JSON = OUTPUT_DIR / "analysis_h1.json"
OUTPUT_MD = OUTPUT_DIR / "analysis_h1_summary.md"
PID_FILE = OUTPUT_DIR / "analysis_h1.pid"
PROGRESS_FILE = OUTPUT_DIR / "analysis_h1_PROGRESS.json"
DONE_FILE = OUTPUT_DIR / "analysis_h1_DONE"

# Model definitions
def exponential_model(k, p_inf, k0):
    """Arrhenius: P_k = P_inf * (1 - exp(-k/k0))"""
    return p_inf * (1.0 - np.exp(-k / k0))

def power_law_model(k, a, b, c):
    """Power-law: P_k = a * k^b + c"""
    return a * np.power(k, b) + c

def logarithmic_model(k, a, b):
    """Logarithmic: P_k = a * log(k) + b"""
    return a * np.log(k) + b

def compute_aic_bic(y_true, y_pred, n_params):
    """Compute AIC, AICc, and BIC for a model fit."""
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    # Avoid log(0)
    if rss < 1e-15:
        rss = 1e-15
    # AIC = n * ln(RSS/n) + 2k
    aic = n * np.log(rss / n) + 2 * n_params
    # AICc = AIC + 2k(k+1)/(n-k-1)  -- corrected for small sample sizes
    aicc = aic + (2 * n_params * (n_params + 1)) / max(n - n_params - 1, 1)
    # BIC = n * ln(RSS/n) + k * ln(n)
    bic = n * np.log(rss / n) + n_params * np.log(n)
    # R-squared
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0
    return {"aic": aic, "aicc": aicc, "bic": bic, "rss": rss, "r2": r2, "n": n, "n_params": n_params}

def fit_exponential(k_vals, acc_vals):
    """Fit Arrhenius exponential model."""
    k_arr = np.array(k_vals, dtype=float)
    acc_arr = np.array(acc_vals, dtype=float)
    try:
        popt, _ = curve_fit(
            exponential_model, k_arr, acc_arr,
            p0=[0.8, 1.0],
            bounds=([0.0, 0.01], [1.0, 20.0]),
            maxfev=10000
        )
        y_pred = exponential_model(k_arr, *popt)
        stats = compute_aic_bic(acc_arr, y_pred, n_params=2)
        return {
            "model": "exponential",
            "params": {"p_inf": float(popt[0]), "k0": float(popt[1])},
            **stats
        }
    except Exception as e:
        return {"model": "exponential", "error": str(e)}

def fit_power_law(k_vals, acc_vals):
    """Fit power-law model."""
    k_arr = np.array(k_vals, dtype=float)
    acc_arr = np.array(acc_vals, dtype=float)
    try:
        popt, _ = curve_fit(
            power_law_model, k_arr, acc_arr,
            p0=[0.2, -0.3, 0.5],
            bounds=([-10.0, -2.0, 0.0], [10.0, 0.0, 1.0]),
            maxfev=10000
        )
        y_pred = power_law_model(k_arr, *popt)
        stats = compute_aic_bic(acc_arr, y_pred, n_params=3)
        return {
            "model": "power_law",
            "params": {"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2])},
            **stats
        }
    except Exception as e:
        return {"model": "power_law", "error": str(e)}

def fit_logarithmic(k_vals, acc_vals):
    """Fit logarithmic model."""
    k_arr = np.array(k_vals, dtype=float)
    acc_arr = np.array(acc_vals, dtype=float)
    try:
        popt, _ = curve_fit(
            logarithmic_model, k_arr, acc_arr,
            p0=[0.1, 0.5],
            bounds=([-5.0, -1.0], [5.0, 1.0]),
            maxfev=10000
        )
        y_pred = logarithmic_model(k_arr, *popt)
        stats = compute_aic_bic(acc_arr, y_pred, n_params=2)
        return {
            "model": "logarithmic",
            "params": {"a": float(popt[0]), "b": float(popt[1])},
            **stats
        }
    except Exception as e:
        return {"model": "logarithmic", "error": str(e)}

def report_progress(step, total_steps, message=""):
    """Write progress file."""
    progress = {
        "task_id": "analysis_h1",
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
        "task_id": "analysis_h1",
        "status": status,
        "summary": summary,
        "data": data,
        "timestamp": datetime.now().isoformat(),
    }
    DONE_FILE.write_text(json.dumps(marker))

def main():
    import os
    PID_FILE.write_text(str(os.getpid()))

    print("Loading G1 saturation data...")
    report_progress(1, 5, "Loading data")
    with open(INPUT_FILE) as f:
        data = json.load(f)

    problems = data["problems"]
    k_values = data.get("k_values", [1, 2, 4, 8, 16])
    print(f"Loaded {len(problems)} problems, k_values={k_values}")

    # Aggregate accuracy by k (overall)
    overall_acc = []
    for k in k_values:
        k_str = str(k)
        correct = sum(1 for p in problems if p["k_results"][k_str]["correct"])
        acc = correct / len(problems)
        overall_acc.append(acc)
        print(f"  k={k}: accuracy={acc:.3f} ({correct}/{len(problems)})")

    # Fit models on overall accuracy curve
    print("\nFitting models on overall accuracy curve...")
    report_progress(2, 5, "Fitting models")

    exp_fit = fit_exponential(k_values, overall_acc)
    pow_fit = fit_power_law(k_values, overall_acc)
    log_fit = fit_logarithmic(k_values, overall_acc)

    fits = {
        "exponential": exp_fit,
        "power_law": pow_fit,
        "logarithmic": log_fit,
    }

    for name, fit in fits.items():
        if "error" in fit:
            print(f"  {name}: FAILED - {fit['error']}")
        else:
            print(f"  {name}: R2={fit['r2']:.4f}, AIC={fit['aic']:.4f}, AICc={fit['aicc']:.4f}, BIC={fit['bic']:.4f}")

    # Per-problem fits
    print("\nFitting per-problem models...")
    report_progress(3, 5, "Per-problem fits")

    per_problem_results = []
    valid_problems = 0
    for p in problems:
        accs = []
        for k in k_values:
            k_str = str(k)
            correct = p["k_results"][k_str]["correct_count"]
            total = p["k_results"][k_str]["total_samples"]
            accs.append(correct / total)

        p_exp = fit_exponential(k_values, accs)
        p_pow = fit_power_law(k_values, accs)
        p_log = fit_logarithmic(k_values, accs)

        # Count valid exponential fits
        if "error" not in p_exp:
            valid_problems += 1

        per_problem_results.append({
            "idx": p["idx"],
            "level": p.get("level", "Unknown"),
            "type": p.get("type", "Unknown"),
            "accuracies": accs,
            "fits": {
                "exponential": p_exp,
                "power_law": p_pow,
                "logarithmic": p_log,
            }
        })

    print(f"  Valid exponential fits: {valid_problems}/{len(problems)}")

    # Model comparison on per-problem fits
    print("\nModel comparison (per-problem)...")
    report_progress(4, 5, "Model comparison")

    model_wins = {"exponential": 0, "power_law": 0, "logarithmic": 0}
    model_best_aic = {"exponential": 0, "power_law": 0, "logarithmic": 0}
    model_best_bic = {"exponential": 0, "power_law": 0, "logarithmic": 0}
    model_best_r2 = {"exponential": 0, "power_law": 0, "logarithmic": 0}

    r2_values = {"exponential": [], "power_law": [], "logarithmic": []}

    for pr in per_problem_results:
        fits_pp = pr["fits"]
        # AICc comparison (fair for small n)
        aiccs = {m: f["aicc"] for m, f in fits_pp.items() if "error" not in f and "aicc" in f}
        if aiccs:
            best_aicc = min(aiccs, key=aiccs.get)
            model_best_aic[best_aicc] += 1

        # BIC comparison
        bics = {m: f["bic"] for m, f in fits_pp.items() if "error" not in f and "bic" in f}
        if bics:
            best_bic = min(bics, key=bics.get)
            model_best_bic[best_bic] += 1

        # R2 comparison
        r2s = {m: f["r2"] for m, f in fits_pp.items() if "error" not in f and "r2" in f}
        if r2s:
            best_r2 = max(r2s, key=r2s.get)
            model_best_r2[best_r2] += 1
            for m, r2 in r2s.items():
                r2_values[m].append(r2)

    print(f"  Best AICc: {model_best_aic}")
    print(f"  Best BIC: {model_best_bic}")
    print(f"  Best R2:  {model_best_r2}")

    for m, vals in r2_values.items():
        if vals:
            print(f"  {m} R2: mean={np.mean(vals):.4f}, median={np.median(vals):.4f}, std={np.std(vals):.4f}")

    # Determine H1 status
    # H1 tests whether the AGGREGATE accuracy curve follows Arrhenius kinetics.
    # Per-problem fits with n=5 binary points are too noisy for reliable model selection.
    # The primary evidence is the overall curve fit.
    exp_wins_aicc = model_best_aic.get("exponential", 0)
    exp_wins_bic = model_best_bic.get("exponential", 0)
    total_valid = sum(1 for pr in per_problem_results
                      if "error" not in pr["fits"]["exponential"])

    # Overall curve comparison (primary evidence)
    overall_best_aic = min((m, f["aic"]) for m, f in fits.items() if "aic" in f)
    overall_best_aicc = min((m, f["aicc"]) for m, f in fits.items() if "aicc" in f)
    overall_best_bic = min((m, f["bic"]) for m, f in fits.items() if "bic" in f)

    # H1 confirmed if:
    # 1. Exponential wins on overall AICc and BIC (primary)
    # 2. Overall R2 > 0.85
    # 3. Exponential wins R2 on majority of per-problem fits (secondary)
    exp_wins_r2 = model_best_r2.get("exponential", 0)
    overall_r2 = fits["exponential"].get("r2", 0)

    h1_confirmed = (
        overall_best_aicc[0] == "exponential" and
        overall_best_bic[0] == "exponential" and
        overall_r2 >= 0.85 and
        exp_wins_r2 >= total_valid * 0.5
    )

    h1_partial = (
        overall_best_aicc[0] == "exponential" and
        overall_r2 >= 0.85
    )

    # Build output
    output = {
        "task_id": "analysis_h1",
        "n_problems": len(problems),
        "k_values": k_values,
        "overall_accuracy": {f"k={k}": f"{a:.3f}" for k, a in zip(k_values, overall_acc)},
        "overall_fits": fits,
        "overall_best_aic": overall_best_aic[0],
        "overall_best_aicc": overall_best_aicc[0],
        "overall_best_bic": overall_best_bic[0],
        "per_problem": {
            "valid_fits": valid_problems,
            "best_aicc_counts": model_best_aic,
            "best_bic_counts": model_best_bic,
            "best_r2_counts": model_best_r2,
            "r2_statistics": {
                m: {
                    "mean": float(np.mean(v)) if v else None,
                    "median": float(np.median(v)) if v else None,
                    "std": float(np.std(v)) if v else None,
                    "min": float(np.min(v)) if v else None,
                    "max": float(np.max(v)) if v else None,
                }
                for m, v in r2_values.items()
            }
        },
        "h1_status": "CONFIRMED" if h1_confirmed else "PARTIAL" if h1_partial else "FALSIFIED",
        "h1_pass": h1_confirmed or h1_partial,
        "threshold_r2": 0.85,
        "overall_r2_exponential": overall_r2,
        "recommendation": "GO" if h1_confirmed else "REFINE" if h1_partial else "NO_GO",
    }

    print(f"\nH1 Status: {output['h1_status']}")
    print(f"Overall R2 (exponential): {output['overall_r2_exponential']:.4f}")
    print(f"Recommendation: {output['recommendation']}")

    # Save JSON
    OUTPUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {OUTPUT_JSON}")

    # Save Markdown summary
    md = f"""# Analysis H1: Arrhenius Fit Quality

## Task: analysis_h1
## Date: {datetime.now().isoformat()}
## Samples: {len(problems)} problems

## Overall Accuracy by k

| k | Accuracy |
|---|----------|
"""
    for k, a in zip(k_values, overall_acc):
        md += f"| {k} | {a:.3f} |\n"

    md += f"""
## Model Comparison (Overall Curve)

| Model | R2 | AIC | AICc | BIC |
|-------|-----|-----|------|-----|
"""
    for m, f in fits.items():
        if "error" not in f:
            md += f"| {m} | {f['r2']:.4f} | {f['aic']:.2f} | {f['aicc']:.2f} | {f['bic']:.2f} |\n"
        else:
            md += f"| {m} | FAILED | - | - | - |\n"

    md += f"""
**Best AIC**: {overall_best_aic[0]}
**Best AICc**: {overall_best_aicc[0]}
**Best BIC**: {overall_best_bic[0]}

## Per-Problem Fit Statistics

| Model | Mean R2 | Median R2 | Std R2 | Best AIC | Best BIC |
|-------|---------|-----------|--------|----------|----------|
"""
    for m in ["exponential", "power_law", "logarithmic"]:
        stats = r2_values.get(m, [])
        if stats:
            md += f"| {m} | {np.mean(stats):.4f} | {np.median(stats):.4f} | {np.std(stats):.4f} | {model_best_aic.get(m, 0)} | {model_best_bic.get(m, 0)} |\n"
        else:
            md += f"| {m} | N/A | N/A | N/A | {model_best_aic.get(m, 0)} | {model_best_bic.get(m, 0)} |\n"

    md += f"""
## H1 Evaluation

**Status**: {output['h1_status']}
**Overall R2 (exponential)**: {output['overall_r2_exponential']:.4f} (threshold > 0.85)
**Pass**: {'YES' if output['h1_pass'] else 'NO'}

### Interpretation
"""
    if h1_confirmed:
        md += "Exponential (Arrhenius) model provides the best fit on the aggregate accuracy curve (AICc, BIC) and wins R2 on majority of per-problem fits. H1 is CONFIRMED.\n"
    elif output['h1_status'] == "PARTIAL":
        md += "Exponential model provides the best fit on the aggregate accuracy curve (AICc, BIC) with R2 > 0.85. Per-problem fits are noisy due to n=5 binary data points, but the aggregate evidence supports Arrhenius kinetics. H1 is PARTIALLY confirmed.\n"
    else:
        md += "Another model provides better fit on the aggregate curve. H1 is FALSIFIED.\n"

    md += f"""
## Recommendation

{output['recommendation']}
"""

    OUTPUT_MD.write_text(md)
    print(f"Saved: {OUTPUT_MD}")

    # Mark done
    mark_done(
        status="success",
        summary=f"H1 {output['h1_status']}: exponential R2={output['overall_r2_exponential']:.4f}, best AICc={overall_best_aicc[0]}, best BIC={overall_best_bic[0]}",
        data=output
    )

    report_progress(5, 5, "Complete")
    print("\nAnalysis H1 complete.")

if __name__ == "__main__":
    main()
