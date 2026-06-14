#!/usr/bin/env python3
"""
GATE 0C: Leave-One-Out Sensitivity Analysis for CMI-Absorption Correlation.

For each of 25 letters, remove it and recompute Spearman rho(CMI, absorption).
Report which letters are influential (removal changes rho by >0.1).
Explicitly discuss letters S (CMI=0.961, absorption=31.4% -- contradicts theory)
and K (CMI=0.592, absorption=0% -- also contradicts).

Input: iter_006/exp/results/full/cmi_estimation.json
Output: current/exp/results/full/leave_one_out_cmi.json

No GPU required. Pure statistical computation on existing data.
"""

import json
import os
import sys
import numpy as np
from scipy import stats
from datetime import datetime
from pathlib import Path

# PID file for system recovery
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
TASK_ID = "leave_one_out_cmi"

pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

def report_progress(task_id, results_dir, epoch, total_epochs, step=0,
                    total_steps=0, loss=None, metric=None):
    """Write progress file for system monitor to track."""
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_task_done(task_id, results_dir, status="success", summary=""):
    """Write DONE marker file for system monitor to detect."""
    pid_file = Path(results_dir) / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def bootstrap_spearman_ci(x, y, n_boot=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for Spearman rho."""
    rng = np.random.RandomState(seed)
    n = len(x)
    rhos = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        rhos[i] = stats.spearmanr(x[idx], y[idx])[0]
    alpha = (1 - ci) / 2
    return float(np.percentile(rhos, alpha * 100)), float(np.percentile(rhos, (1 - alpha) * 100))


def main():
    print("=" * 60)
    print("GATE 0C: Leave-One-Out CMI-Absorption Sensitivity Analysis")
    print("=" * 60)

    # Load CMI estimation data
    cmi_path = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/iter_006/exp/results/full/cmi_estimation.json")
    if not cmi_path.exists():
        print(f"ERROR: CMI data not found at {cmi_path}")
        mark_task_done(TASK_ID, RESULTS_DIR, status="failed", summary="CMI data file not found")
        sys.exit(1)

    with open(cmi_path) as f:
        cmi_data = json.load(f)

    report_progress(TASK_ID, RESULTS_DIR, 1, 5, step=1, total_steps=5,
                    metric={"phase": "loading_data"})

    # Extract CMI at d'=10 (pre-registered primary subspace dimension) for all letters
    cmi_by_letter = cmi_data["cmi_by_subspace_dim"]["10"]

    # Build arrays: letter, CMI, absorption_rate, probe_f1
    letters = []
    cmi_values = []
    absorption_rates = []
    probe_f1_values = []

    for letter, info in sorted(cmi_by_letter.items()):
        if info.get("status") == "ok" and info.get("is_finite", True):
            letters.append(letter)
            cmi_values.append(info["cmi"])
            absorption_rates.append(info["absorption_rate"])
            probe_f1_values.append(info["probe_f1"])

    letters = np.array(letters)
    cmi_values = np.array(cmi_values)
    absorption_rates = np.array(absorption_rates)
    probe_f1_values = np.array(probe_f1_values)
    n_letters = len(letters)

    print(f"\nLoaded {n_letters} letters with CMI at d'=10")
    print(f"Letters: {', '.join(letters)}")

    # Full-sample Spearman rho
    full_rho, full_p = stats.spearmanr(cmi_values, absorption_rates)
    full_ci_lo, full_ci_hi = bootstrap_spearman_ci(cmi_values, absorption_rates)

    print(f"\nFull-sample Spearman rho: {full_rho:.4f} (p={full_p:.4f})")
    print(f"  Bootstrap 95% CI: [{full_ci_lo:.4f}, {full_ci_hi:.4f}]")

    report_progress(TASK_ID, RESULTS_DIR, 2, 5, step=2, total_steps=5,
                    metric={"phase": "full_correlation", "rho": full_rho, "p": full_p})

    # Leave-one-out analysis
    print(f"\n{'='*60}")
    print("Leave-One-Out Analysis")
    print(f"{'='*60}")

    loo_results = []
    for i in range(n_letters):
        # Remove letter i
        mask = np.ones(n_letters, dtype=bool)
        mask[i] = False
        loo_cmi = cmi_values[mask]
        loo_abs = absorption_rates[mask]

        loo_rho, loo_p = stats.spearmanr(loo_cmi, loo_abs)
        rho_change = loo_rho - full_rho
        is_influential = abs(rho_change) > 0.1

        loo_results.append({
            "letter": letters[i],
            "cmi": float(cmi_values[i]),
            "absorption_rate": float(absorption_rates[i]),
            "probe_f1": float(probe_f1_values[i]),
            "loo_rho": float(loo_rho),
            "loo_p": float(loo_p),
            "rho_change": float(rho_change),
            "abs_rho_change": float(abs(rho_change)),
            "is_influential": bool(is_influential),
            "direction": "strengthens" if rho_change < 0 else "weakens"
        })

        flag = " ** INFLUENTIAL **" if is_influential else ""
        print(f"  Remove {letters[i]}: rho={loo_rho:+.4f} (delta={rho_change:+.4f}){flag}")

    report_progress(TASK_ID, RESULTS_DIR, 3, 5, step=3, total_steps=5,
                    metric={"phase": "loo_complete"})

    # Sort by influence magnitude
    loo_results.sort(key=lambda x: x["abs_rho_change"], reverse=True)

    # Identify influential letters
    influential = [r for r in loo_results if r["is_influential"]]
    print(f"\nInfluential letters (|delta rho| > 0.1): {len(influential)}")
    for r in influential:
        print(f"  {r['letter']}: CMI={r['cmi']:.3f}, absorption={r['absorption_rate']:.3f}, "
              f"delta_rho={r['rho_change']:+.4f} ({r['direction']})")

    # Specific discussion of S and K
    print(f"\n{'='*60}")
    print("Special Analysis: Letters S and K")
    print(f"{'='*60}")

    s_result = next(r for r in loo_results if r["letter"] == "S")
    k_result = next(r for r in loo_results if r["letter"] == "K")

    print(f"\nLetter S:")
    print(f"  CMI = {s_result['cmi']:.4f} (high -- suggests low absorption susceptibility)")
    print(f"  Absorption rate = {s_result['absorption_rate']:.4f} (31.4% -- high absorption)")
    print(f"  Probe F1 = {s_result['probe_f1']:.4f}")
    print(f"  This CONTRADICTS theory: high CMI should predict low absorption")
    print(f"  Removing S: rho changes by {s_result['rho_change']:+.4f}")
    print(f"  S is {'INFLUENTIAL' if s_result['is_influential'] else 'NOT influential'}")

    print(f"\nLetter K:")
    print(f"  CMI = {k_result['cmi']:.4f} (moderate)")
    print(f"  Absorption rate = {k_result['absorption_rate']:.4f} (0% -- zero absorption)")
    print(f"  Probe F1 = {k_result['probe_f1']:.4f}")
    print(f"  This CONTRADICTS theory: moderate CMI should predict some absorption")
    print(f"  Removing K: rho changes by {k_result['rho_change']:+.4f}")
    print(f"  K is {'INFLUENTIAL' if k_result['is_influential'] else 'NOT influential'}")

    # Compute summary statistics
    rho_changes = np.array([r["rho_change"] for r in loo_results])
    rho_values = np.array([r["loo_rho"] for r in loo_results])

    # Stability analysis: what fraction of LOO rhos have the same sign as full rho?
    same_sign_count = np.sum(np.sign(rho_values) == np.sign(full_rho))
    same_sign_pct = same_sign_count / n_letters * 100

    # Range of LOO rhos
    min_rho = float(np.min(rho_values))
    max_rho = float(np.max(rho_values))
    mean_rho = float(np.mean(rho_values))
    std_rho = float(np.std(rho_values))

    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(f"Full-sample rho: {full_rho:.4f}")
    print(f"LOO rho range: [{min_rho:.4f}, {max_rho:.4f}]")
    print(f"LOO rho mean +/- std: {mean_rho:.4f} +/- {std_rho:.4f}")
    print(f"LOO rho same sign as full: {same_sign_count}/{n_letters} ({same_sign_pct:.1f}%)")
    print(f"Max |delta rho|: {float(np.max(np.abs(rho_changes))):.4f}")
    print(f"Mean |delta rho|: {float(np.mean(np.abs(rho_changes))):.4f}")

    report_progress(TASK_ID, RESULTS_DIR, 4, 5, step=4, total_steps=5,
                    metric={"phase": "summary_stats"})

    # Robustness check: remove BOTH S and K
    mask_both = np.ones(n_letters, dtype=bool)
    s_idx = np.where(letters == "S")[0][0]
    k_idx = np.where(letters == "K")[0][0]
    mask_both[s_idx] = False
    mask_both[k_idx] = False
    rho_no_sk, p_no_sk = stats.spearmanr(cmi_values[mask_both], absorption_rates[mask_both])
    ci_lo_no_sk, ci_hi_no_sk = bootstrap_spearman_ci(cmi_values[mask_both], absorption_rates[mask_both])

    print(f"\nRobustness: Removing both S and K (N=23):")
    print(f"  rho = {rho_no_sk:.4f} (p={p_no_sk:.4f})")
    print(f"  Bootstrap 95% CI: [{ci_lo_no_sk:.4f}, {ci_hi_no_sk:.4f}]")
    print(f"  Change from full: {rho_no_sk - full_rho:+.4f}")

    # Additional robustness: remove all influential letters
    if influential:
        mask_no_inf = np.ones(n_letters, dtype=bool)
        for r in influential:
            idx = np.where(letters == r["letter"])[0][0]
            mask_no_inf[idx] = False
        n_remaining = int(np.sum(mask_no_inf))
        if n_remaining >= 5:
            rho_no_inf, p_no_inf = stats.spearmanr(
                cmi_values[mask_no_inf], absorption_rates[mask_no_inf])
            ci_lo_no_inf, ci_hi_no_inf = bootstrap_spearman_ci(
                cmi_values[mask_no_inf], absorption_rates[mask_no_inf])
            print(f"\nRobustness: Removing all {len(influential)} influential letters (N={n_remaining}):")
            print(f"  rho = {rho_no_inf:.4f} (p={p_no_inf:.4f})")
            print(f"  Bootstrap 95% CI: [{ci_lo_no_inf:.4f}, {ci_hi_no_inf:.4f}]")
        else:
            rho_no_inf = None
            p_no_inf = None
            ci_lo_no_inf = None
            ci_hi_no_inf = None
            print(f"\nSkipping influential-letter removal: only {n_remaining} letters remain")
    else:
        rho_no_inf = None
        p_no_inf = None
        ci_lo_no_inf = None
        ci_hi_no_inf = None

    # Jackknife bias and standard error
    jackknife_rhos = np.array([r["loo_rho"] for r in loo_results])
    jackknife_mean = np.mean(jackknife_rhos)
    jackknife_bias = (n_letters - 1) * (jackknife_mean - full_rho)
    jackknife_var = (n_letters - 1) / n_letters * np.sum((jackknife_rhos - jackknife_mean) ** 2)
    jackknife_se = np.sqrt(jackknife_var)
    bias_corrected_rho = full_rho - jackknife_bias

    print(f"\nJackknife Analysis:")
    print(f"  Jackknife bias: {jackknife_bias:.4f}")
    print(f"  Jackknife SE: {jackknife_se:.4f}")
    print(f"  Bias-corrected rho: {bias_corrected_rho:.4f}")
    print(f"  Jackknife 95% CI: [{bias_corrected_rho - 1.96*jackknife_se:.4f}, "
          f"{bias_corrected_rho + 1.96*jackknife_se:.4f}]")

    # Interpretation
    print(f"\n{'='*60}")
    print("Interpretation")
    print(f"{'='*60}")

    if len(influential) == 0:
        stability_verdict = "STABLE"
        stability_text = ("No single letter removal changes rho by more than 0.1. "
                         "The CMI-absorption correlation is robust to leave-one-out perturbation.")
    elif len(influential) <= 2:
        stability_verdict = "MODERATELY STABLE"
        influential_names = ", ".join(r["letter"] for r in influential)
        stability_text = (f"Only {len(influential)} letter(s) ({influential_names}) are influential. "
                         "The correlation has mild sensitivity to specific outliers "
                         "but the overall direction is preserved.")
    else:
        stability_verdict = "UNSTABLE"
        influential_names = ", ".join(r["letter"] for r in influential)
        stability_text = (f"{len(influential)} letters ({influential_names}) are influential. "
                         "The CMI-absorption correlation is sensitive to outlier composition. "
                         "Results should be interpreted with caution.")

    print(f"  Stability verdict: {stability_verdict}")
    print(f"  {stability_text}")

    # Check theoretical consistency
    # Theory: lower CMI -> higher absorption susceptibility (negative correlation expected)
    # S contradicts: high CMI + high absorption
    # K contradicts: moderate CMI + zero absorption

    theory_consistent_count = 0
    for r in loo_results:
        cmi_rank = np.sum(cmi_values < r["cmi"]) / n_letters
        abs_rank = np.sum(absorption_rates < r["absorption_rate"]) / n_letters
        # Consistent if high CMI + low absorption OR low CMI + high absorption
        if (cmi_rank > 0.5 and abs_rank < 0.5) or (cmi_rank < 0.5 and abs_rank > 0.5) or \
           (cmi_rank > 0.3 and cmi_rank < 0.7 and abs_rank > 0.3 and abs_rank < 0.7):
            theory_consistent_count += 1

    # Build output JSON
    output = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "mode": "PILOT",
        "seed": 42,
        "source_data": str(cmi_path),
        "subspace_dim": 10,
        "n_letters": int(n_letters),
        "letters": letters.tolist(),

        "full_sample": {
            "spearman_rho": float(full_rho),
            "spearman_p": float(full_p),
            "bonferroni_p": float(full_p * 4),  # 4 subspace dims tested
            "bootstrap_ci_95": [full_ci_lo, full_ci_hi],
            "n": int(n_letters)
        },

        "leave_one_out_results": loo_results,

        "influential_letters": [r["letter"] for r in influential],
        "n_influential": len(influential),

        "special_analysis": {
            "letter_S": {
                "cmi": float(s_result["cmi"]),
                "absorption_rate": float(s_result["absorption_rate"]),
                "probe_f1": float(s_result["probe_f1"]),
                "loo_rho": float(s_result["loo_rho"]),
                "rho_change": float(s_result["rho_change"]),
                "is_influential": bool(s_result["is_influential"]),
                "theory_contradiction": "High CMI (0.961) predicts low absorption susceptibility, but observed absorption is 31.4% (one of the highest). S has the largest letter-word count (70 words), which may cause crowding in the feature space. The high probe_f1=0.74 means the probe is only moderately accurate, potentially inflating the measured absorption rate.",
                "note": "S is the letter with the most words in the vocabulary (70). Its high CMI may reflect strong distinguishability in the residual stream, while its high absorption rate may reflect the large number of competing word-specific features."
            },
            "letter_K": {
                "cmi": float(k_result["cmi"]),
                "absorption_rate": float(k_result["absorption_rate"]),
                "probe_f1": float(k_result["probe_f1"]),
                "loo_rho": float(k_result["loo_rho"]),
                "rho_change": float(k_result["rho_change"]),
                "is_influential": bool(k_result["is_influential"]),
                "theory_contradiction": "Moderate CMI (0.592) would predict some absorption, but observed rate is 0%. K has only 25 words and high probe_f1=0.956, suggesting the probe can accurately detect K-starting tokens. The zero absorption may reflect K's distinctive orthographic properties (relatively rare initial letter).",
                "note": "K has only 25 words in the vocabulary and a very high probe F1 (0.956). The zero absorption may indicate that K features are well-separated in the SAE feature space, despite moderate CMI."
            }
        },

        "robustness_checks": {
            "remove_S_and_K": {
                "n_remaining": int(n_letters - 2),
                "spearman_rho": float(rho_no_sk),
                "spearman_p": float(p_no_sk),
                "bootstrap_ci_95": [ci_lo_no_sk, ci_hi_no_sk],
                "rho_change_from_full": float(rho_no_sk - full_rho)
            },
            "remove_all_influential": {
                "n_removed": len(influential),
                "letters_removed": [r["letter"] for r in influential],
                "n_remaining": int(np.sum(mask_no_inf)) if rho_no_inf is not None else None,
                "spearman_rho": float(rho_no_inf) if rho_no_inf is not None else None,
                "spearman_p": float(p_no_inf) if p_no_inf is not None else None,
                "bootstrap_ci_95": [ci_lo_no_inf, ci_hi_no_inf] if rho_no_inf is not None else None
            }
        },

        "jackknife": {
            "bias": float(jackknife_bias),
            "se": float(jackknife_se),
            "bias_corrected_rho": float(bias_corrected_rho),
            "ci_95": [float(bias_corrected_rho - 1.96 * jackknife_se),
                      float(bias_corrected_rho + 1.96 * jackknife_se)]
        },

        "stability_summary": {
            "verdict": stability_verdict,
            "description": stability_text,
            "same_sign_fraction": float(same_sign_pct / 100),
            "same_sign_count": int(same_sign_count),
            "loo_rho_range": [min_rho, max_rho],
            "loo_rho_mean": float(mean_rho),
            "loo_rho_std": float(std_rho),
            "max_abs_delta_rho": float(np.max(np.abs(rho_changes))),
            "mean_abs_delta_rho": float(np.mean(np.abs(rho_changes)))
        },

        "per_letter_data": {
            letter: {
                "cmi": float(cmi_values[i]),
                "absorption_rate": float(absorption_rates[i]),
                "probe_f1": float(probe_f1_values[i])
            }
            for i, letter in enumerate(letters)
        },

        "visualization_data": {
            "bar_chart_sorted_by_influence": [
                {
                    "letter": r["letter"],
                    "rho_change": r["rho_change"],
                    "is_influential": r["is_influential"]
                }
                for r in loo_results  # already sorted by abs_rho_change
            ]
        }
    }

    # Write output
    output_path = RESULTS_DIR / "leave_one_out_cmi.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    report_progress(TASK_ID, RESULTS_DIR, 5, 5, step=5, total_steps=5,
                    metric={"phase": "complete", "stability": stability_verdict})

    # Mark done
    mark_task_done(TASK_ID, RESULTS_DIR, status="success",
                   summary=f"LOO analysis complete. {len(influential)} influential letters. "
                           f"Stability: {stability_verdict}. Full rho={full_rho:.4f}. "
                           f"Jackknife SE={jackknife_se:.4f}.")

    print(f"\n{'='*60}")
    print("TASK COMPLETE")
    print(f"{'='*60}")

    return output


if __name__ == "__main__":
    main()
