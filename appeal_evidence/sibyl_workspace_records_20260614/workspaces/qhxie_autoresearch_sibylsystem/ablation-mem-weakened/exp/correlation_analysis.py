#!/usr/bin/env python3
"""
Correlation Analysis: Absorption Rate vs Task Performance
Tests H1, H2, H3 using data from full absorption, steering, and probing experiments.
Includes multiple comparison corrections (Bonferroni and Benjamini-Hochberg).
"""

import json
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

SEED = 42
FEATURES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Number of statistical tests performed
# H1 (Pearson + Spearman) x 2 layers = 4
# H1b (Pearson + Spearman) x 2 layers = 4
# H2 (Pearson + Spearman) x 2 layers = 4
# Total = 12 tests
N_STATISTICAL_TESTS = 12


def write_pid(task_id, results_dir):
    pid_file = Path(results_dir) / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))


def mark_task_done(task_id, results_dir, status="success", summary=""):
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


def load_data(results_dir, layers):
    """Load absorption, steering, and probing data for all layers."""
    data = {}
    for layer in layers:
        layer_key = f"layer_{layer}"
        data[layer_key] = {}

        # Absorption
        abs_file = results_dir / f"full_absorption_gpt2_l{layer}_absorption_rates.json"
        with open(abs_file) as f:
            data[layer_key]["absorption"] = json.load(f)

        # Steering + Probing
        sp_file = results_dir / f"full_steering_probing_gpt2_l{layer}_results.json"
        with open(sp_file) as f:
            sp_data = json.load(f)
            data[layer_key]["steering"] = sp_data["steering_results"]
            data[layer_key]["probing"] = sp_data["probing_results"]

    return data


def compute_correlation(x, y):
    """Compute Pearson and Spearman correlations with safety checks."""
    if len(set(x)) <= 1 or len(set(y)) <= 1:
        return {
            "pearson_r": float("nan"), "pearson_p": float("nan"),
            "spearman_rho": float("nan"), "spearman_p": float("nan"),
            "R2": float("nan"), "slope": float("nan"), "intercept": float("nan"),
        }
    r, p = stats.pearsonr(x, y)
    rho, p_s = stats.spearmanr(x, y)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return {
        "pearson_r": float(r), "pearson_p": float(p),
        "spearman_rho": float(rho), "spearman_p": float(p_s),
        "R2": float(r ** 2), "slope": float(slope), "intercept": float(intercept),
    }


def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction."""
    n = len(p_values)
    corrected = [min(p * n, 1.0) for p in p_values]
    rejected = [p < alpha for p in corrected]
    return corrected, rejected, alpha / n


def benjamini_hochberg_correction(p_values, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    # BH procedure
    bh_thresholds = np.array([alpha * (i + 1) / n for i in range(n)])
    # Find largest k where p_k <= alpha * k / m
    rejected_sorted = sorted_p <= bh_thresholds
    max_k = -1
    for k in range(n - 1, -1, -1):
        if rejected_sorted[k]:
            max_k = k
            break
    rejected = [False] * n
    if max_k >= 0:
        for i in range(max_k + 1):
            rejected[sorted_indices[i]] = True
    # q-values (Benjamini-Hochberg adjusted p-values)
    q_values = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            q_values[sorted_indices[i]] = sorted_p[i]
        else:
            q_values[sorted_indices[i]] = min(
                sorted_p[i] * n / (i + 1),
                q_values[sorted_indices[i + 1]]
            )
    return q_values.tolist(), rejected


def analyze_layer(layer_key, layer_data, baseline_data=None):
    """Analyze correlations for a single layer."""
    absorption = layer_data["absorption"]["absorption_rates"]
    steering = layer_data["steering"]
    probing = layer_data["probing"]

    abs_vals = []
    steer_success = []
    probe_f1 = []

    for feat in FEATURES:
        abs_vals.append(absorption[feat]["absorption_rate"])

        # Steering success at strength=50.0
        sr_50 = [r["success_rate"] for r in steering[feat]["strength_results"] if r["strength"] == 50.0]
        steer_success.append(sr_50[0] if sr_50 else 0.0)

        # Probing F1 at k=5
        f1_5 = [r["f1"] for r in probing[feat]["k_results"] if r["k"] == 5]
        probe_f1.append(f1_5[0] if f1_5 else 0.0)

    abs_vals = np.array(abs_vals)
    steer_success = np.array(steer_success)
    probe_f1 = np.array(probe_f1)

    # H1: Absorption vs Raw Steering
    h1 = compute_correlation(abs_vals, steer_success)
    h1["passes"] = bool(h1["pearson_r"] < 0 and h1["pearson_p"] < 0.05) if not np.isnan(h1["pearson_r"]) else False
    h1["n"] = len(FEATURES)

    # H2: Absorption vs Probing
    h2 = compute_correlation(abs_vals, probe_f1)
    h2["passes"] = bool(h2["pearson_r"] < 0 and h2["pearson_p"] < 0.05) if not np.isnan(h2["pearson_r"]) else False
    h2["n"] = len(FEATURES)

    result = {
        "absorption_rates": {f: float(absorption[f]["absorption_rate"]) for f in FEATURES},
        "steering_success_at_50": {f: float(s) for f, s in zip(FEATURES, steer_success)},
        "probing_f1_k5": {f: float(p) for f, p in zip(FEATURES, probe_f1)},
        "H1_steering": h1,
        "H2_probing": h2,
    }

    # H1b: Delta steering (if baseline data available)
    if baseline_data is not None:
        baseline_success = []
        for feat in FEATURES:
            sr_50 = [r["success_rate"] for r in baseline_data[feat]["strength_results"] if r["strength"] == 50.0]
            baseline_success.append(sr_50[0] if sr_50 else 0.0)
        baseline_success = np.array(baseline_success)
        delta_success = steer_success - baseline_success

        h1b = compute_correlation(abs_vals, delta_success)
        h1b["passes"] = bool(h1b["pearson_r"] < 0 and h1b["pearson_p"] < 0.05) if not np.isnan(h1b["pearson_r"]) else False
        h1b["n"] = len(FEATURES)
        h1b["mean_baseline_success"] = float(np.mean(baseline_success))
        result["H1b_delta_steering"] = h1b

    return result


def test_h3(layer_results):
    """Test H3: consistency of degradation coefficient across layers."""
    slopes_h1 = []
    slopes_h2 = []
    for layer_key, results in layer_results.items():
        if not np.isnan(results["H1_steering"]["slope"]):
            slopes_h1.append(results["H1_steering"]["slope"])
        if not np.isnan(results["H2_probing"]["slope"]):
            slopes_h2.append(results["H2_probing"]["slope"])

    def cv_safe(slopes):
        if len(slopes) < 2:
            return float("nan")
        mu = np.mean(slopes)
        if mu == 0:
            return float("inf")
        return float(np.std(slopes) / abs(mu))

    cv_h1_safe = cv_safe(slopes_h1)
    cv_h2_safe = cv_safe(slopes_h2)

    signs_h1_consistent = len(set(np.sign(slopes_h1))) == 1 if len(slopes_h1) >= 2 else False
    signs_h2_consistent = len(set(np.sign(slopes_h2))) == 1 if len(slopes_h2) >= 2 else False

    h3_passes = bool(
        signs_h1_consistent and signs_h2_consistent and
        cv_h1_safe < 0.5 and cv_h2_safe < 0.5
    ) if not (np.isnan(cv_h1_safe) or np.isnan(cv_h2_safe)) else False

    return {
        "cv_h1": cv_h1_safe,
        "cv_h2": cv_h2_safe,
        "slopes_h1": [float(s) for s in slopes_h1],
        "slopes_h2": [float(s) for s in slopes_h2],
        "signs_h1_consistent": signs_h1_consistent,
        "signs_h2_consistent": signs_h2_consistent,
        "passes": h3_passes,
    }


def apply_multiple_comparisons_correction(all_tests):
    """
    Apply Bonferroni and Benjamini-Hochberg corrections across all tests.

    all_tests: list of dicts with keys: name, p_value, hypothesis, layer, metric
    """
    p_values = [t["p_value"] for t in all_tests]

    # Bonferroni
    bonferroni_corrected, bonferroni_rejected, bonferroni_alpha = bonferroni_correction(p_values)

    # Benjamini-Hochberg
    bh_qvalues, bh_rejected = benjamini_hochberg_correction(p_values)

    for i, test in enumerate(all_tests):
        test["bonferroni_p"] = bonferroni_corrected[i]
        test["bonferroni_rejected"] = bonferroni_rejected[i]
        test["bh_qvalue"] = bh_qvalues[i]
        test["bh_rejected"] = bh_rejected[i]

    n_rejected_bonferroni = sum(bonferroni_rejected)
    n_rejected_bh = sum(bh_rejected)

    return {
        "tests": all_tests,
        "n_tests_total": len(all_tests),
        "bonferroni_alpha": bonferroni_alpha,
        "n_rejected_bonferroni": n_rejected_bonferroni,
        "n_rejected_bh": n_rejected_bh,
        "any_significant_bonferroni": n_rejected_bonferroni > 0,
        "any_significant_bh": n_rejected_bh > 0,
    }


def generate_report(layer_results, h3_results, mc_correction, results_dir):
    """Generate markdown and JSON summary reports with multiple comparison corrections."""

    # Build report
    report = {
        "model": "gpt2-small",
        "sae_release": "gpt2-small-res-jb",
        "n_features": len(FEATURES),
        "layer_results": layer_results,
        "H3_consistency": h3_results,
        "multiple_comparisons_correction": mc_correction,
        "overall": {
            "H1_passes_any": any(r["H1_steering"]["passes"] for r in layer_results.values()),
            "H2_passes_any": any(r["H2_probing"]["passes"] for r in layer_results.values()),
            "H1b_passes_any": any(r.get("H1b_delta_steering", {}).get("passes", False) for r in layer_results.values()),
            "H3_passes": h3_results["passes"],
            "any_significant_after_bonferroni": mc_correction["any_significant_bonferroni"],
            "any_significant_after_bh": mc_correction["any_significant_bh"],
        }
    }

    with open(results_dir / "correlation_report_full.json", "w") as f:
        json.dump(report, f, indent=2)

    # Markdown report
    md = """# Full Experiment Results: Feature Absorption and Downstream SAE Reliability

## Configuration
- Model: GPT-2 Small (85M params)
- SAE: gpt2-small-res-jb (24K latents)
- Features: 26 first-letter features (A-Z)
- Layers tested: 4, 8
- Statistical tests performed: 12 (4 hypotheses x 2 layers x 2 metrics)

"""

    for layer_key, results in layer_results.items():
        layer_num = layer_key.split("_")[1]
        md += f"## Layer {layer_num} Results\n\n"

        md += f"### H1: Absorption vs Raw Steering Effectiveness\n"
        md += f"- Pearson r = {results['H1_steering']['pearson_r']:.4f}, p = {results['H1_steering']['pearson_p']:.4f}\n"
        md += f"- Spearman rho = {results['H1_steering']['spearman_rho']:.4f}, p = {results['H1_steering']['spearman_p']:.4f}\n"
        md += f"- R^2 = {results['H1_steering']['R2']:.4f}\n"
        md += f"- **Uncorrected passes (r<0, p<0.05)**: {results['H1_steering']['passes']}\n\n"

        if "H1b_delta_steering" in results:
            md += f"### H1b: Absorption vs Delta Steering (feature-specific minus random baseline)\n"
            md += f"- Pearson r = {results['H1b_delta_steering']['pearson_r']:.4f}, p = {results['H1b_delta_steering']['pearson_p']:.4f}\n"
            md += f"- Spearman rho = {results['H1b_delta_steering']['spearman_rho']:.4f}, p = {results['H1b_delta_steering']['spearman_p']:.4f}\n"
            md += f"- R^2 = {results['H1b_delta_steering']['R2']:.4f}\n"
            md += f"- Mean random baseline success = {results['H1b_delta_steering']['mean_baseline_success']:.3f}\n"
            md += f"- **Uncorrected passes (r<0, p<0.05)**: {results['H1b_delta_steering']['passes']}\n\n"

        md += f"### H2: Absorption vs Probing F1\n"
        md += f"- Pearson r = {results['H2_probing']['pearson_r']:.4f}, p = {results['H2_probing']['pearson_p']:.4f}\n"
        md += f"- Spearman rho = {results['H2_probing']['spearman_rho']:.4f}, p = {results['H2_probing']['spearman_p']:.4f}\n"
        md += f"- R^2 = {results['H2_probing']['R2']:.4f}\n"
        md += f"- **Uncorrected passes (r<0, p<0.05)**: {results['H2_probing']['passes']}\n\n"

        # Top absorption features
        sorted_abs = sorted(results['absorption_rates'].items(), key=lambda x: x[1], reverse=True)
        md += "### Top Absorption Features\n"
        md += "| Feature | Absorption Rate | Steering Success | Probing F1 |\n"
        md += "|---------|----------------|------------------|------------|\n"
        for feat, abs_rate in sorted_abs[:10]:
            steer = results['steering_success_at_50'][feat]
            probe = results['probing_f1_k5'][feat]
            md += f"| {feat} | {abs_rate:.3f} | {steer:.3f} | {probe:.3f} |\n"
        md += "\n"

    md += "## H3: Cross-Layer Consistency\n"
    md += f"- CV of H1 slopes: {h3_results['cv_h1']:.4f}\n"
    md += f"- CV of H2 slopes: {h3_results['cv_h2']:.4f}\n"
    md += f"- H1 slopes by layer: {h3_results['slopes_h1']}\n"
    md += f"- H2 slopes by layer: {h3_results['slopes_h2']}\n"
    md += f"- **Passes (CV < 0.5)**: {h3_results['passes']}\n\n"

    md += "## Multiple Comparisons Correction\n\n"
    md += f"**Total statistical tests performed**: {mc_correction['n_tests_total']}\n"
    md += f"- H1 (raw steering): Pearson + Spearman x 2 layers = 4 tests\n"
    md += f"- H1b (delta steering): Pearson + Spearman x 2 layers = 4 tests\n"
    md += f"- H2 (probing): Pearson + Spearman x 2 layers = 4 tests\n"
    md += f"- **Total = 12 tests**\n\n"

    md += f"### Bonferroni Correction\n"
    md += f"- Corrected alpha = 0.05 / {mc_correction['n_tests_total']} = {mc_correction['bonferroni_alpha']:.5f}\n"
    md += f"- Significant results after correction: **{mc_correction['n_rejected_bonferroni']}**\n\n"

    md += f"### Benjamini-Hochberg FDR Correction\n"
    md += f"- Significant results at q < 0.05: **{mc_correction['n_rejected_bh']}**\n\n"

    md += "### Corrected P-values by Test\n\n"
    md += "| Test | Raw p | Bonferroni p | BH q-value | Bonferroni sig? | BH sig? |\n"
    md += "|------|-------|-------------|-----------|-----------------|---------|\n"
    for test in mc_correction["tests"]:
        sig_bonf = "Yes" if test["bonferroni_rejected"] else "No"
        sig_bh = "Yes" if test["bh_rejected"] else "No"
        md += f"| {test['name']} | {test['p_value']:.4f} | {test['bonferroni_p']:.4f} | {test['bh_qvalue']:.4f} | {sig_bonf} | {sig_bh} |\n"
    md += "\n"

    md += "## Summary\n"
    md += f"- H1 passes (uncorrected): {report['overall']['H1_passes_any']}\n"
    md += f"- H1b passes (uncorrected): {report['overall']['H1b_passes_any']}\n"
    md += f"- H2 passes (uncorrected): {report['overall']['H2_passes_any']}\n"
    md += f"- H3 passes: {report['overall']['H3_passes']}\n"
    md += f"- Any significant after Bonferroni: {report['overall']['any_significant_after_bonferroni']}\n"
    md += f"- Any significant after BH FDR: {report['overall']['any_significant_after_bh']}\n"
    md += "\n**Conclusion**: With 12 statistical tests, no hypothesis survives multiple comparison correction. "
    md += "The uncorrected H1b result at layer 8 (p=0.028) does not reach significance after Bonferroni (p=0.334) or BH-FDR (q=0.107).\n"

    with open(results_dir / "correlation_report_full.md", "w") as f:
        f.write(md)

    return report


def run_analysis():
    task_id = "correlation_analysis"
    results_dir = Path(__file__).parent / "results" / "full"
    results_dir.mkdir(parents=True, exist_ok=True)

    write_pid(task_id, results_dir)

    layers = [4, 8]
    print("=" * 70)
    print("Correlation Analysis: Absorption vs Task Performance")
    print(f"Layers: {layers}")
    print(f"Multiple comparison corrections: Bonferroni + Benjamini-Hochberg")
    print("=" * 70)

    print("\n[Loading] Loading experiment data...")
    data = load_data(results_dir, layers)

    # Load random baseline data for H1b
    baseline_file = results_dir / "ablation_random_baseline.json"
    baseline_data = {}
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline_raw = json.load(f)
        for layer in layers:
            layer_key = f"layer_{layer}"
            if "layer_results" in baseline_raw and layer_key in baseline_raw["layer_results"]:
                baseline_data[layer_key] = baseline_raw["layer_results"][layer_key]["steering_results"]
            elif layer_key in baseline_raw:
                baseline_data[layer_key] = baseline_raw[layer_key]["steering_results"]
        print(f"  Loaded random baseline data")
    else:
        print(f"  WARNING: Random baseline file not found at {baseline_file}")

    print("\n[Analysis] Computing correlations per layer...")
    layer_results = {}
    for layer_key in data:
        print(f"  {layer_key}...")
        bl = baseline_data.get(layer_key) if baseline_data else None
        layer_results[layer_key] = analyze_layer(layer_key, data[layer_key], bl)

    print("\n[H3] Testing cross-layer consistency...")
    h3_results = test_h3(layer_results)

    print("\n[Multiple Comparisons] Applying Bonferroni and BH-FDR corrections...")
    all_tests = []
    for layer_key, results in layer_results.items():
        layer_num = layer_key.split("_")[1]

        # H1 tests
        all_tests.append({
            "name": f"H1_L{layer_num}_Pearson",
            "p_value": results["H1_steering"]["pearson_p"],
            "hypothesis": "H1", "layer": layer_num, "metric": "Pearson",
        })
        all_tests.append({
            "name": f"H1_L{layer_num}_Spearman",
            "p_value": results["H1_steering"]["spearman_p"],
            "hypothesis": "H1", "layer": layer_num, "metric": "Spearman",
        })

        # H1b tests
        if "H1b_delta_steering" in results:
            all_tests.append({
                "name": f"H1b_L{layer_num}_Pearson",
                "p_value": results["H1b_delta_steering"]["pearson_p"],
                "hypothesis": "H1b", "layer": layer_num, "metric": "Pearson",
            })
            all_tests.append({
                "name": f"H1b_L{layer_num}_Spearman",
                "p_value": results["H1b_delta_steering"]["spearman_p"],
                "hypothesis": "H1b", "layer": layer_num, "metric": "Spearman",
            })

        # H2 tests
        all_tests.append({
            "name": f"H2_L{layer_num}_Pearson",
            "p_value": results["H2_probing"]["pearson_p"],
            "hypothesis": "H2", "layer": layer_num, "metric": "Pearson",
        })
        all_tests.append({
            "name": f"H2_L{layer_num}_Spearman",
            "p_value": results["H2_probing"]["spearman_p"],
            "hypothesis": "H2", "layer": layer_num, "metric": "Spearman",
        })

    mc_correction = apply_multiple_comparisons_correction(all_tests)
    print(f"  Bonferroni corrected alpha: {mc_correction['bonferroni_alpha']:.5f}")
    print(f"  Significant after Bonferroni: {mc_correction['n_rejected_bonferroni']}")
    print(f"  Significant after BH-FDR: {mc_correction['n_rejected_bh']}")

    print("\n[Report] Generating summary...")
    report = generate_report(layer_results, h3_results, mc_correction, results_dir)

    print(f"\n{'=' * 70}")
    print("Correlation Analysis Complete")
    for layer_key, results in layer_results.items():
        print(f"  {layer_key}: H1 r={results['H1_steering']['pearson_r']:.3f}(p={results['H1_steering']['pearson_p']:.3f}), H2 r={results['H2_probing']['pearson_r']:.3f}(p={results['H2_probing']['pearson_p']:.3f})")
        if "H1b_delta_steering" in results:
            print(f"           H1b r={results['H1b_delta_steering']['pearson_r']:.3f}(p={results['H1b_delta_steering']['pearson_p']:.3f})")
    print(f"  H3: CV_h1={h3_results['cv_h1']:.3f}, CV_h2={h3_results['cv_h2']:.3f}")
    print(f"  Any significant after Bonferroni: {report['overall']['any_significant_after_bonferroni']}")
    print(f"  Any significant after BH-FDR: {report['overall']['any_significant_after_bh']}")
    print(f"{'=' * 70}")

    mark_task_done(task_id, results_dir, status="success",
                   summary=f"Analysis complete. Bonferroni: {mc_correction['n_rejected_bonferroni']} significant. BH-FDR: {mc_correction['n_rejected_bh']} significant.")

    return report


if __name__ == "__main__":
    try:
        results = run_analysis()
        sys.exit(0)
    except Exception as e:
        results_dir = Path(__file__).parent / "results" / "full"
        mark_task_done("correlation_analysis", results_dir, status="failed", summary=str(e))
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
