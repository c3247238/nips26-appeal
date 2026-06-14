#!/usr/bin/env python3
"""
EC50 Dose-Response Analysis
Compute dose-response curves and EC50 per feature from existing steering data.
Test whether EC50 correlates with absorption rate (H4).
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/exp/results/full")
OUTPUT_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/exp/results/full")

def load_data():
    """Load steering results and absorption rates."""
    with open(RESULTS_DIR / "full_steering_probing_gpt2_l4_results.json") as f:
        steering_l4 = json.load(f)
    with open(RESULTS_DIR / "full_steering_probing_gpt2_l8_results.json") as f:
        steering_l8 = json.load(f)
    with open(RESULTS_DIR / "correlation_report_full.json") as f:
        corr = json.load(f)

    return steering_l4, steering_l8, corr

def compute_ec50(strength_results, target_success=0.5):
    """
    Compute EC50 by interpolating the dose-response curve.
    Uses linear interpolation between the two bracketing points.
    Returns EC50 or None if curve doesn't cross target.
    """
    strengths = [r["strength"] for r in strength_results]
    success_rates = [r["success_rate"] for r in strength_results]

    # Check if curve crosses target
    max_success = max(success_rates)
    if max_success < target_success:
        return None  # Never reaches target

    # Find the two points that bracket target_success
    for i in range(len(success_rates) - 1):
        s_low, s_high = success_rates[i], success_rates[i + 1]
        if s_low <= target_success <= s_high or s_high <= target_success <= s_low:
            # Linear interpolation
            st_low, st_high = strengths[i], strengths[i + 1]
            if s_high == s_low:
                return st_low
            frac = (target_success - s_low) / (s_high - s_low)
            ec50 = st_low + frac * (st_high - st_low)
            return ec50

    # If target is below first point, extrapolate from first two
    if success_rates[0] >= target_success:
        if len(success_rates) > 1 and success_rates[1] != success_rates[0]:
            frac = (target_success - success_rates[0]) / (success_rates[1] - success_rates[0])
            return strengths[0] + frac * (strengths[1] - strengths[0])
        return strengths[0]

    return None

def fit_hill_equation(strengths, success_rates):
    """
    Fit Hill equation: S(s) = S_max * s^n / (EC50^n + s^n)
    Returns fitted parameters and R^2.
    """
    strengths = np.array(strengths, dtype=float)
    success_rates = np.array(success_rates, dtype=float)

    # Initial guesses
    s_max = max(success_rates) if max(success_rates) > 0 else 1.0
    ec50_guess = np.median(strengths)
    n_guess = 1.0

    from scipy.optimize import curve_fit

    def hill(s, s_max, ec50, n):
        return s_max * s**n / (ec50**n + s**n)

    try:
        popt, _ = curve_fit(hill, strengths, success_rates,
                           p0=[s_max, ec50_guess, n_guess],
                           bounds=([0, 0.1, 0.1], [2.0, 100, 5.0]),
                           maxfev=10000)
        s_max_fit, ec50_fit, n_fit = popt

        # Compute R^2
        predicted = hill(strengths, *popt)
        ss_res = np.sum((success_rates - predicted) ** 2)
        ss_tot = np.sum((success_rates - np.mean(success_rates)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {"s_max": s_max_fit, "ec50": ec50_fit, "n": n_fit, "R2": r2}
    except Exception as e:
        return {"error": str(e)}

def analyze_layer(steering_data, absorption_rates, layer_name):
    """Analyze EC50 for all features in a layer."""
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    results = {}

    for letter in letters:
        if letter not in steering_data["steering_results"]:
            continue

        strength_results = steering_data["steering_results"][letter]["strength_results"]
        strengths = [r["strength"] for r in strength_results]
        success_rates = [r["success_rate"] for r in strength_results]

        # Compute EC50 at 50% success
        ec50 = compute_ec50(strength_results, target_success=0.5)

        # Also compute EC25 and EC75
        ec25 = compute_ec50(strength_results, target_success=0.25)
        ec75 = compute_ec50(strength_results, target_success=0.75)

        # Fit Hill equation
        hill_fit = fit_hill_equation(strengths, success_rates)

        absorption = absorption_rates.get(letter, 0.0)

        results[letter] = {
            "absorption_rate": absorption,
            "ec50": ec50,
            "ec25": ec25,
            "ec75": ec75,
            "hill_fit": hill_fit,
            "max_success": max(success_rates),
            "strengths": strengths,
            "success_rates": success_rates,
        }

    # Correlation analysis
    valid_features = [l for l in letters if l in results and results[l]["ec50"] is not None]

    if len(valid_features) >= 5:
        absorptions = [results[l]["absorption_rate"] for l in valid_features]
        ec50s = [results[l]["ec50"] for l in valid_features]

        r_pearson, p_pearson = stats.pearsonr(absorptions, ec50s)
        r_spearman, p_spearman = stats.spearmanr(absorptions, ec50s)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(absorptions, ec50s)

        correlation = {
            "n": len(valid_features),
            "pearson_r": r_pearson,
            "pearson_p": p_pearson,
            "spearman_rho": r_spearman,
            "spearman_p": p_spearman,
            "R2": r_value ** 2,
            "slope": slope,
            "intercept": intercept,
            "std_err": std_err,
        }
    else:
        correlation = {"n": len(valid_features), "error": "Too few valid EC50 values"}

    return {
        "feature_results": results,
        "correlation": correlation,
        "valid_features": valid_features,
    }

def main():
    steering_l4, steering_l8, corr = load_data()

    absorption_l4 = corr["layer_results"]["layer_4"]["absorption_rates"]
    absorption_l8 = corr["layer_results"]["layer_8"]["absorption_rates"]

    print("=" * 70)
    print("EC50 DOSE-RESPONSE ANALYSIS")
    print("=" * 70)

    # Layer 4
    print("\n--- Layer 4 ---")
    results_l4 = analyze_layer(steering_l4, absorption_l4, "layer_4")

    print(f"\nValid features with computable EC50: {len(results_l4['valid_features'])}/26")
    print(f"Features: {results_l4['valid_features']}")

    print("\nPer-feature EC50 and absorption:")
    for letter in sorted(results_l4["feature_results"].keys()):
        r = results_l4["feature_results"][letter]
        print(f"  {letter}: A={r['absorption_rate']:.4f}, EC50={r['ec50']}, "
              f"max_success={r['max_success']:.2f}")

    corr_l4 = results_l4["correlation"]
    if "error" not in corr_l4:
        print(f"\nCorrelation (absorption vs EC50):")
        print(f"  Pearson r = {corr_l4['pearson_r']:.4f}, p = {corr_l4['pearson_p']:.4f}")
        print(f"  Spearman rho = {corr_l4['spearman_rho']:.4f}, p = {corr_l4['spearman_p']:.4f}")
        print(f"  R^2 = {corr_l4['R2']:.4f}")
        print(f"  Slope = {corr_l4['slope']:.4f}")

    # Layer 8
    print("\n--- Layer 8 ---")
    results_l8 = analyze_layer(steering_l8, absorption_l8, "layer_8")

    print(f"\nValid features with computable EC50: {len(results_l8['valid_features'])}/26")
    print(f"Features: {results_l8['valid_features']}")

    print("\nPer-feature EC50 and absorption:")
    for letter in sorted(results_l8["feature_results"].keys()):
        r = results_l8["feature_results"][letter]
        print(f"  {letter}: A={r['absorption_rate']:.4f}, EC50={r['ec50']}, "
              f"max_success={r['max_success']:.2f}")

    corr_l8 = results_l8["correlation"]
    if "error" not in corr_l8:
        print(f"\nCorrelation (absorption vs EC50):")
        print(f"  Pearson r = {corr_l8['pearson_r']:.4f}, p = {corr_l8['pearson_p']:.4f}")
        print(f"  Spearman rho = {corr_l8['spearman_rho']:.4f}, p = {corr_l8['spearman_p']:.4f}")
        print(f"  R^2 = {corr_l8['R2']:.4f}")
        print(f"  Slope = {corr_l8['slope']:.4f}")

    # H4 Test: High-absorption features have higher EC50
    print("\n--- H4 Test: Absorption affects efficiency (higher EC50) ---")

    for layer_name, results, absorption in [
        ("Layer 4", results_l4, absorption_l4),
        ("Layer 8", results_l8, absorption_l8),
    ]:
        print(f"\n{layer_name}:")
        high_abs = []
        low_abs = []

        for letter in results["feature_results"]:
            r = results["feature_results"][letter]
            if r["ec50"] is not None:
                if r["absorption_rate"] > 0.05:
                    high_abs.append(r["ec50"])
                else:
                    low_abs.append(r["ec50"])

        if high_abs and low_abs:
            t_stat, p_val = stats.ttest_ind(high_abs, low_abs)
            print(f"  High-absorption EC50: mean={np.mean(high_abs):.2f}, n={len(high_abs)}")
            print(f"  Low-absorption EC50: mean={np.mean(low_abs):.2f}, n={len(low_abs)}")
            print(f"  t-test: t={t_stat:.4f}, p={p_val:.4f}")
            if p_val < 0.05:
                print(f"  -> H4 SUPPORTED: High-absorption features require significantly higher steering strength")
            else:
                print(f"  -> H4 NOT SUPPORTED: No significant difference in EC50")
        else:
            print(f"  Insufficient data for t-test")

    # Save results
    output = {
        "layer_4": {
            "feature_results": {
                k: {kk: vv for kk, vv in v.items() if kk not in ("strengths", "success_rates")}
                for k, v in results_l4["feature_results"].items()
            },
            "correlation": results_l4["correlation"],
            "valid_features": results_l4["valid_features"],
        },
        "layer_8": {
            "feature_results": {
                k: {kk: vv for kk, vv in v.items() if kk not in ("strengths", "success_rates")}
                for k, v in results_l8["feature_results"].items()
            },
            "correlation": results_l8["correlation"],
            "valid_features": results_l8["valid_features"],
        },
    }

    output_path = OUTPUT_DIR / "ec50_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to {output_path}")

    # Also generate markdown report
    md = generate_markdown_report(results_l4, results_l8)
    md_path = OUTPUT_DIR / "ec50_analysis.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Report saved to {md_path}")

def generate_markdown_report(results_l4, results_l8):
    """Generate markdown report."""
    lines = [
        "# EC50 Dose-Response Analysis Report",
        "",
        "## Summary",
        "",
        "This analysis computes the EC50 (median effective steering strength) for each first-letter feature",
        "and tests whether EC50 correlates with absorption rate (H4).",
        "",
        "## Method",
        "",
        "- EC50 is computed by linear interpolation between the two dose points that bracket 50% success rate.",
        "- Hill equation fitting: S(s) = S_max * s^n / (EC50^n + s^n)",
        "- Correlation tested with Pearson and Spearman methods",
        "",
        "## Results",
        "",
    ]

    for layer_name, results in [("Layer 4", results_l4), ("Layer 8", results_l8)]:
        lines.extend([
            f"### {layer_name}",
            "",
            f"**Valid features with EC50:** {len(results['valid_features'])}/26",
            "",
            "| Feature | Absorption | EC50 | EC25 | EC75 | Max Success |",
            "|---------|-----------|------|------|------|-------------|",
        ])

        for letter in sorted(results["feature_results"].keys()):
            r = results["feature_results"][letter]
            ec50_str = f"{r['ec50']:.2f}" if r['ec50'] is not None else "N/A"
            ec25_str = f"{r['ec25']:.2f}" if r['ec25'] is not None else "N/A"
            ec75_str = f"{r['ec75']:.2f}" if r['ec75'] is not None else "N/A"
            lines.append(f"| {letter} | {r['absorption_rate']:.4f} | {ec50_str} | {ec25_str} | {ec75_str} | {r['max_success']:.2f} |")

        lines.append("")
        corr = results["correlation"]
        if "error" not in corr:
            lines.extend([
                "**Correlation (absorption vs EC50):**",
                f"- Pearson r = {corr['pearson_r']:.4f}, p = {corr['pearson_p']:.4f}",
                f"- Spearman rho = {corr['spearman_rho']:.4f}, p = {corr['spearman_p']:.4f}",
                f"- R^2 = {corr['R2']:.4f}",
                f"- Slope = {corr['slope']:.4f}",
                "",
            ])

    lines.extend([
        "## H4 Test Results",
        "",
        "H4: High-absorption features have higher EC50 (require more steering strength).",
        "",
        "See JSON output for detailed t-test results.",
        "",
        "## Conclusion",
        "",
    ])

    return "\n".join(lines)

if __name__ == "__main__":
    main()
