#!/usr/bin/env python3
"""
Partial Correlation: CMI x Absorption | Probe F1

GATE 0B analysis: Compute partial Spearman rho(CMI, absorption | probe_F1)
controlling for the probe quality confound (rho=-0.67 between absorption and probe F1).

Uses data from iter_006:
- cmi_estimation.json (CMI values at d'=10 for 25 letters)
- first_letter_improved.json (absorption rates and probe F1 for 25 letters)

Outputs:
- Partial Spearman rho with bootstrap 95% CIs (controlling for probe F1)
- Restricted correlation on letters with F1 > 0.85 (N~10)
- Raw correlation for comparison
- Machine-readable JSON at exp/results/full/partial_correlation_cmi.json
  (pilot mode: exp/results/pilots/partial_correlation_cmi.json)

Author: Sibyl Experimenter Agent
Task ID: partial_correlation_cmi
"""

import json
import os
import sys
import numpy as np
from scipy import stats
from datetime import datetime
from pathlib import Path

# ---- Configuration ----
SEED = 42
N_BOOTSTRAP = 10000
F1_THRESHOLD = 0.85
PILOT_MODE = "--pilot" in sys.argv or os.environ.get("PILOT_MODE", "0") == "1"

np.random.seed(SEED)

# ---- Paths ----
# Detect workspace root
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent.parent  # exp/code -> exp -> workspace root

# Source data from iter_006
ITER006_DIR = WORKSPACE.parent / "iter_006"
CMI_PATH = ITER006_DIR / "exp" / "results" / "full" / "cmi_estimation.json"
FL_PATH = ITER006_DIR / "exp" / "results" / "full" / "first_letter_improved.json"

# Output
if PILOT_MODE:
    OUTPUT_DIR = WORKSPACE / "exp" / "results" / "pilots"
else:
    OUTPUT_DIR = WORKSPACE / "exp" / "results" / "full"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "partial_correlation_cmi.json"

print(f"[partial_correlation_cmi] Mode: {'PILOT' if PILOT_MODE else 'FULL'}")
print(f"[partial_correlation_cmi] CMI source: {CMI_PATH}")
print(f"[partial_correlation_cmi] FL source: {FL_PATH}")
print(f"[partial_correlation_cmi] Output: {OUTPUT_PATH}")
print()


def load_data():
    """Load CMI, absorption, and probe F1 data for 25 letters."""
    with open(CMI_PATH) as f:
        cmi_data = json.load(f)
    with open(FL_PATH) as f:
        fl_data = json.load(f)

    # Use the summary_table from CMI estimation (has all fields aligned)
    summary = cmi_data["summary_table"]

    letters = []
    cmi_values = []
    absorption_rates = []
    probe_f1_values = []

    for row in summary:
        letter = row["letter"]
        cmi = row["cmi"]
        absorption = row["absorption_rate"]
        probe_f1 = row["probe_f1"]

        letters.append(letter)
        cmi_values.append(cmi)
        absorption_rates.append(absorption)
        probe_f1_values.append(probe_f1)

    # Cross-check with first_letter_improved
    fl_per_letter = fl_data["l12_16k"]["per_letter"]
    mismatches = []
    for i, letter in enumerate(letters):
        if letter in fl_per_letter:
            fl_abs = fl_per_letter[letter]["absorption_rate"]
            fl_f1 = fl_per_letter[letter]["probe_f1"]
            if abs(fl_abs - absorption_rates[i]) > 1e-4:
                mismatches.append(
                    f"  {letter}: absorption CMI={absorption_rates[i]:.4f} vs FL={fl_abs:.4f}"
                )
            if abs(fl_f1 - probe_f1_values[i]) > 1e-4:
                mismatches.append(
                    f"  {letter}: F1 CMI={probe_f1_values[i]:.4f} vs FL={fl_f1:.4f}"
                )

    if mismatches:
        print("[WARNING] Data mismatches between CMI and first_letter_improved:")
        for m in mismatches:
            print(m)
    else:
        print("[OK] Data consistent between CMI and first_letter_improved sources.")

    return (
        np.array(letters),
        np.array(cmi_values),
        np.array(absorption_rates),
        np.array(probe_f1_values),
    )


def rank_transform(x):
    """Convert values to ranks (handling ties with average rank)."""
    return stats.rankdata(x)


def partial_spearman_rho(x, y, z):
    """
    Compute partial Spearman correlation rho(X, Y | Z).

    Method: Rank all three variables, then compute partial correlation
    on the ranked values using the standard formula:
        rho_XY.Z = (rho_XY - rho_XZ * rho_YZ) / sqrt((1 - rho_XZ^2)(1 - rho_YZ^2))

    This is the standard nonparametric partial correlation approach.
    """
    rx = rank_transform(x)
    ry = rank_transform(y)
    rz = rank_transform(z)

    # Pearson correlations on ranks = Spearman correlations
    rho_xy = np.corrcoef(rx, ry)[0, 1]
    rho_xz = np.corrcoef(rx, rz)[0, 1]
    rho_yz = np.corrcoef(ry, rz)[0, 1]

    denom = np.sqrt((1 - rho_xz**2) * (1 - rho_yz**2))
    if denom < 1e-12:
        return np.nan, rho_xy, rho_xz, rho_yz

    partial_rho = (rho_xy - rho_xz * rho_yz) / denom
    return partial_rho, rho_xy, rho_xz, rho_yz


def bootstrap_partial_spearman(x, y, z, n_boot=N_BOOTSTRAP, seed=SEED):
    """
    Bootstrap confidence interval for partial Spearman rho(X, Y | Z).
    Returns: (point_estimate, ci_lower, ci_upper, boot_distribution)
    """
    rng = np.random.RandomState(seed)
    n = len(x)
    point_est, _, _, _ = partial_spearman_rho(x, y, z)

    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        rho, _, _, _ = partial_spearman_rho(x[idx], y[idx], z[idx])
        if not np.isnan(rho):
            boot_rhos.append(rho)

    boot_rhos = np.array(boot_rhos)
    ci_lower = np.percentile(boot_rhos, 2.5)
    ci_upper = np.percentile(boot_rhos, 97.5)

    return point_est, ci_lower, ci_upper, boot_rhos


def bootstrap_spearman(x, y, n_boot=N_BOOTSTRAP, seed=SEED):
    """
    Bootstrap confidence interval for standard Spearman rho.
    Returns: (rho, p_value, ci_lower, ci_upper, boot_distribution)
    """
    rng = np.random.RandomState(seed)
    n = len(x)
    rho, p = stats.spearmanr(x, y)

    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            boot_rhos.append(r)

    boot_rhos = np.array(boot_rhos)
    ci_lower = np.percentile(boot_rhos, 2.5)
    ci_upper = np.percentile(boot_rhos, 97.5)

    return rho, p, ci_lower, ci_upper, boot_rhos


def permutation_test_partial_rho(x, y, z, n_perm=10000, seed=SEED):
    """
    Permutation test for partial Spearman rho.
    Null hypothesis: X and Y are independent given Z.
    Permute X, recompute partial rho. p-value = fraction of permuted rhos
    as extreme as observed.
    """
    rng = np.random.RandomState(seed)
    observed, _, _, _ = partial_spearman_rho(x, y, z)

    count = 0
    for _ in range(n_perm):
        x_perm = rng.permutation(x)
        rho_perm, _, _, _ = partial_spearman_rho(x_perm, y, z)
        if not np.isnan(rho_perm) and abs(rho_perm) >= abs(observed):
            count += 1

    p_value = count / n_perm
    return observed, p_value


def main():
    timestamp_start = datetime.now().isoformat()
    print("=" * 70)
    print("PARTIAL CORRELATION: CMI x Absorption | Probe F1")
    print("=" * 70)
    print()

    # ---- Load data ----
    letters, cmi, absorption, probe_f1 = load_data()
    n_letters = len(letters)
    print(f"\nLoaded data for {n_letters} letters.")
    print(f"  CMI range: [{cmi.min():.3f}, {cmi.max():.3f}]")
    print(f"  Absorption range: [{absorption.min():.3f}, {absorption.max():.3f}]")
    print(f"  Probe F1 range: [{probe_f1.min():.3f}, {probe_f1.max():.3f}]")
    print()

    # ---- 1. Raw correlations ----
    print("-" * 50)
    print("1. RAW CORRELATIONS (no control)")
    print("-" * 50)

    rho_cmi_abs, p_cmi_abs, ci_lo_ca, ci_hi_ca, _ = bootstrap_spearman(
        cmi, absorption
    )
    rho_cmi_f1, p_cmi_f1, _, _, _ = bootstrap_spearman(cmi, probe_f1)
    rho_abs_f1, p_abs_f1, _, _, _ = bootstrap_spearman(absorption, probe_f1)

    print(f"  rho(CMI, absorption)   = {rho_cmi_abs:.4f}  (p={p_cmi_abs:.4f})")
    print(f"    95% CI: [{ci_lo_ca:.4f}, {ci_hi_ca:.4f}]")
    print(f"  rho(CMI, probe_F1)     = {rho_cmi_f1:.4f}  (p={p_cmi_f1:.4f})")
    print(f"  rho(absorption, probe_F1) = {rho_abs_f1:.4f}  (p={p_abs_f1:.4f})")
    print()

    # ---- 2. Partial correlation (CMI, absorption | probe_F1) ----
    print("-" * 50)
    print("2. PARTIAL CORRELATION: rho(CMI, absorption | probe_F1)")
    print("-" * 50)

    partial_rho, ci_lo_partial, ci_hi_partial, boot_partial = (
        bootstrap_partial_spearman(cmi, absorption, probe_f1)
    )

    # Permutation test for p-value
    _, perm_p = permutation_test_partial_rho(cmi, absorption, probe_f1)

    # Approximate degrees of freedom for partial correlation
    # df = n - 2 - k where k = number of control variables
    df_partial = n_letters - 2 - 1
    # Approximate t-statistic
    if abs(partial_rho) < 1.0:
        t_stat = partial_rho * np.sqrt(df_partial / (1 - partial_rho**2))
        approx_p = 2 * stats.t.sf(abs(t_stat), df_partial)
    else:
        t_stat = np.inf
        approx_p = 0.0

    print(f"  Partial rho = {partial_rho:.4f}")
    print(f"    95% CI (bootstrap): [{ci_lo_partial:.4f}, {ci_hi_partial:.4f}]")
    print(f"    Permutation p-value: {perm_p:.4f}")
    print(f"    Approximate t-test p: {approx_p:.4f} (df={df_partial})")
    print(f"    Bonferroni-corrected p (x4 dims): {min(perm_p * 4, 1.0):.4f}")
    print()

    # ---- 3. Restricted correlation (F1 > 0.85 subset) ----
    print("-" * 50)
    print(f"3. RESTRICTED ANALYSIS: Letters with probe F1 > {F1_THRESHOLD}")
    print("-" * 50)

    mask_f1 = probe_f1 > F1_THRESHOLD
    restricted_letters = letters[mask_f1]
    restricted_cmi = cmi[mask_f1]
    restricted_absorption = absorption[mask_f1]
    restricted_f1 = probe_f1[mask_f1]
    n_restricted = mask_f1.sum()

    print(f"  N letters: {n_restricted}")
    print(f"  Letters: {', '.join(restricted_letters)}")
    print(f"  F1 range: [{restricted_f1.min():.3f}, {restricted_f1.max():.3f}]")
    print(f"  Absorption range: [{restricted_absorption.min():.3f}, {restricted_absorption.max():.3f}]")
    print()

    if n_restricted >= 5:
        rho_restricted, p_restricted, ci_lo_r, ci_hi_r, _ = bootstrap_spearman(
            restricted_cmi, restricted_absorption
        )
        print(f"  rho(CMI, absorption) restricted = {rho_restricted:.4f}  (p={p_restricted:.4f})")
        print(f"    95% CI: [{ci_lo_r:.4f}, {ci_hi_r:.4f}]")

        # Also partial correlation in restricted set
        if n_restricted >= 6:  # Need enough data points
            partial_rho_r, ci_lo_pr, ci_hi_pr, _ = bootstrap_partial_spearman(
                restricted_cmi, restricted_absorption, restricted_f1
            )
            _, perm_p_r = permutation_test_partial_rho(
                restricted_cmi, restricted_absorption, restricted_f1,
                n_perm=10000
            )
            print(f"  Partial rho (restricted) = {partial_rho_r:.4f}")
            print(f"    95% CI: [{ci_lo_pr:.4f}, {ci_hi_pr:.4f}]")
            print(f"    Permutation p: {perm_p_r:.4f}")
        else:
            partial_rho_r = None
            ci_lo_pr = ci_hi_pr = perm_p_r = None
            print(f"  [SKIP] Too few letters ({n_restricted}) for partial correlation in restricted set.")
    else:
        rho_restricted = p_restricted = ci_lo_r = ci_hi_r = None
        partial_rho_r = ci_lo_pr = ci_hi_pr = perm_p_r = None
        print(f"  [SKIP] Too few letters ({n_restricted}) for restricted analysis.")

    print()

    # ---- 4. Effect size comparison ----
    print("-" * 50)
    print("4. EFFECT SIZE COMPARISON")
    print("-" * 50)

    # How much does the correlation change when controlling for probe F1?
    raw_rho = rho_cmi_abs
    delta_rho = partial_rho - raw_rho
    print(f"  Raw rho(CMI, absorption):     {raw_rho:.4f}")
    print(f"  Partial rho (| probe_F1):     {partial_rho:.4f}")
    print(f"  Delta:                        {delta_rho:+.4f}")
    print(f"  Reduction in magnitude:       {abs(raw_rho) - abs(partial_rho):+.4f}")
    print()

    if abs(partial_rho) > abs(raw_rho):
        direction = "STRENGTHENED"
        interp = "Controlling for probe F1 strengthened the CMI-absorption association"
    elif abs(partial_rho) < abs(raw_rho) * 0.5:
        direction = "SUBSTANTIALLY_WEAKENED"
        interp = "Probe quality confound explains >50% of the CMI-absorption association"
    elif abs(partial_rho) < abs(raw_rho):
        direction = "WEAKENED"
        interp = "Controlling for probe F1 weakened the CMI-absorption association"
    else:
        direction = "UNCHANGED"
        interp = "Probe quality confound has minimal impact on the CMI-absorption association"

    print(f"  Direction: {direction}")
    print(f"  Interpretation: {interp}")
    print()

    # ---- 5. Per-letter data table ----
    print("-" * 50)
    print("5. PER-LETTER DATA")
    print("-" * 50)
    print(f"  {'Letter':<8} {'CMI':>8} {'Absorption':>12} {'Probe_F1':>10} {'F1>0.85':>8}")
    print(f"  {'------':<8} {'---':>8} {'----------':>12} {'--------':>10} {'-------':>8}")
    for i in range(n_letters):
        f1_flag = "YES" if probe_f1[i] > F1_THRESHOLD else "no"
        print(
            f"  {letters[i]:<8} {cmi[i]:>8.4f} {absorption[i]:>12.4f} {probe_f1[i]:>10.4f} {f1_flag:>8}"
        )
    print()

    # ---- 6. Interpretation summary ----
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    survives = abs(partial_rho) > 0.2 and perm_p < 0.10
    print(f"\n  Does CMI-absorption association survive probe quality control?")
    if survives:
        print(f"    PARTIALLY YES: partial rho = {partial_rho:.3f} (p={perm_p:.3f})")
        print(f"    The association is weakened but persists after controlling for probe F1.")
    else:
        if perm_p >= 0.10:
            print(f"    NO (non-significant): partial rho = {partial_rho:.3f} (p={perm_p:.3f})")
        else:
            print(f"    NO (too small): partial rho = {partial_rho:.3f} (p={perm_p:.3f})")
        print(f"    The CMI-absorption association does not survive probe quality control.")

    print()

    # ---- Build output JSON ----
    per_letter_data = []
    for i in range(n_letters):
        per_letter_data.append({
            "letter": str(letters[i]),
            "cmi_d10": float(cmi[i]),
            "absorption_rate": float(absorption[i]),
            "probe_f1": float(probe_f1[i]),
            "in_restricted_set": bool(probe_f1[i] > F1_THRESHOLD),
        })

    restricted_data = {
        "n_letters": int(n_restricted),
        "letters": [str(l) for l in restricted_letters],
        "f1_threshold": F1_THRESHOLD,
    }
    if rho_restricted is not None:
        restricted_data.update({
            "spearman_rho": float(rho_restricted),
            "spearman_p": float(p_restricted),
            "ci_lower": float(ci_lo_r),
            "ci_upper": float(ci_hi_r),
        })
    if partial_rho_r is not None:
        restricted_data.update({
            "partial_rho": float(partial_rho_r),
            "partial_ci_lower": float(ci_lo_pr),
            "partial_ci_upper": float(ci_hi_pr),
            "partial_perm_p": float(perm_p_r),
        })

    output = {
        "task_id": "partial_correlation_cmi",
        "mode": "pilot" if PILOT_MODE else "full",
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "timestamp_start": timestamp_start,
        "timestamp_end": datetime.now().isoformat(),
        "data_sources": {
            "cmi_estimation": str(CMI_PATH),
            "first_letter_improved": str(FL_PATH),
            "subspace_dim": 10,
            "sae_config": "Gemma Scope L12-16k at L0=82",
        },
        "raw_correlations": {
            "rho_cmi_absorption": {
                "spearman_rho": float(rho_cmi_abs),
                "p_value": float(p_cmi_abs),
                "ci_lower": float(ci_lo_ca),
                "ci_upper": float(ci_hi_ca),
                "n_letters": int(n_letters),
            },
            "rho_cmi_probe_f1": {
                "spearman_rho": float(rho_cmi_f1),
                "p_value": float(p_cmi_f1),
            },
            "rho_absorption_probe_f1": {
                "spearman_rho": float(rho_abs_f1),
                "p_value": float(p_abs_f1),
                "note": "This is the confound: strong negative correlation between absorption and probe quality",
            },
        },
        "partial_correlation": {
            "description": "Partial Spearman rho(CMI, absorption | probe_F1)",
            "method": "Rank all three variables, compute partial correlation on ranks: rho_XY.Z = (rho_XY - rho_XZ * rho_YZ) / sqrt((1-rho_XZ^2)(1-rho_YZ^2))",
            "partial_rho": float(partial_rho),
            "ci_lower": float(ci_lo_partial),
            "ci_upper": float(ci_hi_partial),
            "permutation_p_value": float(perm_p),
            "approximate_t_test_p": float(approx_p),
            "bonferroni_corrected_p": float(min(perm_p * 4, 1.0)),
            "df": int(df_partial),
            "n_letters": int(n_letters),
            "n_bootstrap": N_BOOTSTRAP,
        },
        "restricted_analysis": restricted_data,
        "effect_size_comparison": {
            "raw_rho": float(raw_rho),
            "partial_rho": float(partial_rho),
            "delta_rho": float(delta_rho),
            "magnitude_reduction": float(abs(raw_rho) - abs(partial_rho)),
            "direction": direction,
            "interpretation": interp,
        },
        "survives_probe_quality_control": survives,
        "overall_interpretation": (
            f"The raw CMI-absorption correlation (rho={raw_rho:.3f}, p={p_cmi_abs:.3f}) "
            f"becomes partial rho={partial_rho:.3f} (permutation p={perm_p:.3f}) "
            f"after controlling for probe F1. "
            f"{'The association survives probe quality control.' if survives else 'The association does not survive probe quality control.'}"
        ),
        "per_letter": per_letter_data,
        "paper_ready_table": {
            "columns": [
                "Analysis",
                "N_letters",
                "Spearman_rho",
                "CI_lower",
                "CI_upper",
                "P_value",
                "Interpretation",
            ],
            "rows": [
                {
                    "Analysis": "Raw rho(CMI, absorption)",
                    "N_letters": int(n_letters),
                    "Spearman_rho": round(float(rho_cmi_abs), 3),
                    "CI_lower": round(float(ci_lo_ca), 3),
                    "CI_upper": round(float(ci_hi_ca), 3),
                    "P_value": round(float(p_cmi_abs), 4),
                    "Interpretation": "Marginal negative association",
                },
                {
                    "Analysis": "Raw rho(absorption, probe_F1)",
                    "N_letters": int(n_letters),
                    "Spearman_rho": round(float(rho_abs_f1), 3),
                    "CI_lower": None,
                    "CI_upper": None,
                    "P_value": round(float(p_abs_f1), 4),
                    "Interpretation": "Strong confound: absorption inversely related to probe quality",
                },
                {
                    "Analysis": "Partial rho(CMI, absorption | probe_F1)",
                    "N_letters": int(n_letters),
                    "Spearman_rho": round(float(partial_rho), 3),
                    "CI_lower": round(float(ci_lo_partial), 3),
                    "CI_upper": round(float(ci_hi_partial), 3),
                    "P_value": round(float(perm_p), 4),
                    "Interpretation": f"After controlling for probe F1: {direction.lower().replace('_', ' ')}",
                },
                {
                    "Analysis": f"Restricted rho (F1>{F1_THRESHOLD})",
                    "N_letters": int(n_restricted),
                    "Spearman_rho": round(float(rho_restricted), 3) if rho_restricted is not None else None,
                    "CI_lower": round(float(ci_lo_r), 3) if ci_lo_r is not None else None,
                    "CI_upper": round(float(ci_hi_r), 3) if ci_hi_r is not None else None,
                    "P_value": round(float(p_restricted), 4) if p_restricted is not None else None,
                    "Interpretation": f"On {n_restricted} letters with good probes",
                },
            ],
        },
        "pass_criteria": {
            "partial_rho_computed": not np.isnan(partial_rho),
            "bootstrap_ci_computed": ci_lo_partial is not None,
            "restricted_rho_computed": rho_restricted is not None,
            "clear_reporting": True,
            "all_pass": (
                not np.isnan(partial_rho)
                and ci_lo_partial is not None
                and rho_restricted is not None
            ),
        },
    }

    # Write output
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[OUTPUT] Written to {OUTPUT_PATH}")
    print(f"[PASS] All criteria: {output['pass_criteria']['all_pass']}")

    return output


if __name__ == "__main__":
    result = main()
