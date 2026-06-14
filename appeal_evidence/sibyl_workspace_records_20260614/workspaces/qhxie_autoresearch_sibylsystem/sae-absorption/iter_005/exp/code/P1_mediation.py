"""
P1_mediation: Formal Mediation Analysis — L0 → Absorption → Quality

Tests whether absorption mediates L0's effect on downstream SAE quality.
Uses Baron & Kenny (1986) framework + Sobel test + bootstrap CIs.

Path model:
  X (log_L0) → M (absorption) → Y (quality)
  Controlling for: log_width, layer

Total effect (c): quality ~ log_L0 + log_width + layer
Direct effect (c'): quality ~ log_L0 + absorption + log_width + layer
Indirect effect (a*b) = c - c'
Proportion mediated = (c - c') / c

Author: Sibyl Experimenter (Pilot mode)
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================
# Configuration
# ============================================================
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption")
DATA_PATH = WORKSPACE / "iter_004/exp/results/full/C3A_saebench_corr.json"
RESULTS_DIR = WORKSPACE / "iter_005/exp/results/pilots"
FULL_RESULTS_DIR = WORKSPACE / "iter_005/exp/results/full"
TASK_ID = "P1_mediation"
SEED = 42
N_BOOTSTRAP = 10000

# Quality metrics to test
QUALITY_METRICS = {
    "sparse_probing_f1": "Sparse Probing F1",
    "scr_score": "SCR",
    "tpp_score": "RAVEL TPP",
    "unlearning_score": "Unlearning",
}

# ============================================================
# PID file for process tracking
# ============================================================
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

start_time = datetime.now()

# ============================================================
# Load and prepare data
# ============================================================
print(f"[{TASK_ID}] Loading data from {DATA_PATH}")
with open(DATA_PATH) as f:
    raw_data = json.load(f)

records = raw_data["raw_records"]
df = pd.DataFrame(records)
print(f"  Total SAEs: {len(df)}")

# Filter to SAEs with known L0
df_l0 = df[df["l0"].notna()].copy()
print(f"  SAEs with known L0: {len(df_l0)}")

# Create log transforms
df_l0["log_l0"] = np.log(df_l0["l0"].astype(float))
df_l0["log_width"] = np.log(df_l0["width_int"].astype(float))

# Rename for clarity
df_l0.rename(columns={"absorption_score": "absorption"}, inplace=True)

print(f"  log_L0 range: [{df_l0['log_l0'].min():.2f}, {df_l0['log_l0'].max():.2f}]")
print(f"  absorption range: [{df_l0['absorption'].min():.3f}, {df_l0['absorption'].max():.3f}]")

# ============================================================
# Mediation Analysis Functions
# ============================================================

def run_ols(y, X_df):
    """Run OLS regression. Returns (coefficients, residuals, R-squared, summary_dict)."""
    import statsmodels.api as sm
    X_with_const = sm.add_constant(X_df)
    model = sm.OLS(y, X_with_const, missing='drop').fit()
    return model


def mediation_analysis(df_data, x_col, m_col, y_col, covariates, metric_label):
    """
    Full mediation analysis for one quality metric.

    X = x_col (log_L0)
    M = m_col (absorption)
    Y = y_col (quality metric)

    Steps:
    1. Path a: M ~ X + covariates  (effect of X on M)
    2. Path b: Y ~ M + X + covariates  (effect of M on Y controlling X)
    3. Path c: Y ~ X + covariates  (total effect)
    4. Path c': Y ~ X + M + covariates  (direct effect)
    5. Indirect = a*b = c - c'
    6. Proportion mediated = indirect/total
    """
    import statsmodels.api as sm

    # Drop rows with missing values for this metric
    cols_needed = [x_col, m_col, y_col] + covariates
    df_valid = df_data[cols_needed].dropna()
    n = len(df_valid)

    if n < 10:
        return {"error": f"Too few valid observations (n={n})", "n": n}

    X_val = df_valid[x_col].values
    M_val = df_valid[m_col].values
    Y_val = df_valid[y_col].values
    cov_vals = df_valid[covariates].values

    # ---- Step 1: Path a (X -> M) ----
    X_a = sm.add_constant(np.column_stack([X_val, cov_vals]))
    model_a = sm.OLS(M_val, X_a).fit()
    a_coef = model_a.params[1]  # coefficient of X
    a_se = model_a.bse[1]
    a_pval = model_a.pvalues[1]

    # ---- Step 2: Path b and c' (Y ~ X + M + covariates) ----
    X_bc = sm.add_constant(np.column_stack([X_val, M_val, cov_vals]))
    model_bc = sm.OLS(Y_val, X_bc).fit()
    c_prime = model_bc.params[1]  # direct effect of X on Y
    c_prime_se = model_bc.bse[1]
    c_prime_pval = model_bc.pvalues[1]
    b_coef = model_bc.params[2]  # effect of M on Y
    b_se = model_bc.bse[2]
    b_pval = model_bc.pvalues[2]

    # ---- Step 3: Path c (Y ~ X + covariates, total effect) ----
    X_c = sm.add_constant(np.column_stack([X_val, cov_vals]))
    model_c = sm.OLS(Y_val, X_c).fit()
    c_total = model_c.params[1]  # total effect
    c_total_se = model_c.bse[1]
    c_total_pval = model_c.pvalues[1]

    # ---- Indirect effect ----
    indirect = a_coef * b_coef
    # Alternative: indirect = c_total - c_prime (should be equivalent)
    indirect_alt = c_total - c_prime

    # ---- Sobel test ----
    sobel_se = np.sqrt(a_coef**2 * b_se**2 + b_coef**2 * a_se**2)
    sobel_z = indirect / sobel_se if sobel_se > 0 else 0.0
    from scipy.stats import norm
    sobel_p = 2 * (1 - norm.cdf(abs(sobel_z)))

    # ---- Proportion mediated ----
    if abs(c_total) > 1e-10:
        prop_mediated = indirect / c_total
    else:
        prop_mediated = np.nan

    # ---- Bootstrap CI for indirect effect ----
    np.random.seed(SEED)
    boot_indirects = []
    boot_props = []

    for _ in range(N_BOOTSTRAP):
        idx = np.random.choice(n, size=n, replace=True)
        X_b = X_val[idx]
        M_b = M_val[idx]
        Y_b = Y_val[idx]
        cov_b = cov_vals[idx]

        try:
            # Path a
            Xa = sm.add_constant(np.column_stack([X_b, cov_b]))
            ma = sm.OLS(M_b, Xa).fit()
            a_boot = ma.params[1]

            # Path b
            Xbc = sm.add_constant(np.column_stack([X_b, M_b, cov_b]))
            mbc = sm.OLS(Y_b, Xbc).fit()
            b_boot = mbc.params[2]

            # Path c
            Xc = sm.add_constant(np.column_stack([X_b, cov_b]))
            mc = sm.OLS(Y_b, Xc).fit()
            c_boot = mc.params[1]

            ind_boot = a_boot * b_boot
            boot_indirects.append(ind_boot)

            if abs(c_boot) > 1e-10:
                boot_props.append(ind_boot / c_boot)

        except Exception:
            continue

    boot_indirects = np.array(boot_indirects)
    boot_props = np.array(boot_props)

    # BCa bootstrap CI for indirect effect
    indirect_ci = np.percentile(boot_indirects, [2.5, 97.5]) if len(boot_indirects) > 0 else [np.nan, np.nan]
    indirect_ci_excludes_zero = bool(
        (indirect_ci[0] > 0 and indirect_ci[1] > 0) or
        (indirect_ci[0] < 0 and indirect_ci[1] < 0)
    )

    # Bootstrap CI for proportion mediated (trim outliers)
    if len(boot_props) > 100:
        # Trim extreme values (prop_mediated can be unstable)
        boot_props_trimmed = boot_props[(boot_props > -5) & (boot_props < 5)]
        prop_ci = np.percentile(boot_props_trimmed, [2.5, 97.5]) if len(boot_props_trimmed) > 0 else [np.nan, np.nan]
    else:
        prop_ci = [np.nan, np.nan]

    # ---- Baron & Kenny criteria ----
    baron_kenny = {
        "step1_x_predicts_m": a_pval < 0.05,
        "step1_a_coef": float(a_coef),
        "step1_a_pval": float(a_pval),
        "step2_m_predicts_y_controlling_x": b_pval < 0.05,
        "step2_b_coef": float(b_coef),
        "step2_b_pval": float(b_pval),
        "step3_x_predicts_y": c_total_pval < 0.05,
        "step3_c_total": float(c_total),
        "step3_c_pval": float(c_total_pval),
        "step4_direct_reduced": abs(c_prime) < abs(c_total),
        "step4_c_prime": float(c_prime),
        "step4_c_prime_pval": float(c_prime_pval),
        "full_mediation": abs(c_prime) < abs(c_total) and c_prime_pval > 0.05,
        "partial_mediation": abs(c_prime) < abs(c_total) and c_prime_pval < 0.05,
    }

    # All BK steps met?
    bk_all_met = all([
        baron_kenny["step1_x_predicts_m"],
        baron_kenny["step2_m_predicts_y_controlling_x"],
        baron_kenny["step3_x_predicts_y"],
        baron_kenny["step4_direct_reduced"],
    ])

    result = {
        "metric": y_col,
        "metric_label": metric_label,
        "n": int(n),
        "path_a": {
            "description": "log_L0 -> absorption",
            "coefficient": float(a_coef),
            "se": float(a_se),
            "p_value": float(a_pval),
            "r_squared": float(model_a.rsquared),
        },
        "path_b": {
            "description": "absorption -> quality (controlling log_L0)",
            "coefficient": float(b_coef),
            "se": float(b_se),
            "p_value": float(b_pval),
        },
        "total_effect_c": {
            "description": "log_L0 -> quality (total)",
            "coefficient": float(c_total),
            "se": float(c_total_se),
            "p_value": float(c_total_pval),
            "r_squared": float(model_c.rsquared),
        },
        "direct_effect_c_prime": {
            "description": "log_L0 -> quality (controlling absorption)",
            "coefficient": float(c_prime),
            "se": float(c_prime_se),
            "p_value": float(c_prime_pval),
            "r_squared": float(model_bc.rsquared),
        },
        "indirect_effect": {
            "a_times_b": float(indirect),
            "c_minus_c_prime": float(indirect_alt),
            "consistency_check": abs(indirect - indirect_alt) < 0.001,
        },
        "sobel_test": {
            "z": float(sobel_z),
            "p_value": float(sobel_p),
            "se": float(sobel_se),
        },
        "bootstrap": {
            "n_resamples": N_BOOTSTRAP,
            "n_successful": len(boot_indirects),
            "indirect_ci_95": [float(indirect_ci[0]), float(indirect_ci[1])],
            "indirect_ci_excludes_zero": indirect_ci_excludes_zero,
            "proportion_mediated": float(prop_mediated) if not np.isnan(prop_mediated) else None,
            "proportion_mediated_ci_95": [float(prop_ci[0]), float(prop_ci[1])] if not any(np.isnan(prop_ci)) else None,
        },
        "baron_kenny": baron_kenny,
        "baron_kenny_all_steps_met": bk_all_met,
        "mediation_type": (
            "full" if baron_kenny.get("full_mediation") else
            "partial" if baron_kenny.get("partial_mediation") else
            "none"
        ),
    }

    return result


def standardized_mediation(df_data, x_col, m_col, y_col, covariates, metric_label):
    """
    Run mediation on standardized variables for comparable effect sizes.
    """
    cols_needed = [x_col, m_col, y_col] + covariates
    df_valid = df_data[cols_needed].dropna().copy()

    if len(df_valid) < 10:
        return None

    # Standardize X, M, Y (not covariates—they're controls)
    for col in [x_col, m_col, y_col]:
        mean = df_valid[col].mean()
        std = df_valid[col].std()
        if std > 0:
            df_valid[col] = (df_valid[col] - mean) / std

    result = mediation_analysis(df_valid, x_col, m_col, y_col, covariates, metric_label)
    if "error" in result:
        return result

    result["standardized"] = True
    return result


# ============================================================
# Report progress
# ============================================================
def report_progress(task_id, results_dir, step, total_steps, metric=None):
    progress = results_dir / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": step, "total_epochs": total_steps,
        "step": step, "total_steps": total_steps,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


# ============================================================
# Main execution
# ============================================================
print(f"\n{'='*60}")
print(f"P1_MEDIATION: Formal Mediation Analysis")
print(f"Path: L0 -> Absorption -> Quality")
print(f"{'='*60}\n")

covariates = ["log_width", "layer"]
results = {}
total_metrics = len(QUALITY_METRICS)

for i, (metric_col, metric_label) in enumerate(QUALITY_METRICS.items()):
    print(f"\n--- [{i+1}/{total_metrics}] Mediation: {metric_label} ---")

    report_progress(TASK_ID, RESULTS_DIR, i+1, total_metrics)

    # Unstandardized mediation
    result_unstd = mediation_analysis(
        df_l0, "log_l0", "absorption", metric_col, covariates, metric_label
    )

    if "error" in result_unstd:
        print(f"  ERROR: {result_unstd['error']}")
        results[metric_col] = {"unstandardized": result_unstd, "standardized": None}
        continue

    # Standardized mediation (for cross-metric comparisons)
    result_std = standardized_mediation(
        df_l0, "log_l0", "absorption", metric_col, covariates, metric_label
    )

    # Print summary
    print(f"  n = {result_unstd['n']}")
    print(f"  Path a (L0→absorption):    β={result_unstd['path_a']['coefficient']:.4f}, p={result_unstd['path_a']['p_value']:.4e}")
    print(f"  Path b (absorption→{metric_label}): β={result_unstd['path_b']['coefficient']:.4f}, p={result_unstd['path_b']['p_value']:.4e}")
    print(f"  Total effect (c):          β={result_unstd['total_effect_c']['coefficient']:.4f}, p={result_unstd['total_effect_c']['p_value']:.4e}")
    print(f"  Direct effect (c'):        β={result_unstd['direct_effect_c_prime']['coefficient']:.4f}, p={result_unstd['direct_effect_c_prime']['p_value']:.4e}")
    print(f"  Indirect effect (a*b):     {result_unstd['indirect_effect']['a_times_b']:.4f}")
    print(f"  Sobel test:                z={result_unstd['sobel_test']['z']:.3f}, p={result_unstd['sobel_test']['p_value']:.4e}")

    prop = result_unstd['bootstrap']['proportion_mediated']
    if prop is not None:
        print(f"  Proportion mediated:       {prop:.3f} ({prop*100:.1f}%)")
    else:
        print(f"  Proportion mediated:       N/A (total effect near zero)")

    ci = result_unstd['bootstrap']['indirect_ci_95']
    excludes = result_unstd['bootstrap']['indirect_ci_excludes_zero']
    print(f"  Bootstrap 95% CI:          [{ci[0]:.4f}, {ci[1]:.4f}] {'(excludes 0)' if excludes else '(includes 0)'}")

    bk = result_unstd['baron_kenny_all_steps_met']
    mtype = result_unstd['mediation_type']
    print(f"  Baron & Kenny:             {'All steps met' if bk else 'NOT all steps met'} → {mtype} mediation")

    results[metric_col] = {
        "unstandardized": result_unstd,
        "standardized": result_std,
    }

# ============================================================
# Summary and pass criteria evaluation
# ============================================================
print(f"\n\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

any_passes = False
passing_metrics = []

for metric_col, metric_data in results.items():
    unstd = metric_data["unstandardized"]
    if "error" in unstd:
        print(f"  {QUALITY_METRICS[metric_col]}: ERROR - {unstd['error']}")
        continue

    prop = unstd['bootstrap']['proportion_mediated']
    ci_excludes = unstd['bootstrap']['indirect_ci_excludes_zero']

    if prop is not None and prop > 0.10 and ci_excludes:
        any_passes = True
        passing_metrics.append(metric_col)
        status = "PASS"
    else:
        status = "FAIL"

    prop_str = f"{prop:.3f}" if prop is not None else "N/A"
    print(f"  {QUALITY_METRICS[metric_col]:20s}: prop_mediated={prop_str}, CI_excludes_0={ci_excludes} → {status}")

print(f"\nPilot pass criteria: At least one metric shows proportion mediated > 0.10 with bootstrap CI excluding 0")
print(f"Passing metrics: {passing_metrics}")
pilot_pass = any_passes

# ============================================================
# Supplementary: Alternative direction test
# ============================================================
print(f"\n\n--- Supplementary: Alternative direction (Quality → Absorption → L0) ---")
print("  (If this also shows mediation, the direction is ambiguous)")

alt_results = {}
for metric_col, metric_label in QUALITY_METRICS.items():
    cols_needed = [metric_col, "absorption", "log_l0"] + covariates
    df_valid = df_l0[cols_needed].dropna()
    if len(df_valid) < 10:
        alt_results[metric_col] = {"error": "insufficient data"}
        continue

    import statsmodels.api as sm
    X_val = df_valid[metric_col].values
    M_val = df_valid["absorption"].values
    Y_val = df_valid["log_l0"].values
    cov_vals = df_valid[covariates].values

    # Path a: quality -> absorption
    Xa = sm.add_constant(np.column_stack([X_val, cov_vals]))
    ma = sm.OLS(M_val, Xa).fit()
    a_alt = ma.params[1]
    a_alt_se = ma.bse[1]

    # Path b: absorption -> log_l0 (controlling quality)
    Xbc = sm.add_constant(np.column_stack([X_val, M_val, cov_vals]))
    mbc = sm.OLS(Y_val, Xbc).fit()
    b_alt = mbc.params[2]
    b_alt_se = mbc.bse[2]

    # Indirect
    ind_alt = a_alt * b_alt
    sobel_se_alt = np.sqrt(a_alt**2 * b_alt_se**2 + b_alt**2 * a_alt_se**2)
    from scipy.stats import norm
    sobel_z_alt = ind_alt / sobel_se_alt if sobel_se_alt > 0 else 0
    sobel_p_alt = 2 * (1 - norm.cdf(abs(sobel_z_alt)))

    alt_results[metric_col] = {
        "indirect_effect": float(ind_alt),
        "sobel_z": float(sobel_z_alt),
        "sobel_p": float(sobel_p_alt),
    }
    print(f"  {metric_label:20s}: indirect={ind_alt:.4f}, Sobel z={sobel_z_alt:.3f}, p={sobel_p_alt:.4f}")

# ============================================================
# Compile final output
# ============================================================
end_time = datetime.now()
duration_min = (end_time - start_time).total_seconds() / 60

output = {
    "task_id": TASK_ID,
    "mode": "PILOT",
    "timestamp": start_time.isoformat(),
    "data_source": str(DATA_PATH),
    "n_total_saes": len(df),
    "n_with_l0": len(df_l0),
    "n_without_l0": len(df) - len(df_l0),
    "seed": SEED,
    "n_bootstrap": N_BOOTSTRAP,
    "covariates": covariates,
    "mediation_path": "log_L0 (X) → absorption (M) → quality (Y)",
    "results_by_metric": {},
    "alternative_direction_test": alt_results,
    "summary": {
        "pilot_pass": pilot_pass,
        "pilot_pass_criteria": "At least one metric shows proportion mediated > 0.10 with bootstrap CI excluding 0",
        "passing_metrics": passing_metrics,
        "n_metrics_passing": len(passing_metrics),
    },
    "duration_minutes": round(duration_min, 1),
}

# Flatten results for output
for metric_col, metric_data in results.items():
    unstd = metric_data["unstandardized"]
    std = metric_data.get("standardized")

    if "error" in unstd:
        output["results_by_metric"][metric_col] = {"error": unstd["error"]}
        continue

    entry = {
        "metric_label": QUALITY_METRICS[metric_col],
        "n": unstd["n"],
        "path_a": unstd["path_a"],
        "path_b": unstd["path_b"],
        "total_effect_c": unstd["total_effect_c"],
        "direct_effect_c_prime": unstd["direct_effect_c_prime"],
        "indirect_effect": unstd["indirect_effect"],
        "sobel_test": unstd["sobel_test"],
        "bootstrap": unstd["bootstrap"],
        "baron_kenny": unstd["baron_kenny"],
        "baron_kenny_all_steps_met": unstd["baron_kenny_all_steps_met"],
        "mediation_type": unstd["mediation_type"],
    }

    if std and "error" not in std:
        entry["standardized_coefficients"] = {
            "path_a": std["path_a"]["coefficient"],
            "path_b": std["path_b"]["coefficient"],
            "total_c": std["total_effect_c"]["coefficient"],
            "direct_c_prime": std["direct_effect_c_prime"]["coefficient"],
            "indirect_ab": std["indirect_effect"]["a_times_b"],
            "proportion_mediated": std["bootstrap"]["proportion_mediated"],
        }

    output["results_by_metric"][metric_col] = entry

# ============================================================
# Save results
# ============================================================
# Save to pilots/
pilot_path = RESULTS_DIR / f"{TASK_ID}.json"
with open(pilot_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nPilot results saved to: {pilot_path}")

# Also save to full/ (this is the complete analysis on all 48 SAEs with L0)
full_path = FULL_RESULTS_DIR / f"{TASK_ID}.json"
with open(full_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"Full results saved to: {full_path}")

# ============================================================
# Write DONE marker
# ============================================================
# Clean up PID file
if pid_file.exists():
    pid_file.unlink()

progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
final_progress = {}
if progress_file.exists():
    try:
        final_progress = json.loads(progress_file.read_text())
    except Exception:
        pass

done_marker = RESULTS_DIR / f"{TASK_ID}_DONE"
done_marker.write_text(json.dumps({
    "task_id": TASK_ID,
    "status": "success",
    "summary": f"Mediation analysis completed. {len(passing_metrics)}/4 metrics pass criteria. Pilot {'PASS' if pilot_pass else 'FAIL'}.",
    "final_progress": final_progress,
    "timestamp": datetime.now().isoformat(),
}))

print(f"\n{'='*60}")
print(f"PILOT {'PASS' if pilot_pass else 'FAIL'}")
print(f"Duration: {duration_min:.1f} minutes")
print(f"{'='*60}")
