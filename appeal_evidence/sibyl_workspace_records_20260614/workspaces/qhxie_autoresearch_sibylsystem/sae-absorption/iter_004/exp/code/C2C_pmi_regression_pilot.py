"""
C2C_pmi_regression_pilot.py
Pilot: PMI Regression Analysis (H2)

Pilot scope:
- Use pilot data from C2B (3 configs x 5 letters = 15 data points)
- Use C2A pilot PMI features (5 letters)
- Fit simplified regression: absorption_rate = beta0 + beta1*log(L0) + beta4*log(PMI)
- Note: 15 data points is too few for reliable inference; pilot only checks that
  regression runs without error and PMI coefficient has expected positive sign.

Pass criteria:
- Regression executes without error
- beta4 (PMI coefficient) is positive
- regression R^2 is finite
"""

import json
import os
import numpy as np
import scipy.stats as stats
from pathlib import Path
from datetime import datetime
import traceback


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

try:
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_hc3
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available, will use scipy OLS")

# ─── Paths ───────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
C2B_PILOT = WORKSPACE / "exp/results/pilots/C2B_absorption_survey_pilot.json"
C2A_FULL  = WORKSPACE / "exp/results/full/C2A_pmi_features.json"
OUT_PILOT = WORKSPACE / "exp/results/pilots/C2C_regression_pilot.json"
OUT_PILOT_MD = WORKSPACE / "exp/results/pilots/C2C_regression_pilot.md"
# Full output path (for pilot we write a note here too)
OUT_FULL  = WORKSPACE / "exp/results/full/C2C_regression_results.json"

os.makedirs(OUT_PILOT.parent, exist_ok=True)
os.makedirs(OUT_FULL.parent, exist_ok=True)

# Write PID for system tracking
pid_file = WORKSPACE / "exp/results/C2C_pmi_regression.pid"
pid_file.write_text(str(os.getpid()))

start_time = datetime.now()

def report_progress(epoch, total_epochs, step="", loss=None, metric=None):
    progress_file = WORKSPACE / "exp/results/C2C_pmi_regression_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": "C2C_pmi_regression",
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    pid_file.unlink(missing_ok=True)
    progress_file = WORKSPACE / "exp/results/C2C_pmi_regression_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except Exception:
            pass
    marker = WORKSPACE / "exp/results/C2C_pmi_regression_DONE"
    marker.write_text(json.dumps({
        "task_id": "C2C_pmi_regression",
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }, cls=NumpyEncoder))

print("=" * 60)
print("C2C PMI Regression Pilot")
print("=" * 60)

report_progress(1, 5, "Loading data")

# ─── Step 1: Load C2B pilot absorption data ───────────────────────────────────
print("\n[1/5] Loading C2B pilot absorption data...")
with open(C2B_PILOT) as f:
    c2b_data = json.load(f)

# Extract data points: each is (config_id, letter, absorption_rate, measured_l0, width_approx, model_layer)
data_points = c2b_data["data_points"]
print(f"  Loaded {len(data_points)} data points from C2B pilot")

# ─── Step 2: Load C2A PMI features and compute per-letter PMI ────────────────
print("\n[2/5] Loading C2A PMI features...")
with open(C2A_FULL) as f:
    c2a_data = json.load(f)

# The PMI data has many tokens per letter.
# We use the MEDIAN PMI of top-10 (highest PMI) tokens per letter as the
# representative "absorption pressure" PMI value.
# This represents the typical strength of the strongest letter associations.
features = c2a_data["features"]
from collections import defaultdict

# Group PMI scores by letter
letter_pmi_scores = defaultdict(list)
for entry in features:
    letter_pmi_scores[entry["letter"]].append(entry["pmi_score"])

# Compute representative PMI per letter (median of top-10)
letter_pmi = {}
for letter, scores in letter_pmi_scores.items():
    scores_sorted = sorted(scores, reverse=True)
    top10 = scores_sorted[:10]
    letter_pmi[letter] = float(np.median(top10))

print(f"  PMI scores per letter (median of top-10):")
for letter, pmi in sorted(letter_pmi.items()):
    print(f"    Letter {letter}: PMI={pmi:.4f}")

report_progress(2, 5, "Building regression dataset")

# ─── Step 3: Merge datasets ───────────────────────────────────────────────────
print("\n[3/5] Merging datasets...")

rows = []
for dp in data_points:
    letter = dp["letter"]
    if letter not in letter_pmi:
        print(f"  Warning: letter {letter} not in PMI data, skipping")
        continue

    pmi_val = letter_pmi[letter]
    l0_val = dp["measured_l0"]
    width_val = dp["width_approx"]
    layer_val = dp["model_layer"]
    absorption = dp["absorption_rate"]

    # Skip if PMI <= 0 (can't log-transform)
    # Instead, shift to ensure all positive (use pmi + offset)
    rows.append({
        "config_id": dp["config_id"],
        "letter": letter,
        "absorption_rate": absorption,
        "measured_l0": l0_val,
        "width_approx": width_val,
        "model_layer": layer_val,
        "pmi": pmi_val,
        "log_l0": np.log(l0_val) if l0_val > 0 else np.nan,
        "log_width": np.log(width_val) if width_val > 0 else np.nan,
        "log_pmi": np.log(pmi_val) if pmi_val > 0 else np.nan,
    })

print(f"  Merged dataset: {len(rows)} rows")

# Handle case where some PMI values are <= 0 (can't log-transform directly)
valid_rows = [r for r in rows if not np.isnan(r.get("log_pmi", np.nan))]
print(f"  Rows with valid log(PMI): {len(valid_rows)}")

# If too few valid rows, use shifted PMI (offset to ensure positive)
if len(valid_rows) < 10:
    print("  Warning: few valid log(PMI) values. Using shifted PMI (min + offset)")
    min_pmi = min(r["pmi"] for r in rows)
    offset = max(0.1 - min_pmi, 0.1)
    for r in rows:
        r["log_pmi_shifted"] = np.log(r["pmi"] + offset)
    valid_rows = rows
    use_shifted = True
    pmi_column = "log_pmi_shifted"
else:
    use_shifted = False
    pmi_column = "log_pmi"
    valid_rows = valid_rows

print(f"  Using PMI column: {pmi_column}, n={len(valid_rows)}")

# Extract arrays for regression
y = np.array([r["absorption_rate"] for r in valid_rows])
log_l0 = np.array([r["log_l0"] for r in valid_rows])
log_pmi = np.array([r[pmi_column] for r in valid_rows])

print(f"\n  y (absorption_rate): mean={y.mean():.4f}, std={y.std():.4f}")
print(f"  log_l0: mean={log_l0.mean():.4f}, std={log_l0.std():.4f}")
print(f"  log_pmi: mean={log_pmi.mean():.4f}, std={log_pmi.std():.4f}")

report_progress(3, 5, "Fitting regression models")

# ─── Step 4: Fit regression models ───────────────────────────────────────────
print("\n[4/5] Fitting regression models...")

results_dict = {
    "task_id": "C2C_pmi_regression",
    "mode": "PILOT",
    "timestamp": datetime.now().isoformat(),
    "n_observations": len(valid_rows),
    "data_summary": {
        "configs": list(set(r["config_id"] for r in valid_rows)),
        "letters": list(set(r["letter"] for r in valid_rows)),
        "n_configs": len(set(r["config_id"] for r in valid_rows)),
        "n_letters": len(set(r["letter"] for r in valid_rows)),
        "absorption_mean": float(y.mean()),
        "absorption_std": float(y.std()),
        "pmi_column_used": pmi_column,
        "pmi_shifted": use_shifted,
    },
    "regression_models": {},
    "pass_criteria": {},
    "pilot_note": "15 data points is too few for reliable inference; pilot only checks pipeline execution and sign of PMI coefficient",
}

# Model 1: Simplified pilot regression — absorption = beta0 + beta1*log(L0) + beta4*log(PMI)
print("  Fitting Model 1: absorption ~ log(L0) + log(PMI)")
X1 = np.column_stack([np.ones(len(valid_rows)), log_l0, log_pmi])

if HAS_STATSMODELS:
    try:
        model1 = sm.OLS(y, X1).fit(cov_type='HC3')
        beta0_1, beta1_1, beta4_1 = model1.params
        pval_beta4 = model1.pvalues[2]
        r2_1 = model1.rsquared
        adj_r2_1 = model1.rsquared_adj

        print(f"    beta0={beta0_1:.4f}, beta1(log_L0)={beta1_1:.4f}, beta4(log_PMI)={beta4_1:.4f}")
        print(f"    p-value(beta4)={pval_beta4:.4f}, R²={r2_1:.4f}, Adj-R²={adj_r2_1:.4f}")

        results_dict["regression_models"]["model1_full_pilot"] = {
            "formula": "absorption_rate ~ 1 + log(L0) + log(PMI)",
            "n_obs": len(valid_rows),
            "beta0": float(beta0_1),
            "beta1_log_l0": float(beta1_1),
            "beta4_log_pmi": float(beta4_1),
            "pvalue_beta4": float(pval_beta4),
            "r_squared": float(r2_1),
            "adj_r_squared": float(adj_r2_1),
            "se_type": "HC3",
            "aic": float(model1.aic),
            "bic": float(model1.bic),
            "summary": str(model1.summary()),
        }
    except Exception as e:
        print(f"    statsmodels OLS failed: {e}")
        HAS_STATSMODELS = False

if not HAS_STATSMODELS:
    # Fallback: scipy lstsq
    coeffs, residuals, rank, sv = np.linalg.lstsq(X1, y, rcond=None)
    beta0_1, beta1_1, beta4_1 = coeffs
    y_pred = X1 @ coeffs
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2_1 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0

    # Compute p-value via t-test (approximate)
    n, k = X1.shape
    dof = n - k
    mse = ss_res / dof if dof > 0 else np.nan
    var_coeffs = mse * np.linalg.pinv(X1.T @ X1) if mse is not np.nan else np.full((k,k), np.nan)
    se_beta4 = np.sqrt(var_coeffs[2, 2]) if not np.isnan(var_coeffs[2, 2]) else np.nan
    t_stat = beta4_1 / se_beta4 if se_beta4 and se_beta4 > 0 else np.nan
    pval_beta4 = 2 * stats.t.sf(abs(t_stat), dof) if not np.isnan(t_stat) else np.nan

    print(f"    (scipy fallback) beta0={beta0_1:.4f}, beta1(log_L0)={beta1_1:.4f}, beta4(log_PMI)={beta4_1:.4f}")
    print(f"    p-value(beta4)~{pval_beta4:.4f}, R²={r2_1:.4f}")

    results_dict["regression_models"]["model1_full_pilot"] = {
        "formula": "absorption_rate ~ 1 + log(L0) + log(PMI)",
        "n_obs": len(valid_rows),
        "beta0": float(beta0_1),
        "beta1_log_l0": float(beta1_1),
        "beta4_log_pmi": float(beta4_1),
        "pvalue_beta4": float(pval_beta4) if not np.isnan(pval_beta4) else None,
        "r_squared": float(r2_1),
        "adj_r_squared": None,
        "se_type": "OLS (scipy fallback)",
    }

# Model 2: PMI-only (ablation A3)
print("  Fitting Model 2: absorption ~ log(PMI) only (A3 ablation)")
X2 = np.column_stack([np.ones(len(valid_rows)), log_pmi])

if HAS_STATSMODELS:
    try:
        model2 = sm.OLS(y, X2).fit(cov_type='HC3')
        print(f"    PMI-only R²={model2.rsquared:.4f}, beta_pmi={model2.params[1]:.4f}")
        results_dict["regression_models"]["model2_pmi_only"] = {
            "formula": "absorption_rate ~ 1 + log(PMI)",
            "n_obs": len(valid_rows),
            "beta0": float(model2.params[0]),
            "beta_pmi": float(model2.params[1]),
            "r_squared": float(model2.rsquared),
            "adj_r_squared": float(model2.rsquared_adj),
        }
    except Exception as e:
        print(f"    Model 2 failed: {e}")

# Compute partial regression (residualize both y and log_pmi on log_l0)
print("  Computing partial regression (absorption vs PMI | log_L0)...")
X_ctrl = np.column_stack([np.ones(len(valid_rows)), log_l0])

if HAS_STATSMODELS:
    try:
        # Residualize y on log_l0
        resid_y = sm.OLS(y, X_ctrl).fit().resid
        # Residualize log_pmi on log_l0
        resid_pmi = sm.OLS(log_pmi, X_ctrl).fit().resid

        # Partial regression
        partial_r, partial_p = stats.pearsonr(resid_pmi, resid_y)
        partial_r2 = partial_r**2

        print(f"    Partial r (PMI | L0)={partial_r:.4f}, partial R²={partial_r2:.4f}, p={partial_p:.4f}")
        results_dict["regression_models"]["partial_regression"] = {
            "description": "Partial regression: absorption vs log(PMI) after residualizing on log(L0)",
            "partial_r": float(partial_r),
            "partial_r_squared": float(partial_r2),
            "partial_p": float(partial_p),
            "n_obs": len(valid_rows),
        }

        # Store residuals for scatter plot
        results_dict["visualization_data"] = {
            "partial_regression_scatter": [
                {"log_pmi_resid": float(resid_pmi[i]), "absorption_resid": float(resid_y[i]),
                 "letter": valid_rows[i]["letter"], "config_id": valid_rows[i]["config_id"]}
                for i in range(len(valid_rows))
            ]
        }
    except Exception as e:
        print(f"    Partial regression failed: {e}")

# Correlation analysis
corr_pmi_absorption, corr_pval = stats.pearsonr(log_pmi, y)
print(f"\n  Raw Pearson r(log_PMI, absorption_rate)={corr_pmi_absorption:.4f}, p={corr_pval:.4f}")
results_dict["correlation"] = {
    "pearson_r_log_pmi_absorption": float(corr_pmi_absorption),
    "pearson_pval": float(corr_pval),
    "spearman_r": float(stats.spearmanr(log_pmi, y)[0]),
    "spearman_pval": float(stats.spearmanr(log_pmi, y)[1]),
}

report_progress(4, 5, "Evaluating pass criteria")

# ─── Step 5: Evaluate pass criteria ──────────────────────────────────────────
print("\n[5/5] Evaluating pass criteria...")

model1_results = results_dict["regression_models"].get("model1_full_pilot", {})
beta4 = model1_results.get("beta4_log_pmi", None)
r2 = model1_results.get("r_squared", None)

crit1_no_error = True  # We got here!
crit2_beta4_positive = (beta4 is not None) and (beta4 > 0)
crit3_r2_finite = (r2 is not None) and np.isfinite(r2)

print(f"  [1] Regression executes without error: {'PASS' if crit1_no_error else 'FAIL'}")
print(f"  [2] beta4 (PMI coefficient) is positive: {'PASS' if crit2_beta4_positive else 'FAIL'} (beta4={beta4:.4f})" if beta4 is not None else f"  [2] beta4 not available: FAIL")
print(f"  [3] Regression R² is finite: {'PASS' if crit3_r2_finite else 'FAIL'} (R²={r2:.4f})" if r2 is not None else f"  [3] R² not available: FAIL")

all_pass = crit1_no_error and crit2_beta4_positive and crit3_r2_finite

results_dict["pass_criteria"] = {
    "criterion_1_no_error": {"pass": crit1_no_error, "description": "Regression executes without error"},
    "criterion_2_beta4_positive": {
        "pass": crit2_beta4_positive,
        "description": "beta4 (PMI coefficient) is positive",
        "beta4_value": float(beta4) if beta4 is not None else None,
    },
    "criterion_3_r2_finite": {
        "pass": crit3_r2_finite,
        "description": "Regression R² is finite",
        "r2_value": float(r2) if r2 is not None else None,
    },
    "all_pass": all_pass,
}

go_no_go = "GO" if all_pass else "NO_GO"
results_dict["go_no_go"] = go_no_go

# Runtime
end_time = datetime.now()
runtime = (end_time - start_time).total_seconds()
results_dict["runtime_seconds"] = runtime

print(f"\n{'='*60}")
print(f"GO/NO-GO: {go_no_go}")
print(f"Runtime: {runtime:.1f}s")
print(f"{'='*60}")

# ─── Save pilot output ────────────────────────────────────────────────────────
with open(OUT_PILOT, "w") as f:
    json.dump(results_dict, f, indent=2, cls=NumpyEncoder)
print(f"\nPilot results saved to: {OUT_PILOT}")

# ─── Write pilot markdown summary ─────────────────────────────────────────────
m1 = results_dict["regression_models"].get("model1_full_pilot", {})
partial = results_dict["regression_models"].get("partial_regression", {})

md_content = f"""# C2C PMI Regression Pilot Summary

**Mode:** PILOT
**Task ID:** C2C_pmi_regression
**Timestamp:** {results_dict['timestamp']}
**GO/NO-GO:** {go_no_go}

## Pilot Scope

Using pilot data from C2B (3 configs x 5 letters = **{results_dict['n_observations']} data points**) and C2A (5-letter PMI features).

**PMI aggregation:** Median of top-10 PMI tokens per letter (proxy for letter-category absorption pressure)

## Simplified Regression Model (Pilot)

```
absorption_rate = beta0 + beta1*log(L0) + beta4*log(PMI)
```

| Parameter | Value |
|-----------|-------|
| beta0 (intercept) | {m1.get('beta0', 'N/A'):.4f} |
| beta1 (log L0) | {m1.get('beta1_log_l0', 'N/A'):.4f} |
| beta4 (log PMI) | {m1.get('beta4_log_pmi', 'N/A'):.4f} |
| p-value (beta4) | {m1.get('pvalue_beta4', 'N/A'):.4f} |
| R² | {m1.get('r_squared', 'N/A'):.4f} |
| Adj-R² | {m1.get('adj_r_squared', 'N/A') if m1.get('adj_r_squared') is not None else 'N/A'} |
| SE type | {m1.get('se_type', 'N/A')} |

## Correlation Analysis

| Metric | Value |
|--------|-------|
| Pearson r(log_PMI, absorption) | {results_dict['correlation']['pearson_r_log_pmi_absorption']:.4f} |
| Pearson p-value | {results_dict['correlation']['pearson_pval']:.4f} |
| Spearman r | {results_dict['correlation']['spearman_r']:.4f} |

## Partial Regression (absorption vs PMI | L0)

| Metric | Value |
|--------|-------|
| Partial r | {partial.get('partial_r', 'N/A'):.4f} |
| Partial R² | {partial.get('partial_r_squared', 'N/A'):.4f} |
| Partial p | {partial.get('partial_p', 'N/A'):.4f} |

## Pass Criteria

| Criterion | Status |
|-----------|--------|
| Regression executes without error | {'PASS' if crit1_no_error else 'FAIL'} |
| beta4 (PMI coefficient) positive | {'PASS (' + str(round(beta4, 4)) + ')' if crit2_beta4_positive else 'FAIL (' + str(round(beta4, 4) if beta4 else 'N/A') + ')'} |
| R² is finite | {'PASS (' + str(round(r2, 4)) + ')' if crit3_r2_finite else 'FAIL'} |

## Caveats

- **15 data points is insufficient for reliable statistical inference** — this is a pipeline check only
- The full regression (C2C full run) requires C2B full data (30 SAE configs x 26 letters = 780 data points)
- PMI aggregation (median of top-10 tokens) is a proxy; the full run will use the canonical per-letter PMI from the sae-spelling feature labels
- With n=15 and k=3 parameters, degrees of freedom = 12; all p-values are unreliable

## Runtime

{runtime:.1f} seconds
"""

with open(OUT_PILOT_MD, "w") as f:
    f.write(md_content)
print(f"Pilot markdown saved to: {OUT_PILOT_MD}")

# Write DONE marker
mark_done(
    status="success",
    summary=f"C2C pilot: go_no_go={go_no_go}, beta4={beta4:.4f}, R²={r2:.4f}, n_obs={results_dict['n_observations']}"
)

print("\nDone!")
