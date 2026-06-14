"""
C2C_pmi_regression_full.py
Full: PMI Regression Analysis (H2) with Ablations A3/A4

Full scope:
- Load C2B absorption survey (31 configs x 26 letters = 806 rows)
- Load C2A PMI features (286k token-level entries, aggregate per letter)
- Merge on letter and fit the specified regression model:
    absorption_rate_il = beta0 + beta1*log(L0_i) + beta2*log(width_i)
                        + beta3*layer_i + beta4*log(PMI_l) + epsilon_il
- Use OLS with HC3 heteroskedasticity-robust standard errors
- Report: R^2, adj R^2, partial R^2 for PMI, p-value for beta4, signs
- Ablation A3: PMI-only model
- Ablation A4: per-layer regression stability
- Partial regression plot: absorption residuals vs log(PMI) residuals

Key methodological notes:
- PMI is per-letter (26 values); SAE config is per-row (31 configs)
- Regression is at the (config x letter) cell level = 806 observations
- Cluster-robust SEs at the letter level would be ideal but HC3 is specified
- PMI aggregation: median of top-10 PMI tokens per letter
"""

import json
import os
import sys
import numpy as np
import scipy.stats as stats
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import traceback
import warnings
warnings.filterwarnings('ignore')


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


# ─── Paths ───────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
C2B_PARQUET = WORKSPACE / "exp/results/full/C2B_absorption_survey.parquet"
C2A_JSON = WORKSPACE / "exp/results/full/C2A_pmi_features.json"
OUT_JSON = WORKSPACE / "exp/results/full/C2C_regression_results.json"
OUT_MD = WORKSPACE / "exp/results/full/C2C_regression_summary.md"
PLOT_DIR = WORKSPACE / "exp/results/full/C2C_plots"

os.makedirs(OUT_JSON.parent, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# PID file for system tracking
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


try:
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_hc3
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                          "pandas", "statsmodels", "matplotlib", "pyarrow"])
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


print("=" * 70)
print("C2C PMI Regression — FULL Analysis (H2)")
print("=" * 70)

report_progress(1, 8, "Loading C2B absorption survey")

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Load C2B absorption survey data
# ═══════════════════════════════════════════════════════════════════════════
print("\n[1/8] Loading C2B absorption survey data...")
df = pd.read_parquet(C2B_PARQUET)
print(f"  Loaded {len(df)} rows from C2B parquet")
print(f"  Configs: {df['config_id'].nunique()}, Letters: {df['letter'].nunique()}")
print(f"  Absorption rate: mean={df['absorption_rate'].mean():.4f}, "
      f"std={df['absorption_rate'].std():.4f}, "
      f"range=[{df['absorption_rate'].min():.4f}, {df['absorption_rate'].max():.4f}]")

# Remove rows with errors
df = df[df['has_error'] == 0].copy()
print(f"  After filtering errors: {len(df)} rows")

report_progress(2, 8, "Loading C2A PMI features")

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Load C2A PMI features and compute per-letter PMI
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2/8] Loading C2A PMI features...")
with open(C2A_JSON) as f:
    c2a_data = json.load(f)

features = c2a_data["features"]
print(f"  Loaded {len(features)} token-level PMI entries")

# Group PMI scores by letter
letter_pmi_scores = defaultdict(list)
for entry in features:
    letter_pmi_scores[entry["letter"]].append(entry["pmi_score"])

# Compute multiple PMI aggregations per letter for robustness
letter_pmi_agg = {}
for letter, scores in sorted(letter_pmi_scores.items()):
    scores_sorted = sorted(scores, reverse=True)
    top10 = scores_sorted[:10]
    top20 = scores_sorted[:20]
    top50 = scores_sorted[:50]

    letter_pmi_agg[letter] = {
        "pmi_median_top10": float(np.median(top10)),
        "pmi_mean_top10": float(np.mean(top10)),
        "pmi_mean_top20": float(np.mean(top20)),
        "pmi_mean_top50": float(np.mean(top50)),
        "pmi_max": float(scores_sorted[0]),
        "pmi_mean_all": float(np.mean(scores)),
        "pmi_std": float(np.std(scores)),
        "n_tokens": len(scores),
        "parent_frequency": c2a_data["letter_stats"][letter]["parent_frequency"],
    }

print(f"  PMI aggregations computed for {len(letter_pmi_agg)} letters")
print(f"  PMI (median top-10) range: "
      f"[{min(v['pmi_median_top10'] for v in letter_pmi_agg.values()):.4f}, "
      f"{max(v['pmi_median_top10'] for v in letter_pmi_agg.values()):.4f}]")

# Print per-letter PMI summary
print("\n  Letter PMI Summary (median top-10):")
for letter in sorted(letter_pmi_agg.keys()):
    info = letter_pmi_agg[letter]
    print(f"    {letter}: PMI_med_t10={info['pmi_median_top10']:.4f}, "
          f"PMI_max={info['pmi_max']:.4f}, parent_freq={info['parent_frequency']}")

report_progress(3, 8, "Merging datasets")

# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Merge datasets
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3/8] Merging datasets...")

# Add PMI features to the absorption dataframe
pmi_df = pd.DataFrame([
    {"letter": letter, **vals}
    for letter, vals in letter_pmi_agg.items()
])
df = df.merge(pmi_df, on="letter", how="inner")
print(f"  Merged dataset: {len(df)} rows")

# Create log-transformed features
# For PMI: all median-top-10 values are positive (they're top PMI scores),
# but we need to handle edge cases
df["log_l0"] = np.log(df["measured_l0"])
df["log_width"] = np.log(df["width_approx"])
df["log_pmi"] = np.log(df["pmi_median_top10"].clip(lower=0.001))
df["log_parent_freq"] = np.log(df["parent_frequency"])

# Check for any issues
n_valid = df["log_pmi"].notna().sum()
n_total = len(df)
print(f"  Valid log(PMI) values: {n_valid}/{n_total}")

# Summary of regression variables
print(f"\n  Regression variables summary:")
for col in ["absorption_rate", "log_l0", "log_width", "model_layer", "log_pmi"]:
    print(f"    {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, "
          f"range=[{df[col].min():.4f}, {df[col].max():.4f}]")

report_progress(4, 8, "Fitting full regression model")

# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Fit the full regression model (H2 main specification)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[4/8] Fitting full regression model...")
print("  Model: absorption_rate = b0 + b1*log(L0) + b2*log(width) + b3*layer + b4*log(PMI)")

y = df["absorption_rate"].values
X_full = df[["log_l0", "log_width", "model_layer", "log_pmi"]].values
X_full_const = sm.add_constant(X_full)

model_full = sm.OLS(y, X_full_const).fit(cov_type='HC3')
print(f"\n{model_full.summary()}")

# Extract key results
full_results = {
    "formula": "absorption_rate ~ 1 + log(L0) + log(width) + layer + log(PMI)",
    "n_obs": int(model_full.nobs),
    "r_squared": float(model_full.rsquared),
    "adj_r_squared": float(model_full.rsquared_adj),
    "f_statistic": float(model_full.fvalue),
    "f_pvalue": float(model_full.f_pvalue),
    "aic": float(model_full.aic),
    "bic": float(model_full.bic),
    "se_type": "HC3",
    "coefficients": {
        "const": {
            "value": float(model_full.params[0]),
            "se": float(model_full.bse[0]),
            "t": float(model_full.tvalues[0]),
            "pvalue": float(model_full.pvalues[0]),
            "ci_lower": float(model_full.conf_int()[0, 0]),
            "ci_upper": float(model_full.conf_int()[0, 1]),
        },
        "log_L0": {
            "value": float(model_full.params[1]),
            "se": float(model_full.bse[1]),
            "t": float(model_full.tvalues[1]),
            "pvalue": float(model_full.pvalues[1]),
            "ci_lower": float(model_full.conf_int()[1, 0]),
            "ci_upper": float(model_full.conf_int()[1, 1]),
        },
        "log_width": {
            "value": float(model_full.params[2]),
            "se": float(model_full.bse[2]),
            "t": float(model_full.tvalues[2]),
            "pvalue": float(model_full.pvalues[2]),
            "ci_lower": float(model_full.conf_int()[2, 0]),
            "ci_upper": float(model_full.conf_int()[2, 1]),
        },
        "layer": {
            "value": float(model_full.params[3]),
            "se": float(model_full.bse[3]),
            "t": float(model_full.tvalues[3]),
            "pvalue": float(model_full.pvalues[3]),
            "ci_lower": float(model_full.conf_int()[3, 0]),
            "ci_upper": float(model_full.conf_int()[3, 1]),
        },
        "log_PMI": {
            "value": float(model_full.params[4]),
            "se": float(model_full.bse[4]),
            "t": float(model_full.tvalues[4]),
            "pvalue": float(model_full.pvalues[4]),
            "ci_lower": float(model_full.conf_int()[4, 0]),
            "ci_upper": float(model_full.conf_int()[4, 1]),
        },
    },
    "summary_text": str(model_full.summary()),
}

# Print coefficient signs
print("\n  Coefficient Signs:")
coef_names = ["const", "log_L0", "log_width", "layer", "log_PMI"]
for name in coef_names:
    c = full_results["coefficients"][name]
    sign = "+" if c["value"] > 0 else "-"
    sig = "***" if c["pvalue"] < 0.001 else "**" if c["pvalue"] < 0.01 else "*" if c["pvalue"] < 0.05 else ""
    print(f"    {name:12s}: {sign}{abs(c['value']):.6f} (p={c['pvalue']:.4f}){sig}")

report_progress(5, 8, "Computing partial R^2 for PMI")

# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Partial R^2 for PMI term
# ═══════════════════════════════════════════════════════════════════════════
print("\n[5/8] Computing partial R^2 for PMI term...")

# Restricted model (without PMI)
X_no_pmi = df[["log_l0", "log_width", "model_layer"]].values
X_no_pmi_const = sm.add_constant(X_no_pmi)
model_no_pmi = sm.OLS(y, X_no_pmi_const).fit(cov_type='HC3')

r2_full = model_full.rsquared
r2_restricted = model_no_pmi.rsquared
partial_r2_pmi = (r2_full - r2_restricted) / (1 - r2_restricted)

print(f"  R^2 (full model): {r2_full:.6f}")
print(f"  R^2 (without PMI): {r2_restricted:.6f}")
print(f"  Partial R^2 (PMI): {partial_r2_pmi:.6f}")

# Also compute via partial regression (Frisch-Waugh-Lovell)
# Residualize y and log_pmi on the other covariates
resid_y_on_config = sm.OLS(y, X_no_pmi_const).fit().resid
resid_pmi_on_config = sm.OLS(df["log_pmi"].values, X_no_pmi_const).fit().resid

# Partial correlation
partial_r, partial_p = stats.pearsonr(resid_pmi_on_config, resid_y_on_config)
print(f"  Partial r (PMI | config): {partial_r:.6f}")
print(f"  Partial p-value: {partial_p:.6f}")

partial_results = {
    "r2_full": float(r2_full),
    "r2_without_pmi": float(r2_restricted),
    "partial_r2_pmi": float(partial_r2_pmi),
    "partial_r_pmi": float(partial_r),
    "partial_p_pmi": float(partial_p),
    "n_obs": int(len(y)),
    "model_without_pmi_summary": str(model_no_pmi.summary()),
}

report_progress(6, 8, "Ablation analyses (A3, A4)")

# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Ablation A3 — PMI-only model
# ═══════════════════════════════════════════════════════════════════════════
print("\n[6/8] Ablation A3: PMI-only model...")

X_pmi_only = sm.add_constant(df["log_pmi"].values)
model_pmi_only = sm.OLS(y, X_pmi_only).fit(cov_type='HC3')

print(f"  PMI-only R^2: {model_pmi_only.rsquared:.6f}")
print(f"  PMI-only beta: {model_pmi_only.params[1]:.6f} (p={model_pmi_only.pvalues[1]:.4f})")
print(f"\n{model_pmi_only.summary()}")

ablation_a3 = {
    "description": "PMI-only model (ablation A3)",
    "formula": "absorption_rate ~ 1 + log(PMI)",
    "n_obs": int(model_pmi_only.nobs),
    "r_squared": float(model_pmi_only.rsquared),
    "adj_r_squared": float(model_pmi_only.rsquared_adj),
    "beta_pmi": float(model_pmi_only.params[1]),
    "se_pmi": float(model_pmi_only.bse[1]),
    "pvalue_pmi": float(model_pmi_only.pvalues[1]),
    "summary_text": str(model_pmi_only.summary()),
}

# ═══════════════════════════════════════════════════════════════════════════
# Step 6b: Ablation A4 — Per-layer regression stability
# ═══════════════════════════════════════════════════════════════════════════
print("\n  Ablation A4: Per-layer regression stability...")

layers = sorted(df["model_layer"].unique())
per_layer_results = {}
pmi_coef_signs = {}

for layer in layers:
    df_layer = df[df["model_layer"] == layer]
    if len(df_layer) < 5:
        print(f"    Layer {layer}: too few rows ({len(df_layer)}), skipping")
        continue

    y_layer = df_layer["absorption_rate"].values

    # Check if there's any variance in absorption at this layer
    if y_layer.std() < 1e-10:
        print(f"    Layer {layer}: zero variance in absorption (all={y_layer.mean():.4f}), skipping")
        per_layer_results[str(layer)] = {
            "n_obs": int(len(df_layer)),
            "absorption_mean": float(y_layer.mean()),
            "note": "zero variance in absorption",
            "beta_pmi": None,
            "pmi_sign": None,
        }
        continue

    # For per-layer: build regressor set adaptively based on available variation
    # Some layers have only 1 config, so log_l0 and log_width are constants
    regressor_cols = []
    regressor_names = []
    pmi_idx = None

    for col, name in [("log_l0", "log_l0"), ("log_width", "log_width"), ("log_pmi", "log_pmi")]:
        if df_layer[col].std() > 1e-10:  # has variation
            regressor_cols.append(col)
            regressor_names.append(name)

    if "log_pmi" not in regressor_cols:
        print(f"    Layer {layer}: no variation in log_pmi, skipping (should not happen)")
        per_layer_results[str(layer)] = {
            "n_obs": int(len(df_layer)),
            "note": "no variation in PMI",
            "beta_pmi": None,
            "pmi_sign": None,
        }
        continue

    pmi_idx = regressor_cols.index("log_pmi") + 1  # +1 for constant

    X_layer = df_layer[regressor_cols].values
    X_layer_const = sm.add_constant(X_layer)

    try:
        model_layer = sm.OLS(y_layer, X_layer_const).fit(cov_type='HC3')
        beta_pmi = float(model_layer.params[pmi_idx])
        pval_pmi = float(model_layer.pvalues[pmi_idx])
        sign = "positive" if beta_pmi > 0 else "negative"
        pmi_coef_signs[str(layer)] = sign

        n_configs = df_layer["config_id"].nunique()
        regressors_used = ", ".join(regressor_names)
        print(f"    Layer {layer}: n={len(df_layer)} ({n_configs} configs), "
              f"regressors=[{regressors_used}], "
              f"beta_pmi={beta_pmi:.6f} ({sign}), "
              f"p={pval_pmi:.4f}, R^2={model_layer.rsquared:.4f}")

        result_entry = {
            "n_obs": int(len(df_layer)),
            "n_configs": n_configs,
            "regressors_used": regressor_names,
            "beta_pmi": float(beta_pmi),
            "se_pmi": float(model_layer.bse[pmi_idx]),
            "pvalue_pmi": float(pval_pmi),
            "r_squared": float(model_layer.rsquared),
            "pmi_sign": sign,
            "summary_text": str(model_layer.summary()),
        }
        # Add other coefficients if present
        for i, name in enumerate(regressor_names):
            if name != "log_pmi":
                result_entry[f"beta_{name}"] = float(model_layer.params[i + 1])
        per_layer_results[str(layer)] = result_entry
    except Exception as e:
        print(f"    Layer {layer}: regression failed — {e}")
        traceback.print_exc()
        per_layer_results[str(layer)] = {
            "n_obs": int(len(df_layer)),
            "error": str(e),
        }

# Check sign consistency
valid_signs = [v for v in pmi_coef_signs.values() if v is not None]
sign_consistent = len(set(valid_signs)) <= 1 if valid_signs else None
print(f"\n  PMI coefficient signs across layers: {pmi_coef_signs}")
print(f"  Sign consistent: {sign_consistent}")

ablation_a4 = {
    "description": "Per-layer regression (ablation A4)",
    "per_layer": per_layer_results,
    "pmi_coefficient_signs": pmi_coef_signs,
    "sign_consistent": sign_consistent,
}

report_progress(7, 8, "Generating visualizations")

# ═══════════════════════════════════════════════════════════════════════════
# Step 7: Additional analyses and diagnostics
# ═══════════════════════════════════════════════════════════════════════════
print("\n[7/8] Additional analyses and visualizations...")

# 7a: Correlation matrix
print("  Computing correlation matrix...")
corr_vars = ["absorption_rate", "log_l0", "log_width", "model_layer", "log_pmi"]
corr_matrix = df[corr_vars].corr()
print(f"\n{corr_matrix.to_string()}")

# Raw correlation between PMI and absorption
raw_r, raw_p = stats.pearsonr(df["log_pmi"], df["absorption_rate"])
raw_spearman, raw_sp_p = stats.spearmanr(df["log_pmi"], df["absorption_rate"])
print(f"\n  Raw Pearson r(log_PMI, absorption): {raw_r:.4f} (p={raw_p:.4f})")
print(f"  Raw Spearman rho(log_PMI, absorption): {raw_spearman:.4f} (p={raw_sp_p:.4f})")

# 7b: Per-letter mean absorption vs PMI scatter
print("  Computing per-letter absorption vs PMI...")
letter_means = df.groupby("letter").agg({
    "absorption_rate": "mean",
    "log_pmi": "first",  # same per letter
    "pmi_median_top10": "first",
}).reset_index()

letter_r, letter_p = stats.pearsonr(letter_means["log_pmi"], letter_means["absorption_rate"])
letter_spearman, letter_sp_p = stats.spearmanr(letter_means["log_pmi"], letter_means["absorption_rate"])
print(f"  Per-letter Pearson r: {letter_r:.4f} (p={letter_p:.4f})")
print(f"  Per-letter Spearman rho: {letter_spearman:.4f} (p={letter_sp_p:.4f})")

# 7c: Generate partial regression plot
print("  Generating partial regression plot...")
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Partial regression (FWL) — each point is a (config, letter) cell
    ax1 = axes[0]
    ax1.scatter(resid_pmi_on_config, resid_y_on_config, alpha=0.3, s=15, c='steelblue')

    # Add regression line
    slope_partial = np.polyfit(resid_pmi_on_config, resid_y_on_config, 1)
    x_line = np.linspace(resid_pmi_on_config.min(), resid_pmi_on_config.max(), 100)
    ax1.plot(x_line, np.polyval(slope_partial, x_line), 'r-', linewidth=2, label=f'r={partial_r:.3f}')
    ax1.set_xlabel("log(PMI) residuals\n(controlling for log(L0), log(width), layer)")
    ax1.set_ylabel("Absorption rate residuals\n(controlling for SAE config)")
    ax1.set_title(f"Partial Regression: Absorption vs log(PMI)\n(n={len(y)}, partial r={partial_r:.3f}, p={partial_p:.4f})")
    ax1.legend()
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

    # Plot 2: Per-letter aggregated scatter (more interpretable)
    ax2 = axes[1]
    ax2.scatter(letter_means["log_pmi"], letter_means["absorption_rate"],
                s=60, c='darkorange', edgecolors='black', linewidths=0.5, zorder=5)

    # Label points with letters
    for _, row in letter_means.iterrows():
        ax2.annotate(row["letter"], (row["log_pmi"], row["absorption_rate"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Add regression line
    slope_letter = np.polyfit(letter_means["log_pmi"], letter_means["absorption_rate"], 1)
    x_line2 = np.linspace(letter_means["log_pmi"].min(), letter_means["log_pmi"].max(), 100)
    ax2.plot(x_line2, np.polyval(slope_letter, x_line2), 'r-', linewidth=2,
             label=f'r={letter_r:.3f}')
    ax2.set_xlabel("log(PMI) — median of top-10 tokens per letter")
    ax2.set_ylabel("Mean absorption rate (across 31 SAE configs)")
    ax2.set_title(f"Per-Letter: Mean Absorption vs log(PMI)\n(n=26, r={letter_r:.3f}, p={letter_p:.4f})")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "C2C_partial_regression_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved to {PLOT_DIR / 'C2C_partial_regression_plot.png'}")
except Exception as e:
    print(f"    Plot generation failed: {e}")
    traceback.print_exc()

# 7d: Per-layer coefficient bar plot
print("  Generating per-layer coefficient plot...")
try:
    valid_layers = [l for l in layers if str(l) in per_layer_results
                    and per_layer_results[str(l)].get("beta_pmi") is not None]
    if valid_layers:
        fig, ax = plt.subplots(figsize=(10, 5))
        betas = [per_layer_results[str(l)]["beta_pmi"] for l in valid_layers]
        ses = [per_layer_results[str(l)].get("se_pmi", 0) for l in valid_layers]
        colors = ['green' if b > 0 else 'red' for b in betas]

        bars = ax.bar(range(len(valid_layers)), betas, yerr=ses,
                      color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(valid_layers)))
        ax.set_xticklabels([f"L{l}" for l in valid_layers])
        ax.set_xlabel("Layer")
        ax.set_ylabel("PMI Coefficient (beta4)")
        ax.set_title("Ablation A4: PMI Coefficient Stability Across Layers")
        ax.axhline(y=0, color='black', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(PLOT_DIR / "C2C_per_layer_pmi_coef.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved to {PLOT_DIR / 'C2C_per_layer_pmi_coef.png'}")
except Exception as e:
    print(f"    Per-layer plot failed: {e}")

# 7e: Width effect plot (absorption vs log(width) by layer group)
print("  Generating width effect plot...")
try:
    # Focus on feature-splitting SAEs at layer 8 for clear width effect
    df_l8_fs = df[df["layer_group"] == "feature_splitting"].sort_values("width_approx")
    if len(df_l8_fs) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        l8_means = df_l8_fs.groupby("width_approx")["absorption_rate"].agg(["mean", "std"]).reset_index()
        ax.errorbar(np.log(l8_means["width_approx"]), l8_means["mean"],
                     yerr=l8_means["std"]/np.sqrt(26), fmt='o-', capsize=5,
                     color='steelblue', linewidth=2, markersize=8)
        ax.set_xlabel("log(SAE Width)")
        ax.set_ylabel("Mean Absorption Rate")
        ax.set_title("Width Effect on Absorption (Layer 8, Feature-Splitting SAEs)")

        # Add width labels
        for _, row in l8_means.iterrows():
            ax.annotate(f"{int(row['width_approx']):,}",
                       (np.log(row['width_approx']), row['mean']),
                       textcoords="offset points", xytext=(0, 10), fontsize=8, ha='center')

        plt.tight_layout()
        plt.savefig(PLOT_DIR / "C2C_width_effect_layer8.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved to {PLOT_DIR / 'C2C_width_effect_layer8.png'}")
except Exception as e:
    print(f"    Width effect plot failed: {e}")

report_progress(8, 8, "Compiling results")

# ═══════════════════════════════════════════════════════════════════════════
# Step 8: Compile and save results
# ═══════════════════════════════════════════════════════════════════════════
print("\n[8/8] Compiling and saving results...")

# Build visualization data for partial regression scatter
viz_data = []
for i in range(len(df)):
    viz_data.append({
        "log_pmi_resid": float(resid_pmi_on_config[i]),
        "absorption_resid": float(resid_y_on_config[i]),
        "letter": df.iloc[i]["letter"],
        "config_id": df.iloc[i]["config_id"],
    })

results = {
    "task_id": "C2C_pmi_regression",
    "mode": "FULL",
    "timestamp": datetime.now().isoformat(),
    "model": "gpt2-small",
    "model_note": "GPT-2 Small (open-model anchor; Gemma-2-2b requires gated HF access)",
    "n_observations": int(len(df)),
    "n_configs": int(df["config_id"].nunique()),
    "n_letters": int(df["letter"].nunique()),
    "data_summary": {
        "absorption_mean": float(df["absorption_rate"].mean()),
        "absorption_std": float(df["absorption_rate"].std()),
        "absorption_range": [float(df["absorption_rate"].min()), float(df["absorption_rate"].max())],
        "pmi_aggregation": "median_top10",
        "pmi_range": [float(df["pmi_median_top10"].min()), float(df["pmi_median_top10"].max())],
        "layers": sorted(df["model_layer"].unique().tolist()),
        "width_range": [int(df["width_approx"].min()), int(df["width_approx"].max())],
    },
    "regression_models": {
        "full_model": full_results,
        "partial_r2_analysis": partial_results,
        "ablation_a3_pmi_only": ablation_a3,
        "ablation_a4_per_layer": ablation_a4,
    },
    "correlation_analysis": {
        "raw_pearson_r_log_pmi_absorption": float(raw_r),
        "raw_pearson_pval": float(raw_p),
        "raw_spearman_rho": float(raw_spearman),
        "raw_spearman_pval": float(raw_sp_p),
        "per_letter_pearson_r": float(letter_r),
        "per_letter_pearson_pval": float(letter_p),
        "per_letter_spearman_rho": float(letter_spearman),
        "per_letter_spearman_pval": float(letter_sp_p),
        "correlation_matrix": corr_matrix.to_dict(),
    },
    "per_letter_pmi": letter_pmi_agg,
    "visualization_data": {
        "partial_regression_scatter": viz_data[:200],  # Subsample for JSON size
        "per_letter_scatter": letter_means.to_dict(orient="records"),
    },
    "hypothesis_assessment": {
        "H2_corpus_pmi_predictor": {
            "beta4_log_pmi": full_results["coefficients"]["log_PMI"]["value"],
            "beta4_pvalue": full_results["coefficients"]["log_PMI"]["pvalue"],
            "beta4_sign": "positive" if full_results["coefficients"]["log_PMI"]["value"] > 0 else "negative",
            "partial_r2_pmi": float(partial_r2_pmi),
            "partial_r_pmi": float(partial_r),
            "partial_p_pmi": float(partial_p),
            "full_r_squared": float(r2_full),
            "pmi_only_r_squared": float(model_pmi_only.rsquared),
            "per_letter_pearson_r": float(letter_r),
            "criterion_partial_r2_ge_010": partial_r2_pmi >= 0.10,
            "criterion_pearson_r_gt_050": abs(letter_r) > 0.50,
            "assessment": None,  # will be filled below
        },
        "A3_pmi_standalone": {
            "pmi_only_r_squared": float(model_pmi_only.rsquared),
            "pmi_only_beta": float(model_pmi_only.params[1]),
            "pmi_only_pvalue": float(model_pmi_only.pvalues[1]),
            "criterion_partial_r2_ge_010": partial_r2_pmi >= 0.10,
        },
        "A4_per_layer_stability": {
            "sign_consistent": sign_consistent,
            "layer_signs": pmi_coef_signs,
            "criterion_same_sign_all_layers": sign_consistent,
        },
    },
    "plots": [
        "exp/results/full/C2C_plots/C2C_partial_regression_plot.png",
        "exp/results/full/C2C_plots/C2C_per_layer_pmi_coef.png",
        "exp/results/full/C2C_plots/C2C_width_effect_layer8.png",
    ],
}

# Generate H2 assessment
h2 = results["hypothesis_assessment"]["H2_corpus_pmi_predictor"]
beta4_val = h2["beta4_log_pmi"]
beta4_p = h2["beta4_pvalue"]
pr2 = h2["partial_r2_pmi"]

if beta4_val > 0 and beta4_p < 0.05 and pr2 >= 0.10:
    h2["assessment"] = "SUPPORTED: PMI significantly predicts absorption with meaningful effect size"
elif beta4_val > 0 and beta4_p < 0.05:
    h2["assessment"] = f"PARTIALLY SUPPORTED: PMI effect positive and significant but partial R^2={pr2:.4f} < 0.10"
elif beta4_val > 0:
    h2["assessment"] = f"WEAKLY SUPPORTED: PMI coefficient positive but not significant (p={beta4_p:.4f})"
else:
    h2["assessment"] = f"NOT SUPPORTED: PMI coefficient is negative ({beta4_val:.4f})"

# Save JSON
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)
print(f"  Full results saved to: {OUT_JSON}")

# ═══════════════════════════════════════════════════════════════════════════
# Generate markdown summary
# ═══════════════════════════════════════════════════════════════════════════
end_time = datetime.now()
runtime_seconds = (end_time - start_time).total_seconds()

coefs = full_results["coefficients"]

md = f"""# C2C PMI Regression Analysis (Full) — Summary

**Mode**: FULL  |  **Model**: GPT-2 Small (open-model anchor)
**Timestamp**: {results['timestamp']}
**N observations**: {results['n_observations']} ({results['n_configs']} SAE configs x {results['n_letters']} letters)
**Runtime**: {runtime_seconds:.1f} seconds

## H2 Assessment

**{h2['assessment']}**

## Full Regression Model

```
absorption_rate = b0 + b1*log(L0) + b2*log(width) + b3*layer + b4*log(PMI)
Standard errors: HC3 (heteroskedasticity-robust)
```

| Coefficient | Value | SE | t | p-value | 95% CI |
|------------|-------|-----|---|---------|--------|
| const | {coefs['const']['value']:.6f} | {coefs['const']['se']:.6f} | {coefs['const']['t']:.3f} | {coefs['const']['pvalue']:.4f} | [{coefs['const']['ci_lower']:.4f}, {coefs['const']['ci_upper']:.4f}] |
| log(L0) | {coefs['log_L0']['value']:.6f} | {coefs['log_L0']['se']:.6f} | {coefs['log_L0']['t']:.3f} | {coefs['log_L0']['pvalue']:.4f} | [{coefs['log_L0']['ci_lower']:.4f}, {coefs['log_L0']['ci_upper']:.4f}] |
| log(width) | {coefs['log_width']['value']:.6f} | {coefs['log_width']['se']:.6f} | {coefs['log_width']['t']:.3f} | {coefs['log_width']['pvalue']:.4f} | [{coefs['log_width']['ci_lower']:.4f}, {coefs['log_width']['ci_upper']:.4f}] |
| layer | {coefs['layer']['value']:.6f} | {coefs['layer']['se']:.6f} | {coefs['layer']['t']:.3f} | {coefs['layer']['pvalue']:.4f} | [{coefs['layer']['ci_lower']:.4f}, {coefs['layer']['ci_upper']:.4f}] |
| **log(PMI)** | **{coefs['log_PMI']['value']:.6f}** | {coefs['log_PMI']['se']:.6f} | {coefs['log_PMI']['t']:.3f} | **{coefs['log_PMI']['pvalue']:.4f}** | [{coefs['log_PMI']['ci_lower']:.4f}, {coefs['log_PMI']['ci_upper']:.4f}] |

| Metric | Value |
|--------|-------|
| R^2 | {full_results['r_squared']:.6f} |
| Adj R^2 | {full_results['adj_r_squared']:.6f} |
| F-statistic | {full_results['f_statistic']:.4f} |
| AIC | {full_results['aic']:.2f} |

## Partial R^2 for PMI

| Metric | Value |
|--------|-------|
| R^2 (full model) | {r2_full:.6f} |
| R^2 (without PMI) | {r2_restricted:.6f} |
| **Partial R^2 (PMI)** | **{partial_r2_pmi:.6f}** |
| Partial r (PMI \\| config) | {partial_r:.6f} |
| Partial p-value | {partial_p:.6f} |

## Correlation Analysis

| Measure | r | p-value |
|---------|---|---------|
| Raw Pearson (log_PMI vs absorption, all cells) | {raw_r:.4f} | {raw_p:.4f} |
| Raw Spearman (log_PMI vs absorption, all cells) | {raw_spearman:.4f} | {raw_sp_p:.4f} |
| Per-letter Pearson (mean absorption vs log_PMI) | {letter_r:.4f} | {letter_p:.4f} |
| Per-letter Spearman | {letter_spearman:.4f} | {letter_sp_p:.4f} |

## Ablation A3: PMI-Only Model

| Metric | Value |
|--------|-------|
| R^2 | {ablation_a3['r_squared']:.6f} |
| beta_PMI | {ablation_a3['beta_pmi']:.6f} |
| p-value | {ablation_a3['pvalue_pmi']:.4f} |

## Ablation A4: Per-Layer PMI Coefficient Stability

| Layer | beta_PMI | SE | p-value | Sign |
|-------|----------|-----|---------|------|
"""

for layer in sorted(per_layer_results.keys(), key=lambda x: int(x)):
    r = per_layer_results[layer]
    if r.get("beta_pmi") is not None:
        md += f"| {layer} | {r['beta_pmi']:.6f} | {r.get('se_pmi', 0):.6f} | {r.get('pvalue_pmi', 0):.4f} | {r.get('pmi_sign', 'N/A')} |\n"
    else:
        md += f"| {layer} | N/A | N/A | N/A | {r.get('note', 'N/A')} |\n"

md += f"""
**Sign consistency across layers**: {sign_consistent}

## Visualizations

- Partial regression plot: `C2C_plots/C2C_partial_regression_plot.png`
- Per-layer coefficient plot: `C2C_plots/C2C_per_layer_pmi_coef.png`
- Width effect at layer 8: `C2C_plots/C2C_width_effect_layer8.png`

## Key Findings

1. **PMI coefficient sign**: {coefs['log_PMI']['value']:.6f} ({'positive — higher PMI letters tend to show higher absorption' if coefs['log_PMI']['value'] > 0 else 'negative — contrary to H2 prediction'})
2. **Statistical significance**: p={coefs['log_PMI']['pvalue']:.4f} ({'significant at alpha=0.05' if coefs['log_PMI']['pvalue'] < 0.05 else 'not significant at alpha=0.05'})
3. **Partial R^2 for PMI**: {partial_r2_pmi:.4f} ({'meets criterion >= 0.10' if partial_r2_pmi >= 0.10 else 'below criterion of 0.10'})
4. **Layer coefficient**: {coefs['layer']['value']:.6f} ({'negative — absorption decreases in later layers' if coefs['layer']['value'] < 0 else 'positive — absorption increases in later layers'})
5. **Width coefficient**: {coefs['log_width']['value']:.6f} ({'positive — wider SAEs show more absorption' if coefs['log_width']['value'] > 0 else 'negative — wider SAEs show less absorption'})

## Caveats

- Model is GPT-2 Small (not Gemma 2 2B as originally planned due to gated access)
- PMI aggregation uses median of top-10 tokens per letter as proxy for letter-category co-occurrence pressure
- The 31 SAE configs span 3 release families (res_jb, feature_splitting, resid_post, resid_mid) with different training procedures; this is a potential confound
- HC3 SEs do not account for clustering at the letter level (26 repeated measurements per letter); clustered SEs would be more conservative
"""

with open(OUT_MD, "w") as f:
    f.write(md)
print(f"  Markdown summary saved to: {OUT_MD}")

# ═══════════════════════════════════════════════════════════════════════════
# Update gpu_progress.json
# ═══════════════════════════════════════════════════════════════════════════
print("\n  Updating gpu_progress.json...")
gpu_progress_file = WORKSPACE / "exp/gpu_progress.json"
try:
    with open(gpu_progress_file) as f:
        gp = json.load(f)
except Exception:
    gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

if "C2C_pmi_regression" not in gp["completed"]:
    gp["completed"].append("C2C_pmi_regression")

# Remove from running if present
gp["running"].pop("C2C_pmi_regression", None)

# Record timing
gp["timings"]["C2C_pmi_regression"] = {
    "planned_min": 30,
    "actual_min": int(runtime_seconds / 60) + 1,
    "start_time": start_time.isoformat(),
    "end_time": end_time.isoformat(),
    "config_snapshot": {
        "task_type": "cpu_regression_analysis",
        "n_observations": len(df),
        "n_configs": int(df["config_id"].nunique()),
        "n_letters": 26,
        "gpu_count": 0,
        "model": "gpt2-small",
    }
}

with open(gpu_progress_file, "w") as f:
    json.dump(gp, f, indent=2, cls=NumpyEncoder)
print(f"  gpu_progress.json updated")

# Mark done
mark_done(
    status="success",
    summary=(f"C2C full: R^2={r2_full:.4f}, partial_R^2_PMI={partial_r2_pmi:.4f}, "
             f"beta_PMI={coefs['log_PMI']['value']:.4f} (p={coefs['log_PMI']['pvalue']:.4f}), "
             f"n_obs={len(df)}")
)

# ═══════════════════════════════════════════════════════════════════════════
# Final Summary
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("C2C PMI REGRESSION — FULL ANALYSIS COMPLETE")
print("=" * 70)
print(f"  R^2 (full model): {r2_full:.4f}")
print(f"  Partial R^2 (PMI): {partial_r2_pmi:.4f}")
print(f"  beta_PMI: {coefs['log_PMI']['value']:.6f} (p={coefs['log_PMI']['pvalue']:.4f})")
print(f"  Per-letter Pearson r: {letter_r:.4f} (p={letter_p:.4f})")
print(f"  H2 Assessment: {h2['assessment']}")
print(f"  Runtime: {runtime_seconds:.1f}s")
print("=" * 70)
