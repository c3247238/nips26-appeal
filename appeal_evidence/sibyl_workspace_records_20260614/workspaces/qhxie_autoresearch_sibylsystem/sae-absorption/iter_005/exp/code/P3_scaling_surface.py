#!/usr/bin/env python3
"""
P3_scaling_surface.py — Absorption Scaling Surface from SAEBench Data

PILOT mode: Load precomputed SAEBench absorption results, extract absorption score,
L0, width, layer for 200+ SAEs. Fit linear, additive, and interaction GAM models.
Generate 2D contour plot. Detect phase boundaries via gradient analysis.

Pass criteria:
- SAEBench data loads with N >= 100 SAEs having both absorption and L0
- Linear model R^2 > 0.15
"""

import json
import os
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

# ---- Config ----
WORKSPACE = "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current"
TASK_ID = "P3_scaling_surface"
RESULTS_DIR = Path(WORKSPACE) / "exp" / "results"
PILOTS_DIR = RESULTS_DIR / "pilots"
FULL_DIR = RESULTS_DIR / "full"

SAEBENCH_CACHE = Path.home() / ".cache/huggingface/hub/datasets--adamkarvonen--sae_bench_results/snapshots/dd8bf8b11fb1a4a3b5106e880c31dd164f69879d"

SEED = 42
np.random.seed(SEED)

# PID file for system recovery
PILOTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

start_time = datetime.now()


def report_progress(task_id, results_dir, epoch, total_epochs, step=0,
                    total_steps=0, loss=None, metric=None):
    """Write progress file for system monitor."""
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id, results_dir, status="success", summary=""):
    """Write DONE marker file."""
    pid_f = Path(results_dir) / f"{task_id}.pid"
    if pid_f.exists():
        pid_f.unlink()
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


# ============================================================
# Step 1: Load and Parse SAEBench Absorption Results
# ============================================================
print("=" * 60)
print("STEP 1: Loading SAEBench absorption results from cache")
print("=" * 60)

report_progress(TASK_ID, RESULTS_DIR, 1, 5, step=0, total_steps=4,
                metric={"phase": "data_loading"})

records = []

# Walk through all absorption result directories
absorption_dir = SAEBENCH_CACHE / "absorption"
core_dir = SAEBENCH_CACHE / "core"

# Only process Gemma 2B SAEs (consistent model)
gemma_2b_releases = []
for d in sorted(absorption_dir.iterdir()):
    if d.is_dir() and ("gemma-2-2b" in d.name or "gemma-scope-2b" in d.name):
        # Exclude 9B
        if "9b" not in d.name:
            gemma_2b_releases.append(d.name)

print(f"Found {len(gemma_2b_releases)} Gemma 2B releases: {gemma_2b_releases}")

for release_name in gemma_2b_releases:
    release_abs_dir = absorption_dir / release_name
    release_core_dir = core_dir / release_name

    for abs_file in sorted(release_abs_dir.glob("*_eval_results.json")):
        try:
            with open(abs_file) as f:
                abs_data = json.load(f)

            # Get absorption score
            absorption_score = abs_data.get("eval_result_metrics", {}).get(
                "mean", {}
            ).get("mean_absorption_score", None)
            if absorption_score is None:
                continue

            # Try to find matching core file for L0
            core_file = release_core_dir / abs_file.name
            l0 = None
            d_sae = None
            architecture = None
            layer = None
            sae_lens_id = None

            if core_file.exists():
                with open(core_file) as f:
                    core_data = json.load(f)
                l0 = core_data.get("eval_result_metrics", {}).get(
                    "sparsity", {}
                ).get("l0", None)
                cfg = core_data.get("sae_cfg_dict", {})
                d_sae = cfg.get("d_sae", None)
                architecture = cfg.get("architecture", None)
                sae_lens_id = core_data.get("sae_lens_id", None)

            # Parse layer from filename
            fname = abs_file.stem
            # Pattern 1: gemma-scope-2b-pt-res_layer_5_width_1m_average_l0_9
            layer_match = re.search(r"layer[_.](\d+)", fname)
            if layer_match:
                layer = int(layer_match.group(1))
            # Pattern 2: blocks.5.hook_resid_post
            blocks_match = re.search(r"blocks\.(\d+)\.", fname)
            if blocks_match and layer is None:
                layer = int(blocks_match.group(1))

            # Parse width from filename or metadata
            width = d_sae
            if width is None:
                # Try to get from filename: width_1m, width_16k, width_65k
                w_match = re.search(r"width[_-](\d+[km]?)", fname)
                if w_match:
                    w_str = w_match.group(1).lower()
                    if w_str.endswith("m"):
                        width = int(w_str[:-1]) * 1_000_000
                    elif w_str.endswith("k"):
                        width = int(w_str[:-1]) * 1_000
                    else:
                        width = int(w_str)
                # width-2pow14 => 2^14 = 16384
                pow_match = re.search(r"width-2pow(\d+)", fname)
                if pow_match:
                    width = 2 ** int(pow_match.group(1))

            # Parse L0 from filename if not from core
            if l0 is None:
                l0_match = re.search(r"average_l0[_-](\d+)", fname)
                if l0_match:
                    l0 = float(l0_match.group(1))

            records.append({
                "sae_id": fname,
                "release": release_name,
                "layer": layer,
                "width": width,
                "l0": l0,
                "architecture": architecture or "unknown",
                "absorption_score": absorption_score,
                "sae_lens_id": sae_lens_id,
            })

        except Exception as e:
            print(f"  Warning: failed to parse {abs_file.name}: {e}")
            continue

print(f"\nTotal records parsed: {len(records)}")

df = pd.DataFrame(records)
print(f"Columns: {list(df.columns)}")
print(f"Non-null L0: {df['l0'].notna().sum()}")
print(f"Non-null width: {df['width'].notna().sum()}")
print(f"Non-null layer: {df['layer'].notna().sum()}")

# Filter to records with both absorption and L0
df_full = df.dropna(subset=["absorption_score", "l0", "width", "layer"]).copy()
df_full = df_full[df_full["l0"] > 0].copy()  # L0 must be positive for log
df_full["log_width"] = np.log2(df_full["width"])
df_full["log_l0"] = np.log2(df_full["l0"])

print(f"\nFiltered dataset: {len(df_full)} SAEs with absorption, L0, width, layer")
print(f"Width range: {df_full['width'].min():.0f} - {df_full['width'].max():.0f}")
print(f"L0 range: {df_full['l0'].min():.1f} - {df_full['l0'].max():.1f}")
print(f"Layer range: {df_full['layer'].min()} - {df_full['layer'].max()}")
print(f"Absorption range: {df_full['absorption_score'].min():.4f} - {df_full['absorption_score'].max():.4f}")

# Architecture breakdown
print("\nArchitecture breakdown:")
print(df_full.groupby("architecture").size())

# Width x L0 breakdown
print("\nWidth x Layer breakdown:")
print(df_full.groupby(["width", "layer"]).size().unstack(fill_value=0))

# ============================================================
# Step 2: Fit Models
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Fitting linear, additive, and interaction models")
print("=" * 60)

report_progress(TASK_ID, RESULTS_DIR, 2, 5, step=1, total_steps=4,
                metric={"phase": "model_fitting", "n_saes": len(df_full)})

# --- Model 1: Linear model ---
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X_linear = df_full[["log_width", "log_l0", "layer"]].values
y = df_full["absorption_score"].values

lr = LinearRegression()
lr.fit(X_linear, y)
y_pred_linear = lr.predict(X_linear)
r2_linear = r2_score(y, y_pred_linear)
print(f"\nLinear model R^2: {r2_linear:.4f}")
print(f"  Coefficients: log_width={lr.coef_[0]:.4f}, log_l0={lr.coef_[1]:.4f}, layer={lr.coef_[2]:.4f}")
print(f"  Intercept: {lr.intercept_:.4f}")

# Compute AIC for linear model (OLS approximation)
n = len(y)
k_linear = 4  # 3 predictors + intercept
rss_linear = np.sum((y - y_pred_linear) ** 2)
aic_linear = n * np.log(rss_linear / n) + 2 * k_linear
print(f"  AIC: {aic_linear:.1f}")

# --- Model 2: Additive GAM (no interaction) ---
try:
    from pygam import LinearGAM, s, te, f

    # Additive GAM: s(log_width) + s(log_l0) + layer (linear)
    gam_additive = LinearGAM(s(0, n_splines=10) + s(1, n_splines=10) + s(2, n_splines=5))
    X_gam = df_full[["log_width", "log_l0", "layer"]].values

    gam_additive.gridsearch(X_gam, y, progress=False)
    y_pred_additive = gam_additive.predict(X_gam)
    r2_additive = r2_score(y, y_pred_additive)

    # GAM statistics
    gam_add_summary = gam_additive.statistics_
    aic_additive = gam_add_summary.get("AIC", None)
    if aic_additive is None:
        rss_add = np.sum((y - y_pred_additive) ** 2)
        edf_add = gam_add_summary.get("edof", 10)
        aic_additive = n * np.log(rss_add / n) + 2 * edf_add

    print(f"\nAdditive GAM R^2: {r2_additive:.4f}")
    print(f"  AIC: {aic_additive:.1f}")
    print(f"  EDoF: {gam_add_summary.get('edof', 'N/A')}")

    # --- Model 3: Interaction GAM ---
    # Use tensor product for log_width x log_l0 interaction
    gam_interaction = LinearGAM(
        s(0, n_splines=10) + s(1, n_splines=10) + s(2, n_splines=5) + te(0, 1, n_splines=[5, 5])
    )
    gam_interaction.gridsearch(X_gam, y, progress=False)
    y_pred_interaction = gam_interaction.predict(X_gam)
    r2_interaction = r2_score(y, y_pred_interaction)

    gam_int_summary = gam_interaction.statistics_
    aic_interaction = gam_int_summary.get("AIC", None)
    if aic_interaction is None:
        rss_int = np.sum((y - y_pred_interaction) ** 2)
        edf_int = gam_int_summary.get("edof", 20)
        aic_interaction = n * np.log(rss_int / n) + 2 * edf_int

    print(f"\nInteraction GAM R^2: {r2_interaction:.4f}")
    print(f"  AIC: {aic_interaction:.1f}")
    print(f"  EDoF: {gam_int_summary.get('edof', 'N/A')}")

    # --- Test interaction significance ---
    # Compare additive vs interaction model using deviance test
    rss_add = np.sum((y - y_pred_additive) ** 2)
    rss_int = np.sum((y - y_pred_interaction) ** 2)
    edf_add = gam_add_summary.get("edof", 10)
    edf_int = gam_int_summary.get("edof", 20)

    if edf_int > edf_add and rss_add > rss_int:
        f_stat = ((rss_add - rss_int) / (edf_int - edf_add)) / (rss_int / (n - edf_int))
        p_interaction = 1 - stats.f.cdf(f_stat, edf_int - edf_add, n - edf_int)
    else:
        f_stat = 0
        p_interaction = 1.0

    print(f"\nInteraction test: F={f_stat:.3f}, p={p_interaction:.6f}")
    print(f"  Interaction significant (p < 0.05): {p_interaction < 0.05}")

    gam_fitted = True

except Exception as e:
    print(f"\nGAM fitting failed: {e}")
    print("Falling back to polynomial interaction model...")
    gam_fitted = False

    # Fallback: polynomial with interaction
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    X_int = df_full[["log_width", "log_l0", "layer"]].values
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X_int)
    lr_int = LinearRegression().fit(X_poly, y)
    y_pred_interaction = lr_int.predict(X_poly)
    r2_interaction = r2_score(y, y_pred_interaction)
    k_int = X_poly.shape[1] + 1
    rss_int = np.sum((y - y_pred_interaction) ** 2)
    aic_interaction = n * np.log(rss_int / n) + 2 * k_int

    r2_additive = None
    aic_additive = None
    p_interaction = None

    print(f"Polynomial interaction R^2: {r2_interaction:.4f}")

# ============================================================
# Step 3: Generate 2D Contour Plot
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Generating contour plots")
print("=" * 60)

report_progress(TASK_ID, RESULTS_DIR, 3, 5, step=2, total_steps=4,
                metric={"phase": "visualization"})

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Plot 1: Raw scatter with absorption score ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plot colored by absorption score
sc = axes[0].scatter(
    df_full["log_width"], df_full["log_l0"],
    c=df_full["absorption_score"], cmap="RdYlBu_r",
    s=40, alpha=0.7, edgecolors="k", linewidth=0.5
)
plt.colorbar(sc, ax=axes[0], label="Absorption Score")
axes[0].set_xlabel("log2(width)")
axes[0].set_ylabel("log2(L0)")
axes[0].set_title("Absorption Score in (width, L0) Space")

# --- Plot 2: GAM predicted surface contour ---
if gam_fitted:
    # Create grid for prediction
    w_range = np.linspace(df_full["log_width"].min() - 0.5, df_full["log_width"].max() + 0.5, 100)
    l0_range = np.linspace(df_full["log_l0"].min() - 0.5, df_full["log_l0"].max() + 0.5, 100)
    W_grid, L0_grid = np.meshgrid(w_range, l0_range)

    # Use median layer for surface prediction
    median_layer = df_full["layer"].median()
    X_grid = np.column_stack([W_grid.ravel(), L0_grid.ravel(),
                               np.full(W_grid.size, median_layer)])
    Z_pred = gam_interaction.predict(X_grid).reshape(W_grid.shape)

    # Contour plot
    contour = axes[1].contourf(W_grid, L0_grid, Z_pred, levels=20, cmap="RdYlBu_r")
    plt.colorbar(contour, ax=axes[1], label="Predicted Absorption")
    axes[1].contour(W_grid, L0_grid, Z_pred, levels=10, colors="k", linewidths=0.5, alpha=0.3)

    # Overlay data points
    axes[1].scatter(
        df_full["log_width"], df_full["log_l0"],
        c="white", s=15, alpha=0.6, edgecolors="k", linewidth=0.3
    )
    axes[1].set_xlabel("log2(width)")
    axes[1].set_ylabel("log2(L0)")
    axes[1].set_title(f"GAM Interaction Surface (layer={median_layer:.0f})")
else:
    axes[1].text(0.5, 0.5, "GAM fitting failed\nUsing polynomial fallback",
                 ha="center", va="center", transform=axes[1].transAxes, fontsize=14)

plt.tight_layout()
fig.savefig(PILOTS_DIR / "P3_absorption_contour.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved absorption contour plot")

# ============================================================
# Step 4: Phase Boundary Detection via Gradient Analysis
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Phase boundary detection")
print("=" * 60)

report_progress(TASK_ID, RESULTS_DIR, 4, 5, step=3, total_steps=4,
                metric={"phase": "gradient_analysis"})

gradient_results = {}

if gam_fitted:
    # Compute gradient of the GAM surface
    # Numerical differentiation
    dw = w_range[1] - w_range[0]
    dl0 = l0_range[1] - l0_range[0]

    # Smooth the predicted surface slightly for gradient stability
    Z_smooth = gaussian_filter(Z_pred, sigma=2.0)

    grad_w = np.gradient(Z_smooth, dw, axis=1)
    grad_l0 = np.gradient(Z_smooth, dl0, axis=0)
    grad_magnitude = np.sqrt(grad_w**2 + grad_l0**2)

    # Find ridge (regions of maximal gradient)
    max_grad = grad_magnitude.max()
    ridge_threshold = max_grad * 0.7
    ridge_mask = grad_magnitude > ridge_threshold

    # Find ridge coordinates
    ridge_indices = np.argwhere(ridge_mask)
    if len(ridge_indices) > 0:
        ridge_w = w_range[ridge_indices[:, 1]]
        ridge_l0 = l0_range[ridge_indices[:, 0]]
        print(f"Phase boundary detected: {len(ridge_indices)} ridge points")
        print(f"  Width range: log2(width) in [{ridge_w.min():.2f}, {ridge_w.max():.2f}]")
        print(f"  L0 range: log2(L0) in [{ridge_l0.min():.2f}, {ridge_l0.max():.2f}]")
        gradient_results = {
            "phase_boundary_detected": True,
            "n_ridge_points": int(len(ridge_indices)),
            "ridge_log_width_range": [float(ridge_w.min()), float(ridge_w.max())],
            "ridge_log_l0_range": [float(ridge_l0.min()), float(ridge_l0.max())],
            "max_gradient_magnitude": float(max_grad),
            "ridge_threshold": float(ridge_threshold),
        }
    else:
        print("No sharp phase boundary detected (smooth scaling)")
        gradient_results = {
            "phase_boundary_detected": False,
            "max_gradient_magnitude": float(max_grad),
        }

    # Plot gradient magnitude surface
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    contour = ax.contourf(W_grid, L0_grid, grad_magnitude, levels=20, cmap="hot_r")
    plt.colorbar(contour, ax=ax, label="Gradient Magnitude")
    if len(ridge_indices) > 0:
        ax.scatter(ridge_w, ridge_l0, c="cyan", s=5, alpha=0.5, label="Phase boundary")
        ax.legend()
    ax.scatter(
        df_full["log_width"], df_full["log_l0"],
        c="white", s=15, alpha=0.4, edgecolors="k", linewidth=0.3
    )
    ax.set_xlabel("log2(width)")
    ax.set_ylabel("log2(L0)")
    ax.set_title("Gradient Magnitude of Absorption Surface")
    plt.tight_layout()
    fig.savefig(PILOTS_DIR / "P3_gradient_surface.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved gradient surface plot")

else:
    gradient_results = {"phase_boundary_detected": False, "note": "GAM fitting failed"}

# ============================================================
# Step 5: Per-layer analysis
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Per-layer analysis")
print("=" * 60)

layer_results = []
for layer_val in sorted(df_full["layer"].unique()):
    mask = df_full["layer"] == layer_val
    sub = df_full[mask]
    if len(sub) < 5:
        continue

    # Spearman correlations
    r_width, p_width = stats.spearmanr(sub["log_width"], sub["absorption_score"])
    r_l0, p_l0 = stats.spearmanr(sub["log_l0"], sub["absorption_score"])

    layer_results.append({
        "layer": int(layer_val),
        "n": int(len(sub)),
        "absorption_mean": float(sub["absorption_score"].mean()),
        "absorption_std": float(sub["absorption_score"].std()),
        "spearman_width": float(r_width),
        "spearman_width_p": float(p_width),
        "spearman_l0": float(r_l0),
        "spearman_l0_p": float(p_l0),
    })
    print(f"  Layer {layer_val:2d}: n={len(sub):3d}, absorption={sub['absorption_score'].mean():.4f}+/-{sub['absorption_score'].std():.4f}, "
          f"rho(width)={r_width:.3f} (p={p_width:.4f}), rho(L0)={r_l0:.3f} (p={p_l0:.4f})")

# ============================================================
# Compile Results
# ============================================================
print("\n" + "=" * 60)
print("COMPILING RESULTS")
print("=" * 60)

end_time = datetime.now()
runtime_sec = (end_time - start_time).total_seconds()

# Model comparison summary
model_comparison = {
    "linear": {
        "r_squared": float(r2_linear),
        "aic": float(aic_linear),
        "coefficients": {
            "log_width": float(lr.coef_[0]),
            "log_l0": float(lr.coef_[1]),
            "layer": float(lr.coef_[2]),
        },
        "intercept": float(lr.intercept_),
    },
}

if r2_additive is not None:
    model_comparison["additive_gam"] = {
        "r_squared": float(r2_additive),
        "aic": float(aic_additive) if aic_additive else None,
    }

model_comparison["interaction_gam" if gam_fitted else "polynomial_interaction"] = {
    "r_squared": float(r2_interaction),
    "aic": float(aic_interaction) if aic_interaction else None,
    "interaction_p_value": float(p_interaction) if p_interaction is not None else None,
    "interaction_significant": bool(p_interaction < 0.05) if p_interaction is not None else None,
}

# Descriptive statistics
desc_stats = {
    "n_total_parsed": len(records),
    "n_with_absorption_l0_width_layer": len(df_full),
    "width_values": sorted(df_full["width"].unique().tolist()),
    "width_range": [float(df_full["width"].min()), float(df_full["width"].max())],
    "l0_range": [float(df_full["l0"].min()), float(df_full["l0"].max())],
    "layer_range": [int(df_full["layer"].min()), int(df_full["layer"].max())],
    "absorption_range": [float(df_full["absorption_score"].min()),
                          float(df_full["absorption_score"].max())],
    "absorption_mean": float(df_full["absorption_score"].mean()),
    "absorption_std": float(df_full["absorption_score"].std()),
    "architecture_breakdown": df_full.groupby("architecture").size().to_dict(),
    "release_breakdown": df_full.groupby("release").size().to_dict(),
}

# Width-L0 breakdown
width_l0_breakdown = []
for (w, l), group in df_full.groupby(["width", "layer"]):
    width_l0_breakdown.append({
        "width": int(w),
        "layer": int(l),
        "n": int(len(group)),
        "l0_range": [float(group["l0"].min()), float(group["l0"].max())],
        "absorption_mean": float(group["absorption_score"].mean()),
    })

# Pilot pass criteria
pilot_pass = len(df_full) >= 100 and r2_linear > 0.15

results = {
    "task_id": TASK_ID,
    "mode": "pilot",
    "timestamp": time.time(),
    "runtime_seconds": runtime_sec,
    "pilot_pass": pilot_pass,
    "pilot_criteria": {
        "n_saes_with_absorption_and_l0": len(df_full),
        "n_threshold": 100,
        "n_pass": len(df_full) >= 100,
        "linear_r_squared": float(r2_linear),
        "r_squared_threshold": 0.15,
        "r_squared_pass": r2_linear > 0.15,
    },
    "descriptive_statistics": desc_stats,
    "model_comparison": model_comparison,
    "gradient_analysis": gradient_results,
    "per_layer_analysis": layer_results,
    "width_l0_breakdown": width_l0_breakdown,
    "visualizations": {
        "contour_plot": str(PILOTS_DIR / "P3_absorption_contour.png"),
        "gradient_plot": str(PILOTS_DIR / "P3_gradient_surface.png"),
    },
}

# Save results
output_path = PILOTS_DIR / "P3_scaling_surface.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

# Also save full results path
full_output_path = FULL_DIR / "P3_scaling_surface.json"
with open(full_output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"Results also saved to {full_output_path}")

# Print summary
print("\n" + "=" * 60)
print("PILOT SUMMARY")
print("=" * 60)
print(f"N SAEs loaded: {len(df_full)} (threshold: >= 100)")
print(f"Linear R^2: {r2_linear:.4f} (threshold: > 0.15)")
if r2_additive is not None:
    print(f"Additive GAM R^2: {r2_additive:.4f}")
print(f"Interaction R^2: {r2_interaction:.4f}")
if p_interaction is not None:
    print(f"Interaction p-value: {p_interaction:.6f}")
print(f"Phase boundary detected: {gradient_results.get('phase_boundary_detected', False)}")
print(f"PILOT {'PASS' if pilot_pass else 'FAIL'}")
print(f"Runtime: {runtime_sec:.1f}s")

# Mark task done
mark_task_done(TASK_ID, RESULTS_DIR,
               status="success" if pilot_pass else "fail",
               summary=f"N={len(df_full)} SAEs, linear R^2={r2_linear:.4f}, interaction R^2={r2_interaction:.4f}, pilot={'PASS' if pilot_pass else 'FAIL'}")

# Update gpu_progress.json
gpu_progress_path = Path(WORKSPACE) / "exp" / "gpu_progress.json"
try:
    with open(gpu_progress_path) as f:
        gp = json.load(f)
except:
    gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

if pilot_pass:
    if TASK_ID not in gp["completed"]:
        gp["completed"].append(TASK_ID)
else:
    if TASK_ID not in gp["failed"]:
        gp["failed"].append(TASK_ID)

# Remove from running
gp["running"].pop(TASK_ID, None)

# Record timing
gp["timings"][TASK_ID] = {
    "planned_min": 45,
    "actual_min": int(runtime_sec / 60) + 1,
    "start_time": start_time.isoformat(),
    "end_time": end_time.isoformat(),
    "config_snapshot": {
        "n_saes": len(df_full),
        "model_types": ["linear", "additive_gam", "interaction_gam"] if gam_fitted else ["linear", "polynomial"],
        "gpu_count": 0,
    },
}

with open(gpu_progress_path, "w") as f:
    json.dump(gp, f, indent=2)
print(f"\nUpdated gpu_progress.json")
