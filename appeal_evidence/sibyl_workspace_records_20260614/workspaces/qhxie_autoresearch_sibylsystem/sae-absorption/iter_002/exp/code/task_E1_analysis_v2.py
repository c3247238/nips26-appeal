"""
Task E1 Analysis v2: Phase Transition Detection — Supplementary Analysis

This script combines:
1. B2 EDA-based absorption proxy data (frac_eda_gt70 as absorption_rate proxy)
2. E1 fresh direct-activation absorption measurements
3. Comprehensive curve fitting and LRT analysis
4. Bootstrap CI for inflection point

Key insight from E1 PILOT: direct activation measurement gives ceiling effect (~95%+)
because SAE features are sparse by design. The EDA proxy (fraction of letter features
with EDA > threshold) from B2 gives more discriminative signal.

This script performs the full E1 analysis using both metrics.
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats, optimize
from scipy.stats import chi2

warnings.filterwarnings("ignore")

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
TASK_ID = "task_E1_phase_transition"
OUTPUT_FILE = RESULTS_DIR / "E1_phase_transition.json"
SEED = 42
np.random.seed(SEED)
start_time = time.time()


def fit_linear(x, y):
    """Fit linear model y = a*x + b."""
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, se = linregress(x, y)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    n = len(x)
    k = 2
    ll = -0.5 * n * np.log(max(ss_res / n, 1e-12))
    bic = k * np.log(n) - 2 * ll
    aic = 2 * k - 2 * ll
    return {
        "slope": float(slope), "intercept": float(intercept),
        "r2": float(r2), "p_value": float(p_value),
        "bic": float(bic), "aic": float(aic), "ll": float(ll),
        "y_pred": y_pred.tolist()
    }


def fit_powerlaw(x, y):
    """Fit power-law y = a * x^b."""
    def powerlaw_func(x, a, b):
        return a * np.power(np.maximum(x, 1e-10), b)
    try:
        popt, _ = optimize.curve_fit(
            powerlaw_func, x, y, p0=[0.5, 0.5], maxfev=5000,
            bounds=([0, -5], [10, 10])
        )
        y_pred = powerlaw_func(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
        n = len(x)
        k = 2
        ll = -0.5 * n * np.log(max(ss_res / n, 1e-12))
        bic = k * np.log(n) - 2 * ll
        aic = 2 * k - 2 * ll
        return {
            "a": float(popt[0]), "b": float(popt[1]),
            "r2": float(r2), "bic": float(bic), "aic": float(aic),
            "ll": float(ll), "y_pred": y_pred.tolist()
        }
    except Exception as e:
        return {"error": str(e)}


def fit_sigmoid(x, y):
    """Fit sigmoid y = L / (1 + exp(-k*(x - x0))) + b."""
    def sigmoid_func(x, L, k, x0, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    best = None
    best_r2 = -np.inf
    x0_candidates = [np.median(x), np.mean(x),
                     np.percentile(x, 25), np.percentile(x, 75)]
    ss_tot = np.sum((y - y.mean()) ** 2)

    for x0_init in x0_candidates:
        try:
            popt, _ = optimize.curve_fit(
                sigmoid_func, x, y,
                p0=[y.max() - y.min(), 100.0, x0_init, y.min()],
                maxfev=10000,
                bounds=([-2, 0.1, 1e-6, -1], [2, 1e6, 1.0, 1])
            )
            y_pred = sigmoid_func(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-12)
            if r2 > best_r2:
                best_r2 = r2
                best = (popt, y_pred, ss_res, r2)
        except:
            pass

    if best is None:
        return {"error": "optimization failed"}

    popt, y_pred, ss_res, r2 = best
    n = len(x)
    k = 4
    ll = -0.5 * n * np.log(max(ss_res / n, 1e-12))
    bic = k * np.log(n) - 2 * ll
    aic = 2 * k - 2 * ll
    inflection_inv_l0 = float(popt[2])
    inflection_l0_c = float(1.0 / inflection_inv_l0) if inflection_inv_l0 > 1e-6 else None
    return {
        "L": float(popt[0]), "k": float(popt[1]),
        "x0": float(popt[2]), "b": float(popt[3]),
        "r2": float(r2), "bic": float(bic), "aic": float(aic), "ll": float(ll),
        "inflection_inv_l0": inflection_inv_l0,
        "inflection_l0_c": inflection_l0_c,
        "y_pred": y_pred.tolist()
    }


def lrt_analysis(fit_null, fit_alt, df_diff=2):
    """Likelihood ratio test: alt vs null."""
    ll_null = fit_null.get("ll", 0)
    ll_alt = fit_alt.get("ll", 0)
    stat = 2 * (ll_alt - ll_null)
    pval = float(1 - chi2.cdf(max(stat, 0), df=df_diff))
    bic_diff = fit_null.get("bic", 0) - fit_alt.get("bic", 0)
    return {
        "lrt_stat": float(stat),
        "lrt_pvalue": pval,
        "df": df_diff,
        "bic_diff": float(bic_diff),
        "alt_wins": bool(fit_alt.get("bic", 0) < fit_null.get("bic", 0)),
        "h4a_supported": bool(bic_diff >= 10 and pval < 0.05)
    }


def bootstrap_l0c(x, y, sigmoid_fit, n_boot=1000, seed=42):
    """Bootstrap CI for inflection point L0_c."""
    if "error" in sigmoid_fit or not sigmoid_fit.get("inflection_l0_c"):
        return {"skipped": "no valid sigmoid fit"}

    def sigmoid_func(x, L, k, x0, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    np.random.seed(seed)
    n = len(x)
    l0c_boots = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        xb, yb = x[idx], y[idx]
        if len(np.unique(xb)) < 3:
            continue
        try:
            popt, _ = optimize.curve_fit(
                sigmoid_func, xb, yb,
                p0=[0.3, 100.0, np.median(x), 0.2],
                maxfev=3000,
                bounds=([-2, 0.1, 1e-6, -1], [2, 1e6, 1.0, 1])
            )
            x0b = popt[2]
            if 1e-6 < x0b < 1.0:
                l0c = 1.0 / x0b
                if 1 < l0c < 1e6:
                    l0c_boots.append(l0c)
        except:
            pass

    if len(l0c_boots) < 20:
        return {"error": f"only {len(l0c_boots)} successful bootstrap samples"}

    l0c_boots = np.array(l0c_boots)
    p99 = np.percentile(l0c_boots, 99)
    l0c_clipped = l0c_boots[l0c_boots <= p99]
    return {
        "n_successful": int(len(l0c_boots)),
        "l0c_mean": float(np.mean(l0c_clipped)),
        "l0c_median": float(np.median(l0c_clipped)),
        "l0c_ci_low": float(np.percentile(l0c_clipped, 2.5)),
        "l0c_ci_high": float(np.percentile(l0c_clipped, 97.5)),
        "nominal_l0c": sigmoid_fit.get("inflection_l0_c")
    }


# ─── Load B2 data ────────────────────────────────────────────────────────────
print("Loading B2 scaling curve data...")
B2_FILE = WORKSPACE / "exp" / "results" / "full" / "B2_scaling_curve.json"
with open(B2_FILE) as f:
    b2_data = json.load(f)

# Load E1 pilot data
E1_FILE = OUTPUT_FILE
with open(E1_FILE) as f:
    e1_data = json.load(f)

# Build combined data from B2 (EDA-based) for phase transition analysis
# Using frac_eda_gt70 as absorption proxy (most discriminative from B2 analysis)
print("\nBuilding B2-based absorption proxy dataset (frac_eda_gt70)...")
b2_points = []
for r in b2_data.get("all_results", []):
    if r.get("status") != "success" or r.get("l0") is None:
        continue
    if r["l0"] <= 0:
        continue
    b2_points.append({
        "sae_id": r["sae_id"],
        "layer": r["layer"],
        "width": r["width"],
        "l0": r["l0"],
        "inv_l0": r["inv_l0"],
        "absorption_proxy_eda70": r["frac_eda_gt70"],
        "absorption_proxy_eda60": r["frac_eda_gt60"],
        "mean_eda_letter": r["mean_eda_letter"],
        "eda_delta": r["eda_delta"],
    })

print(f"  B2 points: {len(b2_points)}")
for p in b2_points:
    print(f"    layer={p['layer']}, w={p['width']}, L0={p['l0']:.1f}, "
          f"frac_eda70={p['absorption_proxy_eda70']:.3f}, "
          f"mean_eda={p['mean_eda_letter']:.4f}")

# Primary analysis: use B2 frac_eda_gt70 as absorption_rate proxy
# Filter to primary (gpt2-small-res-jb) and AJT configs only (exclude width-variation at same L0)
primary_points = [p for p in b2_points if p["layer"] in [2, 4, 6, 8, 10]]
print(f"\n  Primary data points: {len(primary_points)} (varying L0)")

# Full set (all configs)
all_x = np.array([p["inv_l0"] for p in b2_points])
all_y = np.array([p["absorption_proxy_eda70"] for p in b2_points])
all_y_eda60 = np.array([p["absorption_proxy_eda60"] for p in b2_points])

prim_x = np.array([p["inv_l0"] for p in primary_points])
prim_y = np.array([p["absorption_proxy_eda70"] for p in primary_points])

print(f"\n  All data: n={len(all_x)}, x=[{all_x.min():.4f},{all_x.max():.4f}], "
      f"y=[{all_y.min():.3f},{all_y.max():.3f}]")
print(f"  Primary: n={len(prim_x)}, x=[{prim_x.min():.4f},{prim_x.max():.4f}], "
      f"y=[{prim_y.min():.3f},{prim_y.max():.3f}]")

# ─── Curve fitting ───────────────────────────────────────────────────────────
print("\n=== Curve Fitting (B2 EDA-based proxy) ===")

analysis_results = {}

for dataset_name, x, y in [("all_b2_configs", all_x, all_y),
                             ("primary_only", prim_x, prim_y),
                             ("all_b2_configs_eda60", all_x, all_y_eda60)]:
    if len(x) < 3:
        continue
    print(f"\n  Dataset: {dataset_name} (n={len(x)})")

    lin_fit = fit_linear(x, y)
    pw_fit = fit_powerlaw(x, y)
    sig_fit = fit_sigmoid(x, y)

    print(f"    Linear: R2={lin_fit['r2']:.4f}, BIC={lin_fit['bic']:.3f}")
    print(f"    Power-law: R2={pw_fit.get('r2', 'N/A')}, BIC={pw_fit.get('bic', 'N/A')}")
    if "error" not in sig_fit:
        print(f"    Sigmoid: R2={sig_fit['r2']:.4f}, BIC={sig_fit['bic']:.3f}, "
              f"L0_c={sig_fit.get('inflection_l0_c', 'N/A')}")

    lrt = lrt_analysis(lin_fit, sig_fit) if "error" not in sig_fit else {}
    if lrt:
        print(f"    LRT (sigmoid vs linear): stat={lrt['lrt_stat']:.3f}, p={lrt['lrt_pvalue']:.4f}, "
              f"BIC_diff={lrt['bic_diff']:.2f}, H4a={lrt['h4a_supported']}")

    boot_ci = bootstrap_l0c(x, y, sig_fit, n_boot=500, seed=SEED) if "error" not in sig_fit else {}
    if boot_ci and "l0c_mean" in boot_ci:
        print(f"    Bootstrap L0_c: {boot_ci['l0c_mean']:.1f} "
              f"[{boot_ci['l0c_ci_low']:.1f}, {boot_ci['l0c_ci_high']:.1f}]")

    # Spearman
    rho, pval = stats.spearmanr(x, y)
    print(f"    Spearman(1/L0, absorption): rho={rho:.3f}, p={pval:.4f}")

    analysis_results[dataset_name] = {
        "n": len(x),
        "x_inv_l0": x.tolist(),
        "y_absorption_proxy": y.tolist(),
        "curve_fits": {
            "linear": lin_fit,
            "power_law": pw_fit,
            "sigmoid": sig_fit
        },
        "lrt": lrt,
        "bootstrap_ci": boot_ci,
        "spearman": {"rho": float(rho), "p": float(pval)}
    }

# ─── Width analysis ──────────────────────────────────────────────────────────
print("\n=== Width Analysis ===")

# All-width comparison at similar L0
width_data = {}
for p in b2_points:
    w = p["width"]
    if w not in width_data:
        width_data[w] = []
    width_data[w].append(p)

width_summary = []
for w in sorted(width_data.keys()):
    pts = width_data[w]
    rates = [p["absorption_proxy_eda70"] for p in pts]
    l0s = [p["l0"] for p in pts]
    width_summary.append({
        "width": w,
        "log2_width": float(np.log2(w)),
        "mean_absorption_eda70": float(np.mean(rates)),
        "mean_l0": float(np.mean(l0s)),
        "n_configs": len(pts)
    })
    print(f"  w={w} (log2={np.log2(w):.2f}): "
          f"mean_eda70={np.mean(rates):.3f}, n={len(pts)}")

# Width vs critical L0 prediction:
# Theory: wider SAEs need lower L0_c. Check correlation.
widths = np.array([p["width"] for p in b2_points])
eda70s = np.array([p["absorption_proxy_eda70"] for p in b2_points])
l0s_arr = np.array([p["l0"] for p in b2_points])

rho_w, pval_w = stats.spearmanr(np.log2(widths), eda70s)
rho_l0, pval_l0 = stats.spearmanr(l0s_arr, eda70s)
print(f"\n  Spearman(log2_width, eda70): rho={rho_w:.3f}, p={pval_w:.4f}")
print(f"  Spearman(L0, eda70): rho={rho_l0:.3f}, p={pval_l0:.4f}")

# ─── Direct absorption rate analysis (E1 pilot data) ─────────────────────────
print("\n=== Direct Absorption Rate Analysis (E1 Pilot) ===")
print("NOTE: Direct activation measurement gives ceiling effect (~95%+)")
print("This is expected: SAEs are sparse by design, most features don't fire for most inputs.")
print("The EDA proxy from B2 is more discriminative for absorption detection.")
e1_valid = e1_data.get("valid_results_summary", [])
print(f"\nE1 direct measurements (n={len(e1_valid)}):")
for r in e1_valid:
    print(f"  layer={r['layer']}, w={r['width']}, L0={r['l0']:.1f}, "
          f"absorption_direct={r['absorption_rate']:.4f}")

e1_spearman = e1_data.get("spearman", {}).get("all", {})
print(f"\nE1 Spearman(1/L0, direct_absorption): "
      f"rho={e1_spearman.get('rho', 'N/A'):.3f}, p={e1_spearman.get('p', 'N/A'):.4f}")

# ─── Determine final H4a support ─────────────────────────────────────────────
print("\n=== H4a Assessment ===")
primary_lrt = analysis_results.get("primary_only", {}).get("lrt", {})
all_lrt = analysis_results.get("all_b2_configs", {}).get("lrt", {})

h4a_primary = primary_lrt.get("h4a_supported", False)
h4a_all = all_lrt.get("h4a_supported", False)
print(f"  Primary configs only (n={len(primary_points)}): H4a={h4a_primary}")
print(f"  All configs (n={len(b2_points)}): H4a={h4a_all}")

# Directional signal even if not passing full threshold
primary_sig_fit = analysis_results.get("primary_only", {}).get("curve_fits", {}).get("sigmoid", {})
primary_r2_sig = primary_sig_fit.get("r2", 0)
primary_r2_lin = analysis_results.get("primary_only", {}).get("curve_fits", {}).get("linear", {}).get("r2", 0)
print(f"  Primary R2: sigmoid={primary_r2_sig:.4f} vs linear={primary_r2_lin:.4f}")

h4a_final = "NOT SUPPORTED"
h4a_notes = []
if h4a_primary or h4a_all:
    h4a_final = "SUPPORTED"
    h4a_notes.append("BIC diff >= 10 and LRT p < 0.05")
elif primary_r2_sig > primary_r2_lin + 0.1:
    h4a_final = "WEAK DIRECTIONAL EVIDENCE"
    h4a_notes.append(f"Sigmoid R2 better by {primary_r2_sig - primary_r2_lin:.3f} but BIC/LRT not significant")
else:
    h4a_notes.append("Sigmoid fit not superior to linear; absorption appears uniform across L0 range")
    # Check if this is a ceiling/floor effect
    y_range = all_y.max() - all_y.min()
    if y_range < 0.3:
        h4a_notes.append(f"Note: absorption proxy range is narrow ({y_range:.3f}), possible ceiling/floor effect")

print(f"\n  H4a final: {h4a_final}")
for note in h4a_notes:
    print(f"    {note}")

# ─── Save updated E1 results ──────────────────────────────────────────────────
print("\nUpdating E1 results file...")

# Load existing E1 results
with open(OUTPUT_FILE) as f:
    e1_existing = json.load(f)

# Add supplementary EDA-based analysis
e1_existing["eda_based_analysis"] = analysis_results
e1_existing["width_analysis_eda"] = {
    "width_summary": width_summary,
    "spearman_log2width_vs_eda70": {"rho": float(rho_w), "p": float(pval_w)},
    "spearman_l0_vs_eda70": {"rho": float(rho_l0), "p": float(pval_l0)}
}
e1_existing["h4a_final_assessment"] = {
    "h4a_result": h4a_final,
    "h4a_notes": h4a_notes,
    "primary_lrt": primary_lrt,
    "all_configs_lrt": all_lrt,
    "direct_absorption_ceiling_effect": True,
    "eda_proxy_range": float(all_y.max() - all_y.min()),
}
e1_existing["b2_data_points"] = b2_points
e1_existing["n_b2_points"] = len(b2_points)

# Update summary
summary_str = (
    f"E1 FINAL: B2-based EDA proxy (n={len(b2_points)}): "
    f"Spearman(1/L0, eda70)=rho={stats.spearmanr(all_x, all_y)[0]:.3f}, "
    f"p={stats.spearmanr(all_x, all_y)[1]:.4f}. "
    f"H4a: {h4a_final}. "
    f"Direct activation (E1 pilot n={len(e1_valid)}): ceiling effect (rates ~95%+). "
    f"pilot_pass={e1_existing.get('pilot_pass', True)}."
)
e1_existing["primary_finding"] = summary_str
e1_existing["timestamp"] = datetime.now().isoformat()
e1_existing["elapsed_sec"] = time.time() - start_time

OUTPUT_FILE.write_text(json.dumps(e1_existing, indent=2))
print(f"Updated {OUTPUT_FILE}")
print(f"\n=== E1 ANALYSIS COMPLETE ===")
print(f"Summary: {summary_str}")
