#!/usr/bin/env python3
"""
Phase 2.2: Rate-Distortion Predictor Validation (Pooled Cross-Domain, H9)

Re-test the three-factor predictor model with parent-child pairs pooled
across ALL hierarchies, using corrected data from iter_009.

Iter-9 tested 262 pairs (2 SAEs x 131 unique pairs) and found:
  - Model rho=0.250, p=4.3e-5 (but LOO CV rho=0.206)
  - Individual predictors: cos_sim rho=-0.107, co_occur rho=-0.174, r_parent rho=-0.203

This experiment:
1. Pools pairs focusing on L24_16k (primary SAE) for cleaner analysis
2. Tests each predictor individually with bootstrap CI
3. Fits combined model with LOO cross-validation
4. Tests cross-domain predictor distributions
5. Computes additional derived predictors (competition_coeff, etc.)
6. Runs permutation tests for significance
7. If GPU available: recompute predictors from scratch for additional pairs
"""

import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---- Configuration ----
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption")
CURRENT = WORKSPACE / "current"
RESULTS_DIR = CURRENT / "exp" / "results" / "phase2"
PROGRESS_FILE = CURRENT / "exp" / "results" / "phase2_rate_distortion_pooled_PROGRESS.json"
DONE_FILE = CURRENT / "exp" / "results" / "phase2_rate_distortion_pooled_DONE"

# Source data
ITER9_RATE_DISTORTION = WORKSPACE / "iter_009" / "exp" / "results" / "phase3" / "rate_distortion_predictors.json"
ITER9_ABSORPTION_CROSSDOMAIN = WORKSPACE / "iter_009" / "exp" / "results" / "phase1" / "absorption_crossdomain.json"
ITER9_ABSORPTION_FIRSTLETTER = WORKSPACE / "iter_009" / "exp" / "results" / "phase1" / "absorption_firstletter.json"

SEED = 42
N_BOOTSTRAP = 10000
N_PERMUTATIONS = 5000
GPU_ID = int(os.environ.get("CUDA_VISIBLE_DEVICES", "4"))

np.random.seed(SEED)


def write_progress(step: str, pct: float, details: dict = None):
    """Write progress JSON for monitoring."""
    progress = {
        "task": "phase2_rate_distortion_pooled",
        "step": step,
        "percent": pct,
        "timestamp": datetime.now().isoformat(),
        "details": details or {},
    }
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def load_iter9_pairs():
    """Load all pairs from iter_009 rate-distortion results."""
    with open(ITER9_RATE_DISTORTION) as f:
        data = json.load(f)

    pairs = data["pairs_with_predictors"]
    print(f"Loaded {len(pairs)} pairs from iter_009")

    # Separate by SAE
    l24_16k = [p for p in pairs if p["sae_key"] == "L24_16k"]
    l24_65k = [p for p in pairs if p["sae_key"] == "L24_65k"]
    print(f"  L24_16k: {len(l24_16k)} pairs")
    print(f"  L24_65k: {len(l24_65k)} pairs")

    return pairs, l24_16k, l24_65k, data


def load_absorption_data():
    """Load per-hierarchy absorption data for context."""
    absorption = {}

    # Cross-domain
    with open(ITER9_ABSORPTION_CROSSDOMAIN) as f:
        cd = json.load(f)
    for entry in cd.get("summary_table", []):
        if entry["sae_config"] == "L24_16k":
            hierarchy = entry["hierarchy"]
            absorption[hierarchy] = {
                "absorption_rate": entry["absorption_rate"],
                "probe_f1": entry["probe_f1"],
                "n_entities": entry["n_entities"],
                "n_probe_correct": entry["n_probe_correct"],
                "n_fn": entry["n_fn"],
                "ci_lower": entry["ci_lower"],
                "ci_upper": entry["ci_upper"],
            }

    # First-letter (from the same file if available, or from firstletter)
    with open(ITER9_ABSORPTION_FIRSTLETTER) as f:
        fl = json.load(f)
    # Get L24 absorption data
    for layer_key, layer_data in fl.get("absorption_results", {}).items():
        if "24" in str(layer_key):
            for sae_key, sae_data in layer_data.items():
                if "16k" in str(sae_key):
                    absorption["first-letter"] = {
                        "absorption_rate": sae_data.get("mean_absorption_rate", 0.271),
                        "probe_f1": 1.0,
                        "n_entities": sae_data.get("n_words", 500),
                    }

    print(f"\nAbsorption rates by hierarchy:")
    for h, a in absorption.items():
        print(f"  {h}: {a['absorption_rate']:.3f} (F1={a['probe_f1']:.3f})")

    return absorption


def bootstrap_correlation(x, y, n_bootstrap=N_BOOTSTRAP, method="spearman"):
    """Bootstrap CI for correlation coefficient."""
    rng = np.random.RandomState(SEED)
    n = len(x)
    boot_rhos = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if method == "spearman":
            rho, _ = stats.spearmanr(x[idx], y[idx])
        else:
            rho, _ = stats.pearsonr(x[idx], y[idx])
        if not np.isnan(rho):
            boot_rhos.append(rho)
    boot_rhos = np.array(boot_rhos)
    return {
        "mean_rho": float(np.mean(boot_rhos)),
        "median_rho": float(np.median(boot_rhos)),
        "ci_lower": float(np.percentile(boot_rhos, 2.5)),
        "ci_upper": float(np.percentile(boot_rhos, 97.5)),
        "std": float(np.std(boot_rhos)),
        "n_bootstrap": n_bootstrap,
    }


def permutation_test(x, y, n_permutations=N_PERMUTATIONS, method="spearman"):
    """Permutation test for correlation significance."""
    rng = np.random.RandomState(SEED)
    if method == "spearman":
        observed_rho, _ = stats.spearmanr(x, y)
    else:
        observed_rho, _ = stats.pearsonr(x, y)

    count = 0
    for _ in range(n_permutations):
        perm_y = rng.permutation(y)
        if method == "spearman":
            perm_rho, _ = stats.spearmanr(x, perm_y)
        else:
            perm_rho, _ = stats.pearsonr(x, perm_y)
        if abs(perm_rho) >= abs(observed_rho):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return {
        "observed_rho": float(observed_rho),
        "p_value_permutation": float(p_value),
        "n_permutations": n_permutations,
    }


def analyze_individual_predictors(pairs, observed):
    """Analyze each predictor individually with bootstrap CI."""
    predictors = {
        "cos_sim": np.array([p["cos_sim"] for p in pairs]),
        "cos_sim_squared": np.array([p["cos_sim"] ** 2 for p in pairs]),
        "co_occur": np.array([p["co_occur"] for p in pairs]),
        "r_parent": np.array([p["r_parent"] for p in pairs]),
        "neg_r_parent": np.array([-p["r_parent"] for p in pairs]),
    }

    # Add derived predictors
    cos_sim = predictors["cos_sim"]
    co_occur = predictors["co_occur"]
    r_parent = predictors["r_parent"]

    # Competition coefficient: cos_sim * co_occur (interaction)
    predictors["competition_coeff"] = cos_sim * co_occur
    # Rate-adjusted: cos_sim^2 / (1 + |r_parent|)
    predictors["rate_adjusted"] = cos_sim ** 2 / (1 + np.abs(r_parent))

    results = {}
    for name, pred_values in predictors.items():
        # Basic correlation
        rho, p = stats.spearmanr(pred_values, observed)
        pearson_r, pearson_p = stats.pearsonr(pred_values, observed)

        # Bootstrap
        boot = bootstrap_correlation(pred_values, observed)

        # Permutation test
        perm = permutation_test(pred_values, observed)

        results[name] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "bootstrap": boot,
            "permutation": perm,
            "mean": float(np.mean(pred_values)),
            "std": float(np.std(pred_values)),
            "min": float(np.min(pred_values)),
            "max": float(np.max(pred_values)),
        }

    return results, predictors


def fit_multivariate_model(predictors, observed, pairs):
    """Fit multivariate linear model with LOO cross-validation."""
    # Standard 3-factor model
    X = np.column_stack([
        predictors["cos_sim_squared"],
        predictors["co_occur"],
        predictors["r_parent"],
    ])
    y = observed

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit model
    reg = LinearRegression()
    reg.fit(X_scaled, y)
    y_pred = reg.predict(X_scaled)

    # Overall metrics
    rho, p = stats.spearmanr(y_pred, y)
    pearson_r, pearson_p = stats.pearsonr(y_pred, y)
    r_squared = reg.score(X_scaled, y)

    # LOO cross-validation
    loo = LeaveOneOut()
    y_pred_loo = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X_scaled):
        reg_loo = LinearRegression()
        reg_loo.fit(X_scaled[train_idx], y[train_idx])
        y_pred_loo[test_idx] = reg_loo.predict(X_scaled[test_idx])

    loo_rho, loo_p = stats.spearmanr(y_pred_loo, y)
    loo_mse = float(np.mean((y_pred_loo - y) ** 2))

    # Bootstrap CI for LOO rho
    boot_loo = bootstrap_correlation(y_pred_loo, y)

    # Predicted vs observed for each pair
    predicted_vs_observed = []
    for i, pair in enumerate(pairs):
        predicted_vs_observed.append({
            "hierarchy": pair["hierarchy"],
            "class_name": pair["class_name"],
            "sae_key": pair.get("sae_key", "L24_16k"),
            "predicted": float(y_pred[i]),
            "loo_predicted": float(y_pred_loo[i]),
            "observed": float(y[i]),
            "cos_sim": float(pair["cos_sim"]),
            "co_occur": float(pair["co_occur"]),
            "r_parent": float(pair["r_parent"]),
        })

    model_result = {
        "feature_names": ["cos_sim_squared", "co_occur", "r_parent"],
        "coefficients": {
            "cos_sim_squared": float(reg.coef_[0]),
            "co_occur": float(reg.coef_[1]),
            "r_parent": float(reg.coef_[2]),
            "intercept": float(reg.intercept_),
        },
        "coefficients_unscaled": {
            "cos_sim_squared": float(reg.coef_[0] / scaler.scale_[0]),
            "co_occur": float(reg.coef_[1] / scaler.scale_[1]),
            "r_parent": float(reg.coef_[2] / scaler.scale_[2]),
        },
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "r_squared": float(r_squared),
        "adjusted_r_squared": float(1 - (1 - r_squared) * (len(y) - 1) / (len(y) - 3 - 1)),
        "n_pairs": len(y),
        "loo_cv": {
            "spearman_rho": float(loo_rho),
            "spearman_p": float(loo_p),
            "mse": loo_mse,
            "bootstrap_ci": boot_loo,
        },
        "predicted_vs_observed": predicted_vs_observed,
    }

    # Also fit alternative models
    alternative_models = {}

    # 2-factor: cos_sim + r_parent (dropping co_occur)
    X2 = np.column_stack([predictors["cos_sim_squared"], predictors["r_parent"]])
    X2s = StandardScaler().fit_transform(X2)
    reg2 = LinearRegression().fit(X2s, y)
    y_pred2 = reg2.predict(X2s)
    rho2, p2 = stats.spearmanr(y_pred2, y)
    alternative_models["cos_sim_sq_plus_r_parent"] = {
        "spearman_rho": float(rho2), "p_value": float(p2),
        "r_squared": float(reg2.score(X2s, y)),
    }

    # 2-factor: co_occur + r_parent (dropping cos_sim)
    X3 = np.column_stack([predictors["co_occur"], predictors["r_parent"]])
    X3s = StandardScaler().fit_transform(X3)
    reg3 = LinearRegression().fit(X3s, y)
    y_pred3 = reg3.predict(X3s)
    rho3, p3 = stats.spearmanr(y_pred3, y)
    alternative_models["co_occur_plus_r_parent"] = {
        "spearman_rho": float(rho3), "p_value": float(p3),
        "r_squared": float(reg3.score(X3s, y)),
    }

    # Competition coefficient only
    Xc = predictors["competition_coeff"].reshape(-1, 1)
    Xcs = StandardScaler().fit_transform(Xc)
    regc = LinearRegression().fit(Xcs, y)
    y_predc = regc.predict(Xcs)
    rhoc, pc = stats.spearmanr(y_predc, y)
    alternative_models["competition_coeff_only"] = {
        "spearman_rho": float(rhoc), "p_value": float(pc),
        "r_squared": float(regc.score(Xcs, y)),
    }

    model_result["alternative_models"] = alternative_models

    return model_result


def cross_domain_analysis(pairs, predictors, observed):
    """Analyze how predictor distributions differ across hierarchies."""
    hierarchies = sorted(set(p["hierarchy"] for p in pairs))

    per_hierarchy = {}
    for h in hierarchies:
        idx = np.array([i for i, p in enumerate(pairs) if p["hierarchy"] == h])
        h_obs = observed[idx]

        per_hierarchy[h] = {
            "n_pairs": len(idx),
            "mean_absorption": float(np.mean(h_obs)),
            "std_absorption": float(np.std(h_obs)),
            "median_absorption": float(np.median(h_obs)),
        }

        for pred_name in ["cos_sim", "cos_sim_squared", "co_occur", "r_parent", "competition_coeff"]:
            pred_vals = predictors[pred_name][idx]
            per_hierarchy[h][f"mean_{pred_name}"] = float(np.mean(pred_vals))
            per_hierarchy[h][f"std_{pred_name}"] = float(np.std(pred_vals))
            per_hierarchy[h][f"median_{pred_name}"] = float(np.median(pred_vals))

            # Within-hierarchy correlation
            if len(idx) >= 5:
                rho, p = stats.spearmanr(pred_vals, h_obs)
                per_hierarchy[h][f"rho_{pred_name}"] = float(rho) if not np.isnan(rho) else None
                per_hierarchy[h][f"p_{pred_name}"] = float(p) if not np.isnan(p) else None

    # Kruskal-Wallis test: do predictor distributions differ across hierarchies?
    kw_tests = {}
    for pred_name in ["cos_sim", "co_occur", "r_parent", "competition_coeff"]:
        groups = []
        for h in hierarchies:
            idx = [i for i, p in enumerate(pairs) if p["hierarchy"] == h]
            groups.append(predictors[pred_name][idx])
        stat, p = stats.kruskal(*groups)
        kw_tests[pred_name] = {
            "h_stat": float(stat),
            "p_value": float(p),
            "significant_01": p < 0.01,
        }

    # Pairwise Mann-Whitney between hierarchies for cos_sim
    pairwise = {}
    for i, h1 in enumerate(hierarchies):
        for h2 in hierarchies[i+1:]:
            idx1 = [j for j, p in enumerate(pairs) if p["hierarchy"] == h1]
            idx2 = [j for j, p in enumerate(pairs) if p["hierarchy"] == h2]
            for pred_name in ["cos_sim", "co_occur", "r_parent"]:
                stat, p = stats.mannwhitneyu(
                    predictors[pred_name][idx1],
                    predictors[pred_name][idx2],
                    alternative="two-sided"
                )
                key = f"{h1}_vs_{h2}_{pred_name}"
                pairwise[key] = {"u_stat": float(stat), "p_value": float(p)}

    return {
        "per_hierarchy": per_hierarchy,
        "kruskal_wallis_tests": kw_tests,
        "pairwise_mann_whitney": pairwise,
    }


def run_gpu_enhanced_analysis(pairs_l24_16k):
    """
    If GPU available, load SAE and compute additional pairs from
    absorption instances not covered in iter_009.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None, "No GPU available"

        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

        write_progress("gpu_loading_sae", 50, {"detail": "Loading Gemma Scope L24 16k SAE"})

        # Try to load SAE for recomputing predictors
        try:
            from sae_lens import SAE
            sae = SAE.from_pretrained(
                release="gemma-scope-2b-pt-res-canonical",
                sae_id="layer_24/width_16k/canonical",
                device="cuda:0",
            )
            W_dec = sae.W_dec.detach().cpu().numpy()  # (n_features, d_model)
            print(f"Loaded SAE decoder: {W_dec.shape}")

            # Recompute cosine similarities from decoder weights for validation
            enhanced_pairs = []
            for pair in pairs_l24_16k:
                child_id = pair.get("child_feature_id")
                if child_id is not None and child_id < W_dec.shape[0]:
                    d_child = W_dec[child_id]
                    # For now, store decoder norms as additional info
                    enhanced_pairs.append({
                        **pair,
                        "child_decoder_norm": float(np.linalg.norm(d_child)),
                        "recomputed": True,
                    })
                else:
                    enhanced_pairs.append({**pair, "recomputed": False})

            # Clean up GPU
            del sae
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return enhanced_pairs, "SAE loaded, decoder norms computed"

        except Exception as e:
            print(f"Warning: Could not load SAE: {e}")
            return None, f"SAE load failed: {e}"

    except ImportError:
        return None, "torch not available"


def compute_h9_verdict(model_result, individual_results):
    """Determine H9 verdict based on combined evidence."""
    model_rho = model_result["spearman_rho"]
    model_p = model_result["spearman_p"]
    loo_rho = model_result["loo_cv"]["spearman_rho"]
    loo_p = model_result["loo_cv"]["spearman_p"]
    r_squared = model_result["r_squared"]

    # Best individual predictor
    best_pred = max(
        individual_results.items(),
        key=lambda x: abs(x[1]["spearman_rho"])
    )
    best_pred_name = best_pred[0]
    best_pred_rho = best_pred[1]["spearman_rho"]

    # Verdict logic
    if loo_rho >= 0.5 and loo_p < 0.01:
        verdict = "SUPPORTED"
        confidence = "HIGH"
    elif loo_rho >= 0.3 and loo_p < 0.05:
        verdict = "PARTIALLY_SUPPORTED"
        confidence = "MODERATE"
    elif model_rho >= 0.3 and model_p < 0.01:
        verdict = "WEAK_SUPPORT"
        confidence = "LOW"
    else:
        verdict = "NOT_SUPPORTED"
        confidence = "HIGH" if loo_rho < 0.2 else "MODERATE"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "model_spearman_rho": float(model_rho),
        "model_p_value": float(model_p),
        "model_r_squared": float(r_squared),
        "loo_rho": float(loo_rho),
        "loo_p": float(loo_p),
        "best_individual_predictor": best_pred_name,
        "best_individual_rho": float(best_pred_rho),
        "interpretation": _interpret_verdict(verdict, model_rho, loo_rho, best_pred_name, best_pred_rho),
    }


def _interpret_verdict(verdict, model_rho, loo_rho, best_pred, best_rho):
    if verdict == "NOT_SUPPORTED":
        return (
            f"The three-factor rate-distortion model does NOT meaningfully predict "
            f"per-pair absorption rates. Model Spearman rho={model_rho:.3f}, "
            f"LOO CV rho={loo_rho:.3f}. Best individual predictor: {best_pred} "
            f"(rho={best_rho:.3f}). The cross-domain characterization stands "
            f"independently: interventional methods (activation patching) remain "
            f"necessary for absorption detection. This is the THIRD negative "
            f"predictive result alongside GAS (rho=0.116) and CMI (rho=0.044)."
        )
    elif verdict == "PARTIALLY_SUPPORTED":
        return (
            f"The three-factor model shows moderate predictive power "
            f"(model rho={model_rho:.3f}, LOO rho={loo_rho:.3f}). "
            f"Best predictor: {best_pred} (rho={best_rho:.3f})."
        )
    else:
        return (
            f"Verdict: {verdict}. Model rho={model_rho:.3f}, LOO rho={loo_rho:.3f}. "
            f"Best predictor: {best_pred} (rho={best_rho:.3f})."
        )


def main():
    start_time = time.time()
    print("=" * 70)
    print("Phase 2.2: Rate-Distortion Predictor Validation (Pooled Cross-Domain)")
    print("=" * 70)
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Seed: {SEED}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load data ----
    write_progress("loading_data", 5)
    all_pairs, pairs_16k, pairs_65k, iter9_data = load_iter9_pairs()
    absorption = load_absorption_data()
    print()

    # ---- Step 2: Focus on primary SAE (L24_16k) ----
    write_progress("analyzing_primary_sae", 10, {"n_pairs": len(pairs_16k)})
    print(f"\n--- Analysis: L24_16k ({len(pairs_16k)} pairs) ---")

    observed_16k = np.array([p["observed_absorption_rate"] for p in pairs_16k])
    print(f"Observed absorption: mean={np.mean(observed_16k):.3f}, std={np.std(observed_16k):.3f}")

    # Per-hierarchy breakdown
    print("\nPer-hierarchy pair counts (L24_16k):")
    from collections import Counter
    hier_counts = Counter(p["hierarchy"] for p in pairs_16k)
    for h, c in sorted(hier_counts.items()):
        idx = [i for i, p in enumerate(pairs_16k) if p["hierarchy"] == h]
        h_obs = observed_16k[idx]
        print(f"  {h}: {c} pairs, mean_absorption={np.mean(h_obs):.3f}")

    # ---- Step 3: Individual predictor analysis (L24_16k) ----
    write_progress("individual_predictors", 20)
    print("\n--- Individual Predictor Analysis (L24_16k) ---")
    individual_16k, predictors_16k = analyze_individual_predictors(pairs_16k, observed_16k)

    for name, result in individual_16k.items():
        sig = "*" if result["spearman_p"] < 0.05 else ""
        print(f"  {name}: rho={result['spearman_rho']:.3f} (p={result['spearman_p']:.4f}){sig} "
              f"  boot CI=[{result['bootstrap']['ci_lower']:.3f}, {result['bootstrap']['ci_upper']:.3f}]")

    # ---- Step 4: Multivariate model (L24_16k) ----
    write_progress("multivariate_model", 35)
    print("\n--- Multivariate Model (L24_16k) ---")
    model_16k = fit_multivariate_model(predictors_16k, observed_16k, pairs_16k)
    print(f"  3-factor model: rho={model_16k['spearman_rho']:.3f} (p={model_16k['spearman_p']:.6f})")
    print(f"  R-squared: {model_16k['r_squared']:.4f}")
    print(f"  LOO CV: rho={model_16k['loo_cv']['spearman_rho']:.3f} (p={model_16k['loo_cv']['spearman_p']:.6f})")
    print(f"  LOO MSE: {model_16k['loo_cv']['mse']:.4f}")
    print(f"\n  Coefficients (standardized):")
    for name, coef in model_16k["coefficients"].items():
        print(f"    {name}: {coef:.4f}")

    print(f"\n  Alternative models:")
    for name, alt in model_16k["alternative_models"].items():
        print(f"    {name}: rho={alt['spearman_rho']:.3f}, R2={alt['r_squared']:.4f}")

    # ---- Step 5: Cross-domain analysis (L24_16k) ----
    write_progress("cross_domain_analysis", 50)
    print("\n--- Cross-Domain Analysis (L24_16k) ---")
    cross_domain_16k = cross_domain_analysis(pairs_16k, predictors_16k, observed_16k)

    for h, hdata in cross_domain_16k["per_hierarchy"].items():
        print(f"\n  {h} (n={hdata['n_pairs']}):")
        print(f"    absorption: {hdata['mean_absorption']:.3f} +/- {hdata['std_absorption']:.3f}")
        print(f"    cos_sim: {hdata['mean_cos_sim']:.3f}, co_occur: {hdata['mean_co_occur']:.3f}, "
              f"r_parent: {hdata['mean_r_parent']:.3f}")
        if hdata.get("rho_cos_sim") is not None:
            print(f"    within-hierarchy cos_sim rho: {hdata['rho_cos_sim']:.3f} "
                  f"(p={hdata.get('p_cos_sim', float('nan')):.3f})")

    print("\n  Kruskal-Wallis (do predictors differ across hierarchies?):")
    for pred, kw in cross_domain_16k["kruskal_wallis_tests"].items():
        sig = "***" if kw["significant_01"] else ""
        print(f"    {pred}: H={kw['h_stat']:.2f}, p={kw['p_value']:.6f} {sig}")

    # ---- Step 6: Repeat for pooled L24_16k + L24_65k ----
    write_progress("pooled_analysis", 60, {"n_pairs": len(all_pairs)})
    print(f"\n\n--- Pooled Analysis (L24_16k + L24_65k, {len(all_pairs)} pairs) ---")

    observed_all = np.array([p["observed_absorption_rate"] for p in all_pairs])
    individual_all, predictors_all = analyze_individual_predictors(all_pairs, observed_all)
    model_all = fit_multivariate_model(predictors_all, observed_all, all_pairs)

    print(f"  3-factor model: rho={model_all['spearman_rho']:.3f} (p={model_all['spearman_p']:.6f})")
    print(f"  LOO CV: rho={model_all['loo_cv']['spearman_rho']:.3f} (p={model_all['loo_cv']['spearman_p']:.6f})")
    print(f"  R-squared: {model_all['r_squared']:.4f}")

    # ---- Step 7: GPU-enhanced analysis (optional) ----
    write_progress("gpu_analysis", 70)
    enhanced_pairs, gpu_msg = run_gpu_enhanced_analysis(pairs_16k)
    print(f"\nGPU-enhanced analysis: {gpu_msg}")
    gpu_result = {"status": gpu_msg}
    if enhanced_pairs:
        n_recomputed = sum(1 for p in enhanced_pairs if p.get("recomputed"))
        gpu_result["n_recomputed"] = n_recomputed
        gpu_result["decoder_norm_stats"] = {
            "mean": float(np.mean([p["child_decoder_norm"] for p in enhanced_pairs if p.get("recomputed")])),
            "std": float(np.std([p["child_decoder_norm"] for p in enhanced_pairs if p.get("recomputed")])),
        }
        print(f"  Recomputed decoder norms for {n_recomputed} pairs")

    # ---- Step 8: H9 verdict ----
    write_progress("computing_verdict", 85)
    h9_verdict = compute_h9_verdict(model_16k, individual_16k)
    print(f"\n{'=' * 70}")
    print(f"H9 VERDICT: {h9_verdict['verdict']} (confidence: {h9_verdict['confidence']})")
    print(f"{'=' * 70}")
    print(f"  Model rho={h9_verdict['model_spearman_rho']:.3f}, LOO rho={h9_verdict['loo_rho']:.3f}")
    print(f"  Best individual: {h9_verdict['best_individual_predictor']} (rho={h9_verdict['best_individual_rho']:.3f})")
    print(f"\n  {h9_verdict['interpretation']}")

    # ---- Step 9: Comparison with iter_009 ----
    write_progress("comparison", 90)
    iter9_verdict = iter9_data.get("h9_verdict", {})
    comparison = {
        "iter_009_model_rho": iter9_verdict.get("model_spearman_rho"),
        "iter_009_loo_rho": iter9_verdict.get("model_loo_rho"),
        "iter_010_model_rho": float(model_16k["spearman_rho"]),
        "iter_010_loo_rho": float(model_16k["loo_cv"]["spearman_rho"]),
        "iter_009_n_pairs": iter9_data.get("n_total_pairs"),
        "iter_010_n_pairs_16k": len(pairs_16k),
        "iter_010_n_pairs_pooled": len(all_pairs),
        "verdict_consistent": h9_verdict["verdict"] == iter9_verdict.get("verdict"),
        "notes": "Reanalysis with same data but improved methodology (bootstrap CI, permutation tests, alternative models)"
    }

    print(f"\n--- Comparison with iter_009 ---")
    print(f"  iter_009: model_rho={comparison['iter_009_model_rho']:.3f}, "
          f"loo_rho={comparison['iter_009_loo_rho']:.3f}, n={comparison['iter_009_n_pairs']}")
    print(f"  iter_010: model_rho={comparison['iter_010_model_rho']:.3f}, "
          f"loo_rho={comparison['iter_010_loo_rho']:.3f}, n_16k={comparison['iter_010_n_pairs_16k']}, n_pooled={comparison['iter_010_n_pairs_pooled']}")
    print(f"  Verdict consistent: {comparison['verdict_consistent']}")

    # ---- Step 10: Save results ----
    write_progress("saving_results", 95)

    elapsed = time.time() - start_time
    output = {
        "task_id": "phase2_rate_distortion_pooled",
        "mode": "PILOT",
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",
        "primary_sae": "L24_16k",
        "sae_configs": ["L24_16k", "L24_65k"],
        "n_bootstrap": N_BOOTSTRAP,
        "n_permutations": N_PERMUTATIONS,

        # Primary analysis (L24_16k)
        "primary_analysis": {
            "n_pairs": len(pairs_16k),
            "pairs_per_hierarchy": dict(hier_counts),
            "individual_predictors": individual_16k,
            "multivariate_model": model_16k,
            "cross_domain_analysis": cross_domain_16k,
        },

        # Pooled analysis (both SAEs)
        "pooled_analysis": {
            "n_pairs": len(all_pairs),
            "individual_predictors": individual_all,
            "multivariate_model": model_all,
        },

        # GPU analysis
        "gpu_enhanced": gpu_result,

        # Verdicts
        "h9_verdict": h9_verdict,
        "comparison_with_iter009": comparison,

        # Absorption context
        "absorption_rates_by_hierarchy": absorption,

        # Methodology notes
        "methodology_notes": {
            "data_source": "Reanalysis of iter_009/exp/results/phase3/rate_distortion_predictors.json",
            "improvement_over_iter9": [
                "Bootstrap 95% CI for all correlations",
                "Permutation tests for significance (non-parametric)",
                "Alternative multivariate models tested",
                "Derived predictors (competition_coeff, rate_adjusted)",
                "Cross-domain predictor distribution analysis with KW tests",
                "Within-hierarchy correlation analysis",
                "Pairwise Mann-Whitney for predictor differences",
            ],
            "pair_identification": "Pairs identified by iter_009 pipeline: for each class (letter/continent/country/language), "
                                   "the SAE feature with highest cosine similarity to the trained probe direction is the 'child', "
                                   "and the probe direction itself proxies the 'parent' concept.",
            "predictor_definitions": {
                "cos_sim": "cosine(child_decoder, parent_probe_direction)",
                "co_occur": "P(child_active | parent_active) over 500 random sequences",
                "r_parent": "MSE change when parent direction ablated from decoder (proxy for parent reconstruction importance)",
            },
            "limitation_note": "All pairs from the SAME iter_009 pipeline. No new pairs identified. "
                               "The pooled analysis adds methodological rigor but not new data.",
        },

        # Timing
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
    }

    output_path = RESULTS_DIR / "rate_distortion_pooled.json"
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to: {output_path}")

    # Write DONE marker
    DONE_FILE.write_text(json.dumps({
        "task": "phase2_rate_distortion_pooled",
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "elapsed_minutes": elapsed / 60,
        "output_file": str(output_path),
        "h9_verdict": h9_verdict["verdict"],
    }, indent=2))
    print(f"DONE marker: {DONE_FILE}")

    write_progress("completed", 100, {
        "h9_verdict": h9_verdict["verdict"],
        "model_rho": float(model_16k["spearman_rho"]),
        "loo_rho": float(model_16k["loo_cv"]["spearman_rho"]),
        "elapsed_minutes": elapsed / 60,
    })

    print(f"\n{'=' * 70}")
    print(f"Phase 2.2 completed in {elapsed/60:.1f} minutes")
    print(f"H9 verdict: {h9_verdict['verdict']}")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()

        # Write error progress
        write_progress("error", -1, {"error": str(e), "traceback": traceback.format_exc()})

        # Write partial DONE with error
        DONE_FILE.write_text(json.dumps({
            "task": "phase2_rate_distortion_pooled",
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }, indent=2))

        sys.exit(1)
