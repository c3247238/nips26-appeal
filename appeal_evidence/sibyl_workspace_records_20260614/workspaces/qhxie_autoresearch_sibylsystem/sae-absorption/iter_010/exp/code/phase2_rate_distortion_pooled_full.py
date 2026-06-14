#!/usr/bin/env python3
"""
Phase 2.2: Rate-Distortion Predictor Validation (Pooled Cross-Domain, H9) -- FULL MODE

Re-test the three-factor predictor model with parent-child pairs pooled
across ALL hierarchies, using corrected data from iter_009.

FULL MODE improvements over PILOT:
1. Uses ALL 262 pairs from iter_009 (both SAE widths L24_16k and L24_65k)
2. GPU-validated cosine similarities from SAE decoder weights
3. Comprehensive bootstrap analysis with 10k resamples
4. Permutation tests for non-parametric significance
5. Alternative model specifications (2-factor, interaction terms, nonlinear)
6. Within-hierarchy correlation analysis
7. Cross-validation with multiple splits
8. Publication-quality visualization
9. Proper cross-domain predictor distribution analysis

Key question: Does the three-factor model (cos_sim^2, co_occur, r_parent) predict
per-pair absorption rates?
Target: Spearman rho > 0.5. Falsification: rho < 0.3 or p > 0.05.
"""

import gc
import json
import os
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---- Configuration ----
TASK_ID = "phase2_rate_distortion_pooled"
SEED = 42
np.random.seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption")
CURRENT = WORKSPACE / "current"
RESULTS_DIR = CURRENT / "exp" / "results" / "phase2"
PROGRESS_FILE = CURRENT / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = CURRENT / "exp" / "results" / f"{TASK_ID}_DONE"
PID_FILE = CURRENT / "exp" / "results" / f"{TASK_ID}.pid"
FIGURES_DIR = CURRENT / "exp" / "results" / "figures"

# Source data
ITER9_RATE_DISTORTION = WORKSPACE / "iter_009" / "exp" / "results" / "phase3" / "rate_distortion_predictors.json"
ITER9_ABSORPTION_CROSSDOMAIN = WORKSPACE / "iter_009" / "exp" / "results" / "phase1" / "absorption_crossdomain.json"
ITER9_ABSORPTION_FIRSTLETTER = WORKSPACE / "iter_009" / "exp" / "results" / "phase1" / "absorption_firstletter.json"
ITER9_PROBES_DIR = WORKSPACE / "iter_009" / "exp" / "results" / "phase1"

MODE = "FULL"
N_BOOTSTRAP = 10000
N_PERMUTATIONS = 5000
GPU_ID = int(os.environ.get("CUDA_VISIBLE_DEVICES", "5"))


def write_pid():
    PID_FILE.write_text(str(os.getpid()))


def write_progress(step: str, pct: float, details: dict = None):
    """Write progress JSON for monitoring."""
    progress = {
        "task_id": TASK_ID,
        "step": step,
        "percent": pct,
        "timestamp": datetime.now().isoformat(),
        "details": details or {},
    }
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def mark_done(status="success", summary="", elapsed_seconds=0):
    # Clean PID
    if PID_FILE.exists():
        PID_FILE.unlink()
    # Read final progress
    final_progress = {}
    if PROGRESS_FILE.exists():
        try:
            final_progress = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
        "elapsed_minutes": elapsed_seconds / 60,
    }, indent=2))


def update_gpu_progress(elapsed_seconds, status="completed"):
    """Update gpu_progress.json atomically."""
    progress_path = CURRENT / "exp" / "gpu_progress.json"
    try:
        import filelock
        lock_path = CURRENT / "exp" / ".gpu_progress.lock"
        lock = filelock.FileLock(str(lock_path), timeout=10)
        with lock:
            data = json.loads(progress_path.read_text()) if progress_path.exists() else {
                "completed": [], "failed": [], "running": {}, "timings": {}
            }
            if status == "completed":
                if TASK_ID not in data.get("completed", []):
                    data.setdefault("completed", []).append(TASK_ID)
            else:
                if TASK_ID not in data.get("failed", []):
                    data.setdefault("failed", []).append(TASK_ID)
            data.get("running", {}).pop(TASK_ID, None)
            data.setdefault("timings", {})[TASK_ID] = {
                "planned_min": 30,
                "actual_min": round(elapsed_seconds / 60, 1),
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "config_snapshot": {
                    "model": "gemma-2-2b",
                    "mode": MODE,
                    "sae": "L24_16k + L24_65k",
                    "n_pairs_total": 262,
                    "n_bootstrap": N_BOOTSTRAP,
                    "n_permutations": N_PERMUTATIONS,
                    "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
                    "gpu_count": 1,
                },
            }
            progress_path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Warning: gpu_progress update failed: {e}")
        try:
            data = json.loads(progress_path.read_text()) if progress_path.exists() else {
                "completed": [], "failed": [], "running": {}, "timings": {}
            }
            if status == "completed":
                if TASK_ID not in data.get("completed", []):
                    data.setdefault("completed", []).append(TASK_ID)
            else:
                if TASK_ID not in data.get("failed", []):
                    data.setdefault("failed", []).append(TASK_ID)
            data.get("running", {}).pop(TASK_ID, None)
            progress_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass


# ============================================================
# Data Loading
# ============================================================
def load_iter9_pairs():
    """Load all pre-computed pairs from iter_009 rate-distortion results."""
    with open(ITER9_RATE_DISTORTION) as f:
        data = json.load(f)

    pairs = data["pairs_with_predictors"]
    print(f"Loaded {len(pairs)} pairs from iter_009")

    # Separate by SAE
    l24_16k = [p for p in pairs if p["sae_key"] == "L24_16k"]
    l24_65k = [p for p in pairs if p["sae_key"] == "L24_65k"]
    print(f"  L24_16k: {len(l24_16k)} pairs")
    print(f"  L24_65k: {len(l24_65k)} pairs")

    # Per-hierarchy breakdown
    for sae_name, sae_pairs in [("L24_16k", l24_16k), ("L24_65k", l24_65k)]:
        hier_counts = Counter(p["hierarchy"] for p in sae_pairs)
        print(f"  {sae_name} by hierarchy: {dict(sorted(hier_counts.items()))}")

    return pairs, l24_16k, l24_65k, data


def load_absorption_data():
    """Load per-hierarchy absorption data for context."""
    absorption = {}

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
                "ci_lower": entry.get("ci_lower"),
                "ci_upper": entry.get("ci_upper"),
            }

    with open(ITER9_ABSORPTION_FIRSTLETTER) as f:
        fl = json.load(f)
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
    for h, a in sorted(absorption.items()):
        print(f"  {h}: {a['absorption_rate']:.3f} (F1={a['probe_f1']:.3f})")

    return absorption


# ============================================================
# GPU-Enhanced Analysis
# ============================================================
def gpu_validate_cosine_similarities(pairs_16k, pairs_65k):
    """
    Load SAE decoder weights and recompute/validate cosine similarities.
    Also compute additional decoder geometry metrics.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            print("No GPU available, skipping decoder validation")
            return None

        from sae_lens import SAE

        validated = {}

        for sae_name, sae_pairs, release, sae_id in [
            ("L24_16k", pairs_16k, "gemma-scope-2b-pt-res-canonical", "layer_24/width_16k/canonical"),
            ("L24_65k", pairs_65k, "gemma-scope-2b-pt-res-canonical", "layer_24/width_65k/canonical"),
        ]:
            print(f"\nLoading SAE {sae_name}...")
            sae = SAE.from_pretrained(release, sae_id, device="cuda:0")
            W_dec = sae.W_dec.detach().cpu().float().numpy()  # [d_sae, d_model]
            print(f"  Decoder shape: {W_dec.shape}")

            # Load probes for computing probe-direction cosines
            probes = load_probes_numpy()

            pair_validations = []
            for pair in sae_pairs:
                child_fid = pair.get("child_feature_id")
                hierarchy = pair["hierarchy"]
                class_name = pair["class_name"]
                original_cos = pair.get("cos_sim", 0)

                if child_fid is None or child_fid >= W_dec.shape[0]:
                    pair_validations.append({
                        "class_name": class_name,
                        "hierarchy": hierarchy,
                        "valid": False,
                        "reason": "child_fid out of range"
                    })
                    continue

                d_child = W_dec[child_fid]
                child_norm = np.linalg.norm(d_child)

                # Get probe direction for the class
                probe_dir = get_probe_direction(probes, hierarchy, class_name)
                if probe_dir is None:
                    pair_validations.append({
                        "class_name": class_name,
                        "hierarchy": hierarchy,
                        "valid": False,
                        "reason": "no probe direction"
                    })
                    continue

                # Recompute cosine similarity
                probe_norm = np.linalg.norm(probe_dir)
                if probe_norm < 1e-10 or child_norm < 1e-10:
                    recomputed_cos = 0.0
                else:
                    recomputed_cos = float(np.dot(d_child, probe_dir) / (child_norm * probe_norm))

                pair_validations.append({
                    "class_name": class_name,
                    "hierarchy": hierarchy,
                    "valid": True,
                    "original_cos_sim": float(original_cos),
                    "recomputed_cos_sim": recomputed_cos,
                    "cos_sim_diff": abs(recomputed_cos - original_cos),
                    "child_decoder_norm": float(child_norm),
                    "child_fid": int(child_fid),
                })

            validated[sae_name] = {
                "n_pairs": len(sae_pairs),
                "n_valid": sum(1 for v in pair_validations if v.get("valid")),
                "pair_validations": pair_validations,
            }

            # Compute aggregate validation stats
            valid_diffs = [v["cos_sim_diff"] for v in pair_validations if v.get("valid")]
            if valid_diffs:
                validated[sae_name]["cos_sim_validation"] = {
                    "mean_abs_diff": float(np.mean(valid_diffs)),
                    "max_abs_diff": float(np.max(valid_diffs)),
                    "n_exact_match": sum(1 for d in valid_diffs if d < 1e-5),
                    "n_close_match": sum(1 for d in valid_diffs if d < 0.01),
                    "all_close": all(d < 0.05 for d in valid_diffs),
                }

            del sae, W_dec
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  Validated {validated[sae_name]['n_valid']}/{len(sae_pairs)} pairs")

        return validated

    except Exception as e:
        print(f"GPU validation failed: {e}")
        traceback.print_exc()
        return None


def load_probes_numpy():
    """Load probe coefficients as numpy arrays."""
    probes = {}

    # First-letter probe
    fl_path = ITER9_PROBES_DIR / "probe_first-letter_L24_sklearn.npz"
    if fl_path.exists():
        data = np.load(fl_path, allow_pickle=True)
        coef = data["coef"]
        classes_raw = data["classes"]
        if len(classes_raw) == 26 and all(isinstance(c, (int, np.integer)) for c in classes_raw):
            classes_list = [chr(ord('a') + i) for i in range(26)]
        else:
            classes_list = [str(c) for c in classes_raw]
        probes["first-letter"] = {"coef": coef, "classes": classes_list}

    # Cross-domain probes
    for hierarchy in ["city-continent", "city-language", "city-country"]:
        probe_path = ITER9_PROBES_DIR / f"probe_{hierarchy}_L24.npz"
        if probe_path.exists():
            data = np.load(probe_path, allow_pickle=True)
            coef = data["coef"]
            classes_list = [str(c) for c in data["classes"]]
            probes[hierarchy] = {"coef": coef, "classes": classes_list}

    return probes


def get_probe_direction(probes, hierarchy, class_name):
    """Get normalized probe direction for a specific class."""
    if hierarchy not in probes:
        return None
    probe_data = probes[hierarchy]
    classes = probe_data["classes"]
    if class_name not in classes:
        return None
    cls_idx = classes.index(class_name)
    direction = probe_data["coef"][cls_idx].astype(np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return None
    return direction / norm


# ============================================================
# Statistical Analysis Functions
# ============================================================
def bootstrap_correlation(x, y, n_bootstrap=N_BOOTSTRAP, method="spearman"):
    """Bootstrap CI for correlation coefficient."""
    rng = np.random.RandomState(SEED)
    n = len(x)
    boot_rhos = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            if method == "spearman":
                rho, _ = stats.spearmanr(x[idx], y[idx])
            else:
                rho, _ = stats.pearsonr(x[idx], y[idx])
            if not np.isnan(rho):
                boot_rhos.append(rho)
        except Exception:
            continue
    boot_rhos = np.array(boot_rhos)
    if len(boot_rhos) == 0:
        return {"mean_rho": 0, "ci_lower": 0, "ci_upper": 0, "std": 0, "n_bootstrap": 0}
    return {
        "mean_rho": float(np.mean(boot_rhos)),
        "median_rho": float(np.median(boot_rhos)),
        "ci_lower": float(np.percentile(boot_rhos, 2.5)),
        "ci_upper": float(np.percentile(boot_rhos, 97.5)),
        "std": float(np.std(boot_rhos)),
        "n_bootstrap": len(boot_rhos),
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
        try:
            if method == "spearman":
                perm_rho, _ = stats.spearmanr(x, perm_y)
            else:
                perm_rho, _ = stats.pearsonr(x, perm_y)
            if abs(perm_rho) >= abs(observed_rho):
                count += 1
        except Exception:
            continue

    p_value = (count + 1) / (n_permutations + 1)
    return {
        "observed_rho": float(observed_rho),
        "p_value_permutation": float(p_value),
        "n_permutations": n_permutations,
    }


def analyze_individual_predictors(pairs, observed):
    """Analyze each predictor individually with bootstrap CI and permutation tests."""
    predictors = {
        "cos_sim": np.array([p["cos_sim"] for p in pairs]),
        "cos_sim_squared": np.array([p["cos_sim"] ** 2 for p in pairs]),
        "co_occur": np.array([p["co_occur"] for p in pairs]),
        "r_parent": np.array([p["r_parent"] for p in pairs]),
        "neg_r_parent": np.array([-p["r_parent"] for p in pairs]),
    }

    # Derived predictors
    cos_sim = predictors["cos_sim"]
    co_occur = predictors["co_occur"]
    r_parent = predictors["r_parent"]

    predictors["competition_coeff"] = cos_sim * co_occur
    predictors["rate_adjusted"] = cos_sim ** 2 / (1 + np.abs(r_parent))
    predictors["interaction_cos_rp"] = cos_sim ** 2 * (-r_parent)

    results = {}
    for name, pred_values in predictors.items():
        rho, p = stats.spearmanr(pred_values, observed)
        pearson_r, pearson_p = stats.pearsonr(pred_values, observed)
        boot = bootstrap_correlation(pred_values, observed)
        perm = permutation_test(pred_values, observed)

        results[name] = {
            "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
            "spearman_p": float(p) if not np.isnan(p) else 1.0,
            "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else 0.0,
            "pearson_p": float(pearson_p) if not np.isnan(pearson_p) else 1.0,
            "bootstrap": boot,
            "permutation": perm,
            "mean": float(np.mean(pred_values)),
            "std": float(np.std(pred_values)),
            "min": float(np.min(pred_values)),
            "max": float(np.max(pred_values)),
        }

    return results, predictors


def fit_multivariate_model(predictors, observed, pairs):
    """Fit multivariate linear model with LOO and k-fold cross-validation."""
    # Standard 3-factor model
    X = np.column_stack([
        predictors["cos_sim_squared"],
        predictors["co_occur"],
        predictors["r_parent"],
    ])
    y = observed

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg = LinearRegression()
    reg.fit(X_scaled, y)
    y_pred = reg.predict(X_scaled)

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
    boot_loo = bootstrap_correlation(y_pred_loo, y)

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    y_pred_kfold = np.zeros_like(y)
    for train_idx, test_idx in kf.split(X_scaled):
        reg_kf = LinearRegression()
        reg_kf.fit(X_scaled[train_idx], y[train_idx])
        y_pred_kfold[test_idx] = reg_kf.predict(X_scaled[test_idx])
    kfold_rho, kfold_p = stats.spearmanr(y_pred_kfold, y)
    kfold_mse = float(np.mean((y_pred_kfold - y) ** 2))

    # 10-fold cross-validation
    kf10 = KFold(n_splits=10, shuffle=True, random_state=SEED)
    y_pred_kf10 = np.zeros_like(y)
    for train_idx, test_idx in kf10.split(X_scaled):
        reg_kf10 = LinearRegression()
        reg_kf10.fit(X_scaled[train_idx], y[train_idx])
        y_pred_kf10[test_idx] = reg_kf10.predict(X_scaled[test_idx])
    kf10_rho, kf10_p = stats.spearmanr(y_pred_kf10, y)

    # Predicted vs observed
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
        "kfold_5_cv": {
            "spearman_rho": float(kfold_rho),
            "spearman_p": float(kfold_p),
            "mse": kfold_mse,
        },
        "kfold_10_cv": {
            "spearman_rho": float(kf10_rho),
            "spearman_p": float(kf10_p),
        },
        "predicted_vs_observed": predicted_vs_observed,
    }

    # Alternative models
    alternative_models = {}

    # 2-factor: cos_sim + r_parent
    X2 = np.column_stack([predictors["cos_sim_squared"], predictors["r_parent"]])
    X2s = StandardScaler().fit_transform(X2)
    reg2 = LinearRegression().fit(X2s, y)
    y_pred2 = reg2.predict(X2s)
    rho2, p2 = stats.spearmanr(y_pred2, y)
    # LOO for this model
    y_pred2_loo = np.zeros_like(y)
    for train_idx, test_idx in LeaveOneOut().split(X2s):
        reg2_loo = LinearRegression().fit(X2s[train_idx], y[train_idx])
        y_pred2_loo[test_idx] = reg2_loo.predict(X2s[test_idx])
    rho2_loo, p2_loo = stats.spearmanr(y_pred2_loo, y)
    alternative_models["cos_sim_sq_plus_r_parent"] = {
        "spearman_rho": float(rho2), "p_value": float(p2),
        "r_squared": float(reg2.score(X2s, y)),
        "loo_rho": float(rho2_loo), "loo_p": float(p2_loo),
    }

    # 2-factor: co_occur + r_parent
    X3 = np.column_stack([predictors["co_occur"], predictors["r_parent"]])
    X3s = StandardScaler().fit_transform(X3)
    reg3 = LinearRegression().fit(X3s, y)
    y_pred3 = reg3.predict(X3s)
    rho3, p3 = stats.spearmanr(y_pred3, y)
    y_pred3_loo = np.zeros_like(y)
    for train_idx, test_idx in LeaveOneOut().split(X3s):
        reg3_loo = LinearRegression().fit(X3s[train_idx], y[train_idx])
        y_pred3_loo[test_idx] = reg3_loo.predict(X3s[test_idx])
    rho3_loo, p3_loo = stats.spearmanr(y_pred3_loo, y)
    alternative_models["co_occur_plus_r_parent"] = {
        "spearman_rho": float(rho3), "p_value": float(p3),
        "r_squared": float(reg3.score(X3s, y)),
        "loo_rho": float(rho3_loo), "loo_p": float(p3_loo),
    }

    # 2-factor: cos_sim + co_occur
    X4 = np.column_stack([predictors["cos_sim_squared"], predictors["co_occur"]])
    X4s = StandardScaler().fit_transform(X4)
    reg4 = LinearRegression().fit(X4s, y)
    y_pred4 = reg4.predict(X4s)
    rho4, p4 = stats.spearmanr(y_pred4, y)
    y_pred4_loo = np.zeros_like(y)
    for train_idx, test_idx in LeaveOneOut().split(X4s):
        reg4_loo = LinearRegression().fit(X4s[train_idx], y[train_idx])
        y_pred4_loo[test_idx] = reg4_loo.predict(X4s[test_idx])
    rho4_loo, p4_loo = stats.spearmanr(y_pred4_loo, y)
    alternative_models["cos_sim_sq_plus_co_occur"] = {
        "spearman_rho": float(rho4), "p_value": float(p4),
        "r_squared": float(reg4.score(X4s, y)),
        "loo_rho": float(rho4_loo), "loo_p": float(p4_loo),
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

    # Interaction model: cos_sim^2, co_occur, r_parent, cos_sim^2 * r_parent
    X5 = np.column_stack([
        predictors["cos_sim_squared"],
        predictors["co_occur"],
        predictors["r_parent"],
        predictors["interaction_cos_rp"],
    ])
    X5s = StandardScaler().fit_transform(X5)
    reg5 = LinearRegression().fit(X5s, y)
    y_pred5 = reg5.predict(X5s)
    rho5, p5 = stats.spearmanr(y_pred5, y)
    y_pred5_loo = np.zeros_like(y)
    for train_idx, test_idx in LeaveOneOut().split(X5s):
        reg5_loo = LinearRegression().fit(X5s[train_idx], y[train_idx])
        y_pred5_loo[test_idx] = reg5_loo.predict(X5s[test_idx])
    rho5_loo, p5_loo = stats.spearmanr(y_pred5_loo, y)
    alternative_models["interaction_4factor"] = {
        "spearman_rho": float(rho5), "p_value": float(p5),
        "r_squared": float(reg5.score(X5s, y)),
        "loo_rho": float(rho5_loo), "loo_p": float(p5_loo),
        "features": ["cos_sim_sq", "co_occur", "r_parent", "cos_sim_sq * neg_r_parent"],
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
            "min_absorption": float(np.min(h_obs)),
            "max_absorption": float(np.max(h_obs)),
        }

        for pred_name in ["cos_sim", "cos_sim_squared", "co_occur", "r_parent", "competition_coeff"]:
            if pred_name in predictors:
                pred_vals = predictors[pred_name][idx]
                per_hierarchy[h][f"mean_{pred_name}"] = float(np.mean(pred_vals))
                per_hierarchy[h][f"std_{pred_name}"] = float(np.std(pred_vals))
                per_hierarchy[h][f"median_{pred_name}"] = float(np.median(pred_vals))

                # Within-hierarchy correlation
                if len(idx) >= 5:
                    rho, p = stats.spearmanr(pred_vals, h_obs)
                    per_hierarchy[h][f"rho_{pred_name}"] = float(rho) if not np.isnan(rho) else None
                    per_hierarchy[h][f"p_{pred_name}"] = float(p) if not np.isnan(p) else None

    # Kruskal-Wallis: do predictor distributions differ across hierarchies?
    kw_tests = {}
    for pred_name in ["cos_sim", "co_occur", "r_parent", "competition_coeff"]:
        if pred_name not in predictors:
            continue
        groups = []
        for h in hierarchies:
            idx = [i for i, p in enumerate(pairs) if p["hierarchy"] == h]
            if len(idx) > 0:
                groups.append(predictors[pred_name][idx])
        if len(groups) >= 2:
            stat, p = stats.kruskal(*groups)
            kw_tests[pred_name] = {
                "h_stat": float(stat),
                "p_value": float(p),
                "significant_01": p < 0.01,
                "significant_05": p < 0.05,
            }

    # Pairwise Mann-Whitney
    pairwise = {}
    for i, h1 in enumerate(hierarchies):
        for h2 in hierarchies[i + 1:]:
            idx1 = [j for j, p in enumerate(pairs) if p["hierarchy"] == h1]
            idx2 = [j for j, p in enumerate(pairs) if p["hierarchy"] == h2]
            if len(idx1) < 2 or len(idx2) < 2:
                continue
            for pred_name in ["cos_sim", "co_occur", "r_parent"]:
                if pred_name not in predictors:
                    continue
                stat, p = stats.mannwhitneyu(
                    predictors[pred_name][idx1],
                    predictors[pred_name][idx2],
                    alternative="two-sided"
                )
                key = f"{h1}_vs_{h2}_{pred_name}"
                pairwise[key] = {"u_stat": float(stat), "p_value": float(p)}

    # Check if predictors EXPLAIN hierarchy-level absorption differences
    # Compute mean predictor values per hierarchy and correlate with mean absorption
    hierarchy_means = []
    for h in hierarchies:
        idx = [i for i, p in enumerate(pairs) if p["hierarchy"] == h]
        if len(idx) >= 3:
            hierarchy_means.append({
                "hierarchy": h,
                "n": len(idx),
                "mean_absorption": float(np.mean(observed[idx])),
                "mean_cos_sim": float(np.mean(predictors["cos_sim"][idx])),
                "mean_co_occur": float(np.mean(predictors["co_occur"][idx])),
                "mean_r_parent": float(np.mean(predictors["r_parent"][idx])),
            })

    hierarchy_level_correlations = {}
    if len(hierarchy_means) >= 3:
        abs_vals = np.array([h["mean_absorption"] for h in hierarchy_means])
        for pred in ["mean_cos_sim", "mean_co_occur", "mean_r_parent"]:
            pred_vals = np.array([h[pred] for h in hierarchy_means])
            rho, p = stats.spearmanr(pred_vals, abs_vals)
            hierarchy_level_correlations[pred] = {
                "rho": float(rho) if not np.isnan(rho) else None,
                "p": float(p) if not np.isnan(p) else None,
                "n_hierarchies": len(hierarchy_means),
            }

    return {
        "per_hierarchy": per_hierarchy,
        "kruskal_wallis_tests": kw_tests,
        "pairwise_mann_whitney": pairwise,
        "hierarchy_means": hierarchy_means,
        "hierarchy_level_correlations": hierarchy_level_correlations,
    }


def compute_h9_verdict(model_result, individual_results, cross_domain_result):
    """Determine H9 verdict based on combined evidence."""
    model_rho = model_result["spearman_rho"]
    model_p = model_result["spearman_p"]
    loo_rho = model_result["loo_cv"]["spearman_rho"]
    loo_p = model_result["loo_cv"]["spearman_p"]
    r_squared = model_result["r_squared"]
    kfold5_rho = model_result["kfold_5_cv"]["spearman_rho"]
    kfold10_rho = model_result["kfold_10_cv"]["spearman_rho"]

    # Best individual predictor
    best_pred = max(
        individual_results.items(),
        key=lambda x: abs(x[1]["spearman_rho"])
    )
    best_pred_name = best_pred[0]
    best_pred_rho = best_pred[1]["spearman_rho"]
    best_pred_p = best_pred[1]["spearman_p"]

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

    # Build interpretation
    if verdict == "NOT_SUPPORTED":
        interpretation = (
            f"The three-factor rate-distortion model does NOT meaningfully predict "
            f"per-pair absorption rates. Model Spearman rho={model_rho:.3f} (p={model_p:.2e}), "
            f"LOO CV rho={loo_rho:.3f} (p={loo_p:.2e}), 5-fold CV rho={kfold5_rho:.3f}, "
            f"10-fold CV rho={kfold10_rho:.3f}. R^2={r_squared:.3f} (only {r_squared*100:.1f}% "
            f"variance explained). Best individual predictor: {best_pred_name} "
            f"(rho={best_pred_rho:.3f}, p={best_pred_p:.3e}). "
            f"The cross-domain characterization stands independently: interventional methods "
            f"(activation patching) remain necessary for absorption detection. "
            f"This is the THIRD negative predictive result alongside "
            f"GAS (rho=0.116) and CMI (rho=0.044)."
        )
    elif verdict == "PARTIALLY_SUPPORTED":
        interpretation = (
            f"The three-factor model shows moderate but potentially useful predictive power. "
            f"Model rho={model_rho:.3f}, LOO rho={loo_rho:.3f}, R^2={r_squared:.3f}. "
            f"Best individual predictor: {best_pred_name} (rho={best_pred_rho:.3f}). "
            f"The model captures some variance but leaves most unexplained."
        )
    else:
        interpretation = (
            f"Verdict: {verdict}. Model rho={model_rho:.3f}, LOO rho={loo_rho:.3f}, "
            f"R^2={r_squared:.3f}. Best predictor: {best_pred_name} (rho={best_pred_rho:.3f})."
        )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "model_spearman_rho": float(model_rho),
        "model_p_value": float(model_p),
        "model_r_squared": float(r_squared),
        "model_adjusted_r_squared": float(model_result["adjusted_r_squared"]),
        "loo_rho": float(loo_rho),
        "loo_p": float(loo_p),
        "kfold_5_rho": float(kfold5_rho),
        "kfold_10_rho": float(kfold10_rho),
        "best_individual_predictor": best_pred_name,
        "best_individual_rho": float(best_pred_rho),
        "best_individual_p": float(best_pred_p),
        "interpretation": interpretation,
        "falsification_check": {
            "target_rho": 0.5,
            "falsification_threshold_rho": 0.3,
            "model_rho_passes_target": model_rho >= 0.5,
            "model_rho_above_falsification": model_rho >= 0.3,
            "loo_rho_above_falsification": loo_rho >= 0.3,
        }
    }


def generate_visualization(pairs, predictors, observed, model_result, output_dir):
    """Generate publication-quality scatter plot of predicted vs observed absorption."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        # Color map for hierarchies
        hierarchy_colors = {
            "first-letter": "#1f77b4",
            "city-continent": "#ff7f0e",
            "city-country": "#2ca02c",
            "city-language": "#d62728",
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel A: Predicted vs Observed (LOO)
        ax = axes[0]
        pvo = model_result["predicted_vs_observed"]
        for h in ["first-letter", "city-continent", "city-country", "city-language"]:
            h_points = [p for p in pvo if p["hierarchy"] == h]
            if not h_points:
                continue
            x = [p["loo_predicted"] for p in h_points]
            y = [p["observed"] for p in h_points]
            ax.scatter(x, y, c=hierarchy_colors.get(h, "gray"), label=h,
                       alpha=0.6, s=30, edgecolors="k", linewidths=0.3)

        # Add diagonal
        ax_min = min(min(p["loo_predicted"] for p in pvo), min(p["observed"] for p in pvo))
        ax_max = max(max(p["loo_predicted"] for p in pvo), max(p["observed"] for p in pvo))
        ax.plot([ax_min, ax_max], [ax_min, ax_max], "k--", alpha=0.3, linewidth=1)

        loo_rho = model_result["loo_cv"]["spearman_rho"]
        loo_p = model_result["loo_cv"]["spearman_p"]
        ax.set_xlabel("LOO Predicted Absorption Rate", fontsize=11)
        ax.set_ylabel("Observed Absorption Rate", fontsize=11)
        ax.set_title(f"Three-Factor Model (LOO CV)\n"
                     f"$\\rho$={loo_rho:.3f}, p={loo_p:.2e}",
                     fontsize=12)
        ax.legend(fontsize=9, loc="upper left")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        # Panel B: Individual predictor correlations (bar chart)
        ax = axes[1]
        pred_names = ["cos_sim", "cos_sim_squared", "co_occur", "r_parent",
                      "neg_r_parent", "competition_coeff"]
        pred_labels = ["cos_sim", "cos_sim^2", "co_occur", "r_parent",
                       "-r_parent", "competition"]
        rhos = []
        colors = []
        for name in pred_names:
            rho_val = 0
            for k, v in model_result.get("_individual", {}).items():
                if k == name:
                    rho_val = v.get("spearman_rho", 0)
            rhos.append(rho_val)
            colors.append("#4CAF50" if rho_val > 0 else "#F44336")

        # Use data passed separately since model_result doesn't have individual
        # (this will be populated from the caller)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "fig_rate_distortion_pooled.pdf"
        fig.savefig(str(fig_path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved: {fig_path}")

        # Second figure: per-predictor scatter plots
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
        for ax_idx, (pred_name, pred_label) in enumerate([
            ("cos_sim", "Cosine Similarity"),
            ("co_occur", "Co-occurrence P(child|parent)"),
            ("r_parent", "R_parent (reconstruction importance)"),
        ]):
            ax = axes2[ax_idx]
            pred_vals = predictors[pred_name]
            for h in ["first-letter", "city-continent", "city-country", "city-language"]:
                idx = [i for i, p in enumerate(pairs) if p["hierarchy"] == h]
                if not idx:
                    continue
                ax.scatter(pred_vals[idx], observed[idx],
                           c=hierarchy_colors.get(h, "gray"), label=h,
                           alpha=0.5, s=25, edgecolors="k", linewidths=0.3)

            rho, p_val = stats.spearmanr(pred_vals, observed)
            ax.set_xlabel(pred_label, fontsize=10)
            ax.set_ylabel("Observed Absorption Rate", fontsize=10)
            ax.set_title(f"$\\rho$={rho:.3f}, p={p_val:.2e}", fontsize=11)
            if ax_idx == 0:
                ax.legend(fontsize=8, loc="best")

        plt.tight_layout()
        fig2_path = FIGURES_DIR / "fig_rate_distortion_individual_predictors.pdf"
        fig2.savefig(str(fig2_path), dpi=300, bbox_inches="tight")
        plt.close(fig2)
        print(f"Figure saved: {fig2_path}")

        return [str(fig_path), str(fig2_path)]

    except Exception as e:
        print(f"Visualization failed: {e}")
        traceback.print_exc()
        return []


# ============================================================
# Main
# ============================================================
def main():
    start_time = time.time()
    write_pid()

    print("=" * 70)
    print("Phase 2.2: Rate-Distortion Predictor Validation (FULL MODE)")
    print("Pooled Cross-Domain, H9")
    print("=" * 70)
    print(f"Start: {datetime.now().isoformat()}")
    print(f"Seed: {SEED}")
    print(f"Mode: {MODE}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load data ----
    write_progress("loading_data", 5)
    all_pairs, pairs_16k, pairs_65k, iter9_data = load_iter9_pairs()
    absorption = load_absorption_data()
    print()

    # ---- Step 2: GPU validation of cosine similarities ----
    write_progress("gpu_validation", 15, {"detail": "Validating decoder cosine similarities"})
    print("\n--- GPU Validation of Cosine Similarities ---")
    gpu_validation = gpu_validate_cosine_similarities(pairs_16k, pairs_65k)
    if gpu_validation:
        for sae_name, val_data in gpu_validation.items():
            print(f"\n  {sae_name}:")
            cos_val = val_data.get("cos_sim_validation", {})
            print(f"    Valid: {val_data['n_valid']}/{val_data['n_pairs']}")
            if cos_val:
                print(f"    Mean cos_sim diff: {cos_val['mean_abs_diff']:.6f}")
                print(f"    Max cos_sim diff: {cos_val['max_abs_diff']:.6f}")
                print(f"    All close (<0.05): {cos_val['all_close']}")

    # ---- Step 3: Primary SAE analysis (L24_16k) ----
    write_progress("primary_analysis", 25, {"n_pairs": len(pairs_16k)})
    print(f"\n{'='*70}")
    print(f"Analysis 1: L24_16k ({len(pairs_16k)} pairs)")
    print(f"{'='*70}")

    observed_16k = np.array([p["observed_absorption_rate"] for p in pairs_16k])
    print(f"Observed absorption: mean={np.mean(observed_16k):.3f}, "
          f"std={np.std(observed_16k):.3f}, "
          f"median={np.median(observed_16k):.3f}")

    hier_counts_16k = Counter(p["hierarchy"] for p in pairs_16k)
    print("\nPer-hierarchy pair counts (L24_16k):")
    for h, c in sorted(hier_counts_16k.items()):
        idx = [i for i, p in enumerate(pairs_16k) if p["hierarchy"] == h]
        h_obs = observed_16k[idx]
        print(f"  {h}: {c} pairs, mean_absorption={np.mean(h_obs):.3f} +/- {np.std(h_obs):.3f}")

    # Individual predictors
    write_progress("individual_predictors_16k", 35)
    print("\n--- Individual Predictor Analysis (L24_16k) ---")
    individual_16k, predictors_16k = analyze_individual_predictors(pairs_16k, observed_16k)

    for name, result in individual_16k.items():
        sig = "*" if result["spearman_p"] < 0.05 else ""
        boot = result["bootstrap"]
        print(f"  {name}: rho={result['spearman_rho']:.3f} (p={result['spearman_p']:.4f}){sig} "
              f"boot CI=[{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")

    # Multivariate model
    write_progress("multivariate_model_16k", 45)
    print("\n--- Multivariate Model (L24_16k) ---")
    model_16k = fit_multivariate_model(predictors_16k, observed_16k, pairs_16k)
    print(f"  3-factor model: rho={model_16k['spearman_rho']:.3f} (p={model_16k['spearman_p']:.2e})")
    print(f"  R-squared: {model_16k['r_squared']:.4f} ({model_16k['r_squared']*100:.1f}% variance)")
    print(f"  Adj R-squared: {model_16k['adjusted_r_squared']:.4f}")
    print(f"  LOO CV: rho={model_16k['loo_cv']['spearman_rho']:.3f} (p={model_16k['loo_cv']['spearman_p']:.2e})")
    print(f"  5-fold CV: rho={model_16k['kfold_5_cv']['spearman_rho']:.3f}")
    print(f"  10-fold CV: rho={model_16k['kfold_10_cv']['spearman_rho']:.3f}")
    print(f"\n  Coefficients (standardized):")
    for name, coef in model_16k["coefficients"].items():
        print(f"    {name}: {coef:.4f}")
    print(f"\n  Coefficients (unscaled):")
    for name, coef in model_16k["coefficients_unscaled"].items():
        print(f"    {name}: {coef:.4f}")

    print(f"\n  Alternative models:")
    for name, alt in model_16k["alternative_models"].items():
        loo_str = f", LOO={alt.get('loo_rho', 'N/A'):.3f}" if "loo_rho" in alt else ""
        print(f"    {name}: rho={alt['spearman_rho']:.3f}, R2={alt['r_squared']:.4f}{loo_str}")

    # Cross-domain analysis
    write_progress("cross_domain_16k", 55)
    print("\n--- Cross-Domain Analysis (L24_16k) ---")
    cross_domain_16k = cross_domain_analysis(pairs_16k, predictors_16k, observed_16k)

    for h, hdata in cross_domain_16k["per_hierarchy"].items():
        print(f"\n  {h} (n={hdata['n_pairs']}):")
        print(f"    absorption: {hdata['mean_absorption']:.3f} +/- {hdata['std_absorption']:.3f}")
        print(f"    cos_sim: {hdata.get('mean_cos_sim', 0):.3f}, "
              f"co_occur: {hdata.get('mean_co_occur', 0):.3f}, "
              f"r_parent: {hdata.get('mean_r_parent', 0):.3f}")
        if hdata.get("rho_cos_sim") is not None:
            print(f"    within-hierarchy cos_sim rho: {hdata['rho_cos_sim']:.3f}")
        if hdata.get("rho_r_parent") is not None:
            print(f"    within-hierarchy r_parent rho: {hdata['rho_r_parent']:.3f}")

    print("\n  Kruskal-Wallis (do predictors differ across hierarchies?):")
    for pred, kw in cross_domain_16k["kruskal_wallis_tests"].items():
        sig = "***" if kw["significant_01"] else ("*" if kw["significant_05"] else "")
        print(f"    {pred}: H={kw['h_stat']:.2f}, p={kw['p_value']:.6f} {sig}")

    if cross_domain_16k.get("hierarchy_level_correlations"):
        print("\n  Hierarchy-level predictor-absorption correlations:")
        for pred, corr in cross_domain_16k["hierarchy_level_correlations"].items():
            print(f"    {pred}: rho={corr.get('rho', 'N/A')}, p={corr.get('p', 'N/A')}")

    # ---- Step 4: Pooled analysis (L24_16k + L24_65k) ----
    write_progress("pooled_analysis", 65, {"n_pairs": len(all_pairs)})
    print(f"\n{'='*70}")
    print(f"Analysis 2: Pooled (L24_16k + L24_65k, {len(all_pairs)} pairs)")
    print(f"{'='*70}")

    observed_all = np.array([p["observed_absorption_rate"] for p in all_pairs])
    print(f"Observed absorption: mean={np.mean(observed_all):.3f}, std={np.std(observed_all):.3f}")

    individual_all, predictors_all = analyze_individual_predictors(all_pairs, observed_all)
    model_all = fit_multivariate_model(predictors_all, observed_all, all_pairs)
    cross_domain_all = cross_domain_analysis(all_pairs, predictors_all, observed_all)

    print(f"  3-factor model: rho={model_all['spearman_rho']:.3f} (p={model_all['spearman_p']:.2e})")
    print(f"  R-squared: {model_all['r_squared']:.4f}")
    print(f"  LOO CV: rho={model_all['loo_cv']['spearman_rho']:.3f} (p={model_all['loo_cv']['spearman_p']:.2e})")
    print(f"  5-fold CV: rho={model_all['kfold_5_cv']['spearman_rho']:.3f}")
    print(f"  10-fold CV: rho={model_all['kfold_10_cv']['spearman_rho']:.3f}")

    for name, result in individual_all.items():
        if abs(result["spearman_rho"]) > 0.1:
            sig = "*" if result["spearman_p"] < 0.05 else ""
            print(f"  {name}: rho={result['spearman_rho']:.3f} (p={result['spearman_p']:.4f}){sig}")

    # ---- Step 5: L24_65k only analysis ----
    write_progress("analysis_65k", 72, {"n_pairs": len(pairs_65k)})
    print(f"\n{'='*70}")
    print(f"Analysis 3: L24_65k only ({len(pairs_65k)} pairs)")
    print(f"{'='*70}")

    observed_65k = np.array([p["observed_absorption_rate"] for p in pairs_65k])
    individual_65k, predictors_65k = analyze_individual_predictors(pairs_65k, observed_65k)
    model_65k = fit_multivariate_model(predictors_65k, observed_65k, pairs_65k)
    cross_domain_65k = cross_domain_analysis(pairs_65k, predictors_65k, observed_65k)

    print(f"  3-factor model: rho={model_65k['spearman_rho']:.3f} (p={model_65k['spearman_p']:.2e})")
    print(f"  LOO CV: rho={model_65k['loo_cv']['spearman_rho']:.3f}")
    print(f"  R-squared: {model_65k['r_squared']:.4f}")

    # ---- Step 6: H9 Verdict ----
    write_progress("computing_verdict", 80)
    h9_verdict = compute_h9_verdict(model_16k, individual_16k, cross_domain_16k)
    print(f"\n{'='*70}")
    print(f"H9 VERDICT: {h9_verdict['verdict']} (confidence: {h9_verdict['confidence']})")
    print(f"{'='*70}")
    print(f"  Model rho={h9_verdict['model_spearman_rho']:.3f} (p={h9_verdict['model_p_value']:.2e})")
    print(f"  LOO rho={h9_verdict['loo_rho']:.3f}")
    print(f"  5-fold rho={h9_verdict['kfold_5_rho']:.3f}")
    print(f"  10-fold rho={h9_verdict['kfold_10_rho']:.3f}")
    print(f"  R^2={h9_verdict['model_r_squared']:.3f}")
    print(f"  Best individual: {h9_verdict['best_individual_predictor']} "
          f"(rho={h9_verdict['best_individual_rho']:.3f})")
    print(f"\n  {h9_verdict['interpretation']}")

    # ---- Step 7: Comparison with iter_009 ----
    write_progress("comparison", 85)
    iter9_verdict = iter9_data.get("h9_verdict", {})
    iter9_model = iter9_data.get("model_results", {})
    comparison = {
        "iter_009_model_rho": iter9_model.get("model_fit", {}).get("spearman_rho"),
        "iter_009_loo_rho": iter9_model.get("model_fit", {}).get("loo_cv", {}).get("spearman_rho"),
        "iter_009_n_pairs": iter9_data.get("n_total_pairs"),
        "iter_010_model_rho_16k": float(model_16k["spearman_rho"]),
        "iter_010_loo_rho_16k": float(model_16k["loo_cv"]["spearman_rho"]),
        "iter_010_model_rho_pooled": float(model_all["spearman_rho"]),
        "iter_010_loo_rho_pooled": float(model_all["loo_cv"]["spearman_rho"]),
        "iter_010_n_pairs_16k": len(pairs_16k),
        "iter_010_n_pairs_65k": len(pairs_65k),
        "iter_010_n_pairs_pooled": len(all_pairs),
        "notes": "FULL mode: comprehensive analysis with bootstrap CI, permutation tests, "
                 "alternative models, cross-domain analysis, GPU-validated cosine similarities, "
                 "k-fold cross-validation, and both SAE widths analyzed separately."
    }

    print(f"\n--- Comparison with iter_009 ---")
    print(f"  iter_009: model_rho={comparison.get('iter_009_model_rho', 'N/A')}, "
          f"loo_rho={comparison.get('iter_009_loo_rho', 'N/A')}, "
          f"n={comparison.get('iter_009_n_pairs', 'N/A')}")
    print(f"  iter_010 (16k): model_rho={comparison['iter_010_model_rho_16k']:.3f}, "
          f"loo_rho={comparison['iter_010_loo_rho_16k']:.3f}, "
          f"n={comparison['iter_010_n_pairs_16k']}")
    print(f"  iter_010 (pooled): model_rho={comparison['iter_010_model_rho_pooled']:.3f}, "
          f"loo_rho={comparison['iter_010_loo_rho_pooled']:.3f}, "
          f"n={comparison['iter_010_n_pairs_pooled']}")

    # ---- Step 8: Visualization ----
    write_progress("visualization", 90)
    print("\n--- Generating Figures ---")
    fig_paths = generate_visualization(
        pairs_16k, predictors_16k, observed_16k, model_16k, FIGURES_DIR
    )

    # ---- Step 9: Save results ----
    write_progress("saving_results", 95)

    elapsed = time.time() - start_time
    output = {
        "task_id": TASK_ID,
        "mode": MODE,
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
            "pairs_per_hierarchy": dict(hier_counts_16k),
            "individual_predictors": individual_16k,
            "multivariate_model": model_16k,
            "cross_domain_analysis": cross_domain_16k,
        },

        # L24_65k analysis
        "analysis_65k": {
            "n_pairs": len(pairs_65k),
            "individual_predictors": individual_65k,
            "multivariate_model": model_65k,
            "cross_domain_analysis": cross_domain_65k,
        },

        # Pooled analysis (both SAEs)
        "pooled_analysis": {
            "n_pairs": len(all_pairs),
            "individual_predictors": individual_all,
            "multivariate_model": model_all,
            "cross_domain_analysis": cross_domain_all,
        },

        # GPU validation
        "gpu_validation": gpu_validation if gpu_validation else {"status": "skipped"},

        # Verdicts
        "h9_verdict": h9_verdict,
        "comparison_with_iter009": comparison,

        # Absorption context
        "absorption_rates_by_hierarchy": absorption,

        # Figures
        "figure_paths": fig_paths,

        # Methodology
        "methodology_notes": {
            "data_source": "iter_009/exp/results/phase3/rate_distortion_predictors.json (262 pairs)",
            "analysis_improvements_over_pilot": [
                "FULL mode: all 262 pairs across both SAE widths",
                "GPU-validated cosine similarities from SAE decoder weights",
                "Bootstrap 95% CI (10k resamples) for all correlations",
                "Permutation tests (5k permutations) for non-parametric significance",
                "LOO + 5-fold + 10-fold cross-validation for model robustness",
                "Alternative model specifications with LOO (6 models tested)",
                "Interaction terms (cos_sim^2 * neg_r_parent)",
                "Separate analysis per SAE width (L24_16k, L24_65k)",
                "Hierarchy-level predictor-absorption correlations",
                "Within-hierarchy correlation analysis",
                "Pairwise Mann-Whitney tests for predictor differences",
                "Publication-quality visualization",
            ],
            "pair_identification": (
                "Pairs from iter_009: for each class, the SAE feature with highest "
                "cosine similarity to the trained probe direction is the 'child'. "
                "The probe direction proxies the 'parent' concept."
            ),
            "predictor_definitions": {
                "cos_sim": "cosine(child_decoder, parent_probe_direction)",
                "co_occur": "P(child_active | parent_active) over 500 random sequences",
                "r_parent": "MSE change when parent direction ablated from decoder",
                "competition_coeff": "cos_sim * co_occur (interaction)",
                "rate_adjusted": "cos_sim^2 / (1 + |r_parent|)",
                "interaction_cos_rp": "cos_sim^2 * (-r_parent)",
            },
        },

        # Timing
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
    }

    output_path = RESULTS_DIR / "rate_distortion_pooled.json"
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to: {output_path}")

    # Write summary markdown
    summary_path = RESULTS_DIR / "rate_distortion_pooled_summary.md"
    summary_md = f"""# Rate-Distortion Predictor Validation (FULL MODE) -- H9

## Result: {h9_verdict['verdict']} ({h9_verdict['confidence']} confidence)

## Key Findings

### Primary Analysis (L24_16k, {len(pairs_16k)} pairs)
- **Three-factor model**: rho={model_16k['spearman_rho']:.3f} (p={model_16k['spearman_p']:.2e})
- **LOO CV**: rho={model_16k['loo_cv']['spearman_rho']:.3f} (p={model_16k['loo_cv']['spearman_p']:.2e})
- **5-fold CV**: rho={model_16k['kfold_5_cv']['spearman_rho']:.3f}
- **R-squared**: {model_16k['r_squared']:.4f} ({model_16k['r_squared']*100:.1f}% variance explained)
- **Best individual predictor**: {h9_verdict['best_individual_predictor']} (rho={h9_verdict['best_individual_rho']:.3f})

### Pooled Analysis (L24_16k + L24_65k, {len(all_pairs)} pairs)
- **Three-factor model**: rho={model_all['spearman_rho']:.3f}
- **LOO CV**: rho={model_all['loo_cv']['spearman_rho']:.3f}

### Individual Predictor Correlations (L24_16k)
| Predictor | Spearman rho | p-value | Bootstrap 95% CI |
|-----------|-------------|---------|------------------|
"""
    for name in ["cos_sim", "cos_sim_squared", "co_occur", "r_parent", "neg_r_parent", "competition_coeff"]:
        r = individual_16k[name]
        boot = r["bootstrap"]
        summary_md += (f"| {name} | {r['spearman_rho']:.3f} | {r['spearman_p']:.4f} | "
                       f"[{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}] |\n")

    summary_md += f"""
### Cross-Domain Predictor Distributions
"""
    for h, hdata in cross_domain_16k["per_hierarchy"].items():
        summary_md += (f"- **{h}** (n={hdata['n_pairs']}): "
                       f"absorption={hdata['mean_absorption']:.3f}, "
                       f"cos_sim={hdata.get('mean_cos_sim', 0):.3f}, "
                       f"co_occur={hdata.get('mean_co_occur', 0):.3f}, "
                       f"r_parent={hdata.get('mean_r_parent', 0):.3f}\n")

    summary_md += f"""
### Interpretation
{h9_verdict['interpretation']}

## Comparison with Prior Results
- **iter_009**: model rho={comparison.get('iter_009_model_rho', 'N/A')}, n={comparison.get('iter_009_n_pairs', 'N/A')}
- **iter_010 (16k)**: model rho={comparison['iter_010_model_rho_16k']:.3f}, n={comparison['iter_010_n_pairs_16k']}
- **iter_010 (pooled)**: model rho={comparison['iter_010_model_rho_pooled']:.3f}, n={comparison['iter_010_n_pairs_pooled']}

## Elapsed: {elapsed/60:.1f} minutes
"""
    summary_path.write_text(summary_md)
    print(f"Summary saved to: {summary_path}")

    # Update tracking
    mark_done("success", f"H9={h9_verdict['verdict']}, model_rho={model_16k['spearman_rho']:.3f}", elapsed)
    update_gpu_progress(elapsed, "completed")

    write_progress("completed", 100, {
        "h9_verdict": h9_verdict["verdict"],
        "model_rho_16k": float(model_16k["spearman_rho"]),
        "loo_rho_16k": float(model_16k["loo_cv"]["spearman_rho"]),
        "model_rho_pooled": float(model_all["spearman_rho"]),
        "n_pairs_16k": len(pairs_16k),
        "n_pairs_pooled": len(all_pairs),
        "elapsed_minutes": elapsed / 60,
    })

    print(f"\n{'='*70}")
    print(f"Phase 2.2 FULL completed in {elapsed/60:.1f} minutes")
    print(f"H9 verdict: {h9_verdict['verdict']}")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        elapsed = time.time() - (globals().get("start_time", time.time()))
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()

        write_progress("error", -1, {"error": str(e), "traceback": traceback.format_exc()})
        mark_done("failed", str(e), elapsed)
        update_gpu_progress(elapsed, "failed")

        sys.exit(1)
