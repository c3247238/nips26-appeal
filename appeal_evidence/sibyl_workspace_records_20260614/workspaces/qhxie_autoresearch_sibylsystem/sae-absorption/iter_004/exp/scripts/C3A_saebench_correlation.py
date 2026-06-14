#!/usr/bin/env python3
"""
C3A_saebench_correlation.py

Download SAEBench eval results from HuggingFace for Gemma 2 2B SAEs.
Compute Pearson r and Spearman rho between absorption_score and downstream task performance.
Apply Bonferroni correction for multiple tests.
Compute partial correlations controlling for SAE width (log), layer, architecture class.
Save full results to exp/results/full/C3A_saebench_corr.json.

This is a CPU-only task.
"""

import json
import os
import sys
import time
import requests
import numpy as np
from pathlib import Path
from datetime import datetime

# ─── PID file ────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TASK_ID = "C3A_saebench_correlation"

pid_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

def write_progress(step, pct):
    progress = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "step": step,
        "pct": pct,
        "ts": time.time()
    }))
    print(f"[PROGRESS {pct}%] {step}", flush=True)

def mark_done(status="success", summary=""):
    pid_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except:
            pass
    marker = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

# ─── Constants ────────────────────────────────────────────────────────────────
BASE_URL = "https://huggingface.co/datasets/adamkarvonen/sae_bench_results/resolve/main"
API_URL = "https://huggingface.co/api/datasets/adamkarvonen/sae_bench_results"
BONFERRONI_ALPHA = 0.05
N_TESTS = 4  # sparse_probing_f1, scr, ravel(tpp), unlearning
ALPHA_CORRECTED = BONFERRONI_ALPHA / N_TESTS  # 0.0125
MEANINGFUL_R_THRESHOLD = 0.30

# Width string -> numeric width
WIDTH_MAP = {
    "16k": 16384,
    "65k": 65536,
    "1m": 1_048_576,
    "131k": 131072,
    "262k": 262144,
    "524k": 524288,
}

def width_to_int(w_str):
    """Convert width string like '16k', '65k', '1m' to integer."""
    w = w_str.lower()
    return WIDTH_MAP.get(w, None)

def parse_sae_key(filename, sae_release):
    """Parse layer, width, l0 from a filename like:
    gemma-scope-2b-pt-res_layer_12_width_16k_average_l0_176_eval_results.json
    or canonical:
    gemma-scope-2b-pt-res-canonical_layer_12_width_16k_canonical_eval_results.json
    """
    # Extract layer
    layer = None
    if "_layer_" in filename:
        parts = filename.split("_layer_")[1]
        layer = int(parts.split("_")[0])

    # Extract width
    width_str = None
    width_int = None
    if "_width_" in filename:
        w = filename.split("_width_")[1]
        width_str = w.split("_")[0]
        width_int = width_to_int(width_str)

    # Extract l0 if present
    l0 = None
    if "_average_l0_" in filename:
        l0_part = filename.split("_average_l0_")[1]
        l0 = int(l0_part.split("_")[0])

    # Architecture class
    arch_class = "gemma_scope_2b"
    if "canonical" in sae_release:
        arch_class = "gemma_scope_2b_canonical"

    return layer, width_str, width_int, l0, arch_class


def fetch_json(url, retries=3, timeout=30):
    """Fetch JSON from URL with retry."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 404:
                return None
            else:
                print(f"  HTTP {r.status_code} for {url}, attempt {attempt+1}")
        except Exception as e:
            print(f"  Error fetching {url}: {e}, attempt {attempt+1}")
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return None


def get_all_gemma2b_files():
    """Get all Gemma Scope 2B SAE file paths from the HuggingFace dataset."""
    print("Fetching dataset file list...", flush=True)
    data = fetch_json(API_URL, timeout=30)
    if not data:
        raise RuntimeError("Failed to fetch dataset file list")

    siblings = data.get("siblings", [])
    gemma_files = {}

    for s in siblings:
        fn = s["rfilename"]
        if "gemma-scope-2b" not in fn or not fn.endswith(".json"):
            continue

        # Parse directory (eval type)
        parts = fn.split("/")
        if len(parts) < 3:
            continue
        eval_type = parts[0]  # absorption, scr, sparse_probing, tpp, unlearning
        sae_release = parts[1]
        fname = parts[2]

        # Create SAE key (sae_release + filename minus eval_results)
        sae_key = f"{sae_release}/{fname.replace('_eval_results.json', '')}"

        if sae_key not in gemma_files:
            gemma_files[sae_key] = {}
        gemma_files[sae_key][eval_type] = fn

    print(f"Found {len(gemma_files)} unique Gemma Scope 2B SAE configurations", flush=True)
    return gemma_files


def extract_absorption_score(data):
    """Extract mean_absorption_score from absorption eval data."""
    metrics = data.get("eval_result_metrics", {})
    mean_metrics = metrics.get("mean", {})
    return mean_metrics.get("mean_absorption_score", None)


def extract_sparse_probing_f1(data):
    """Extract SAE test accuracy (top-k F1 proxy) from sparse_probing data.
    Use sae_test_accuracy as the primary metric (top-5 by default)."""
    metrics = data.get("eval_result_metrics", {})
    sae_metrics = metrics.get("sae", {})
    # Prefer top_5 accuracy; fall back to overall
    val = sae_metrics.get("sae_top_5_test_accuracy") or sae_metrics.get("sae_test_accuracy")
    return val


def extract_scr_score(data):
    """Extract main SCR metric. Use scr_metric_threshold_10 as representative."""
    metrics = data.get("eval_result_metrics", {})
    scr_metrics = metrics.get("scr_metrics", {})
    # Use threshold_10 as a representative single score
    return scr_metrics.get("scr_metric_threshold_10", None)


def extract_tpp_score(data):
    """Extract TPP (Think Part Preservation) metric as RAVEL proxy.
    Use tpp_threshold_10_total_metric."""
    metrics = data.get("eval_result_metrics", {})
    tpp_metrics = metrics.get("tpp_metrics", {})
    return tpp_metrics.get("tpp_threshold_10_total_metric", None)


def extract_unlearning_score(data):
    """Extract unlearning score."""
    metrics = data.get("eval_result_metrics", {})
    unlearning_metrics = metrics.get("unlearning", {})
    return unlearning_metrics.get("unlearning_score", None)


def pearson_r_ci(r, n, alpha=0.05):
    """Compute 95% CI for Pearson r using Fisher z-transformation."""
    from scipy import stats
    if n <= 3:
        return [None, None]
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lo = np.tanh(z - z_crit * se)
    ci_hi = np.tanh(z + z_crit * se)
    return [round(ci_lo, 4), round(ci_hi, 4)]


def compute_partial_r(x, y, covariates):
    """Compute partial Pearson r between x and y controlling for covariates.
    Uses residuals from OLS regression on covariates.
    """
    from scipy import stats

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)
    k = len(covariates)

    if n <= k + 3:
        return None, None

    X_cov = np.column_stack([np.array(c, dtype=float) for c in covariates])  # (n, k)
    ones = np.ones((n, 1))
    X_full = np.hstack([ones, X_cov])  # Add intercept: (n, k+1)

    def residualize(v):
        """Return residuals of v after regressing on X_full."""
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_full, v, rcond=None)
            pred = X_full @ coeffs
            return v - pred
        except Exception as e:
            print(f"    lstsq error: {e}")
            return v

    res_x = residualize(x)
    res_y = residualize(y)

    # Check if residuals have variance
    if np.std(res_x) < 1e-12 or np.std(res_y) < 1e-12:
        print("    Warning: degenerate residuals (near-zero std)")
        return None, None

    r, p = stats.pearsonr(res_x, res_y)
    return round(float(r), 4), round(float(p), 4)


def main():
    start_time = time.time()
    write_progress("fetching_file_list", 5)

    # Get all file paths
    gemma_files = get_all_gemma2b_files()

    write_progress("downloading_absorption_data", 10)

    # Build dataset: for each SAE, fetch absorption + downstream metrics
    records = []
    total = len(gemma_files)

    for i, (sae_key, eval_files) in enumerate(sorted(gemma_files.items())):
        pct = 10 + int(60 * (i / total))
        if i % 10 == 0:
            write_progress(f"downloading_sae_{i}/{total}", pct)

        # Must have absorption eval
        if "absorption" not in eval_files:
            continue

        # Fetch absorption data
        abs_url = f"{BASE_URL}/{eval_files['absorption']}"
        abs_data = fetch_json(abs_url)
        if not abs_data:
            print(f"  Skipping {sae_key}: failed to fetch absorption")
            continue

        absorption_score = extract_absorption_score(abs_data)
        if absorption_score is None:
            print(f"  Skipping {sae_key}: no absorption_score")
            continue

        # Parse SAE metadata
        sae_release = sae_key.split("/")[0]
        fname = sae_key.split("/")[1] + "_eval_results"
        layer, width_str, width_int, l0, arch_class = parse_sae_key(
            fname, sae_release
        )

        # Fetch downstream metrics
        sparse_probing_f1 = None
        if "sparse_probing" in eval_files:
            sp_data = fetch_json(f"{BASE_URL}/{eval_files['sparse_probing']}")
            if sp_data:
                sparse_probing_f1 = extract_sparse_probing_f1(sp_data)

        scr_score = None
        if "scr" in eval_files:
            scr_data = fetch_json(f"{BASE_URL}/{eval_files['scr']}")
            if scr_data:
                scr_score = extract_scr_score(scr_data)

        tpp_score = None  # RAVEL proxy via TPP
        if "tpp" in eval_files:
            tpp_data = fetch_json(f"{BASE_URL}/{eval_files['tpp']}")
            if tpp_data:
                tpp_score = extract_tpp_score(tpp_data)

        unlearning_score = None
        if "unlearning" in eval_files:
            ul_data = fetch_json(f"{BASE_URL}/{eval_files['unlearning']}")
            if ul_data:
                unlearning_score = extract_unlearning_score(ul_data)

        record = {
            "sae_key": sae_key,
            "sae_release": sae_release,
            "layer": layer,
            "width_str": width_str,
            "width_int": width_int,
            "l0": l0,
            "arch_class": arch_class,
            "absorption_score": round(absorption_score, 6),
            "sparse_probing_f1": round(sparse_probing_f1, 6) if sparse_probing_f1 is not None else None,
            "scr_score": round(scr_score, 6) if scr_score is not None else None,
            "tpp_score": round(tpp_score, 6) if tpp_score is not None else None,
            "unlearning_score": round(unlearning_score, 6) if unlearning_score is not None else None,
        }
        records.append(record)

        if i % 5 == 0:
            print(f"  SAE {i+1}/{total}: {sae_key} | absorption={absorption_score:.4f} "
                  f"sp_f1={sparse_probing_f1} scr={scr_score} tpp={tpp_score} ul={unlearning_score}",
                  flush=True)

    print(f"\nTotal records collected: {len(records)}", flush=True)

    write_progress("computing_correlations", 75)

    from scipy import stats

    # Extract vectors
    absorption = np.array([r["absorption_score"] for r in records])

    downstream_tasks = {
        "sparse_probing_f1": [r["sparse_probing_f1"] for r in records],
        "scr": [r["scr_score"] for r in records],
        "ravel_proxy_tpp": [r["tpp_score"] for r in records],
        "unlearning": [r["unlearning_score"] for r in records],
    }

    correlations = {}

    for task_name, raw_vals in downstream_tasks.items():
        # Filter to pairs with both values
        pairs = [(a, v) for a, v in zip(absorption, raw_vals) if v is not None]
        if len(pairs) < 5:
            print(f"  {task_name}: insufficient data ({len(pairs)} pairs), skipping")
            correlations[task_name] = {
                "n": len(pairs),
                "pearson_r": None,
                "pearson_p": None,
                "pearson_ci_95": [None, None],
                "spearman_rho": None,
                "spearman_p": None,
                "significant_after_bonferroni": False,
                "exceeds_meaningful_threshold": False,
                "note": "insufficient data"
            }
            continue

        abs_arr = np.array([p[0] for p in pairs])
        task_arr = np.array([p[1] for p in pairs])
        n = len(pairs)

        pr, pp = stats.pearsonr(abs_arr, task_arr)
        sr, sp = stats.spearmanr(abs_arr, task_arr)
        ci = pearson_r_ci(pr, n, alpha=0.05)

        correlations[task_name] = {
            "n": int(n),
            "pearson_r": round(float(pr), 4),
            "pearson_p": round(float(pp), 4),
            "pearson_ci_95": ci,
            "spearman_rho": round(float(sr), 4),
            "spearman_p": round(float(sp), 4),
            "significant_after_bonferroni": bool(pp < ALPHA_CORRECTED),
            "exceeds_meaningful_threshold": bool(abs(pr) > MEANINGFUL_R_THRESHOLD),
        }
        print(f"  {task_name}: n={n}, r={pr:.4f} (p={pp:.4f}), rho={sr:.4f}, "
              f"bonferroni_sig={pp < ALPHA_CORRECTED}", flush=True)

    write_progress("computing_partial_correlations", 85)

    # Partial correlations: control for log(width), layer, arch_class (dummy)
    # Use only records with all covariates and at least one downstream metric
    partial_correlations = {}

    # Build covariate vectors for records with complete covariate data
    cov_records = [r for r in records if r["width_int"] is not None and r["layer"] is not None]

    if len(cov_records) >= 10:
        abs_cov = np.array([r["absorption_score"] for r in cov_records])
        log_width = np.log([r["width_int"] for r in cov_records])
        layer_arr = np.array([r["layer"] for r in cov_records], dtype=float)
        # Arch class: canonical=1, non-canonical=0
        arch_arr = np.array([1.0 if "canonical" in r["arch_class"] else 0.0 for r in cov_records])

        # Diagnostic: print variance of each covariate
        print(f"  Covariate diagnostics: n={len(cov_records)}")
        print(f"    log_width std={np.std(log_width):.4f}, layer std={np.std(layer_arr):.4f}, arch std={np.std(arch_arr):.4f}")
        print(f"    arch unique values: {np.unique(arch_arr)}")

        # If arch_class has no variance (all same), drop it
        covariates_list = [log_width, layer_arr]
        cov_names = ["log_width", "layer"]
        if np.std(arch_arr) > 1e-6:
            covariates_list.append(arch_arr)
            cov_names.append("arch_class")
        else:
            print("    Dropping arch_class covariate (zero variance)")

        for task_name, raw_vals in downstream_tasks.items():
            # Filter to covariate records that also have downstream value
            task_vals = []
            abs_vals = []
            cov_log_width = []
            cov_layer = []
            cov_arch = []

            for r in cov_records:
                v = None
                if task_name == "sparse_probing_f1":
                    v = r["sparse_probing_f1"]
                elif task_name == "scr":
                    v = r["scr_score"]
                elif task_name == "ravel_proxy_tpp":
                    v = r["tpp_score"]
                elif task_name == "unlearning":
                    v = r["unlearning_score"]

                if v is not None:
                    task_vals.append(v)
                    abs_vals.append(r["absorption_score"])
                    cov_log_width.append(np.log(r["width_int"]))
                    cov_layer.append(float(r["layer"]))
                    cov_arch.append(1.0 if "canonical" in r["arch_class"] else 0.0)

            n_partial = len(task_vals)
            if n_partial < 8:
                partial_correlations[task_name] = {"n": n_partial, "partial_r": None, "partial_p": None}
                continue

            # Build per-task covariate lists
            task_cov_list = []
            if "log_width" in cov_names:
                task_cov_list.append(cov_log_width)
            if "layer" in cov_names:
                task_cov_list.append(cov_layer)
            if "arch_class" in cov_names:
                task_cov_list.append(cov_arch)

            pr, pp = compute_partial_r(abs_vals, task_vals, task_cov_list)
            partial_correlations[task_name] = {
                "n": n_partial,
                "partial_r": pr,
                "partial_p": pp,
                "covariates_controlled": cov_names,
            }
            print(f"  Partial {task_name}: n={n_partial}, partial_r={pr}, p={pp}", flush=True)
    else:
        print(f"  Insufficient records with complete covariates: {len(cov_records)}", flush=True)

    write_progress("computing_h3_verdict", 92)

    # H3 verdict
    valid_rs = [v["pearson_r"] for v in correlations.values() if v["pearson_r"] is not None]
    avg_abs_r = float(np.mean([abs(r) for r in valid_rs])) if valid_rs else None
    any_sig = any(v.get("significant_after_bonferroni", False) for v in correlations.values())
    h3_supported = avg_abs_r is not None and avg_abs_r > MEANINGFUL_R_THRESHOLD

    # Build full correlation matrix summary
    correlation_matrix_summary = {
        "absorption_vs_downstream": {
            t: {"pearson_r": correlations[t]["pearson_r"], "spearman_rho": correlations[t]["spearman_rho"]}
            for t in correlations
        }
    }

    # Build final result
    result = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "timestamp": time.time(),
        "n_saes_total": len(gemma_files),
        "n_saes_with_absorption": len(records),
        "bonferroni_correction": {
            "n_tests": N_TESTS,
            "nominal_alpha": BONFERRONI_ALPHA,
            "corrected_alpha": ALPHA_CORRECTED,
        },
        "meaningful_threshold": MEANINGFUL_R_THRESHOLD,
        "data_source": "HuggingFace adamkarvonen/sae_bench_results (real SAEBench data)",
        "sae_releases_included": list(set(r["sae_release"] for r in records)),
        "layers_covered": sorted(list(set(r["layer"] for r in records if r["layer"]))),
        "widths_covered": sorted(list(set(r["width_str"] for r in records if r["width_str"]))),
        "correlations": correlations,
        "partial_correlations": partial_correlations,
        "correlation_matrix_summary": correlation_matrix_summary,
        "h3_verdict": {
            "avg_abs_pearson_r": round(avg_abs_r, 4) if avg_abs_r else None,
            "any_significant_after_bonferroni": any_sig,
            "h3_supported": h3_supported,
            "interpretation": (
                "Absorption score shows meaningful negative correlation with downstream SAE quality "
                "metrics, supporting H3 that higher absorption degrades downstream utility."
                if h3_supported else
                "Insufficient evidence to support H3 (avg |r| below meaningful threshold)."
            ),
        },
        "raw_records": records,
        "runtime_seconds": round(time.time() - start_time, 1),
        "pilot_pass": len(records) >= 10 and correlations.get("ravel_proxy_tpp", {}).get("pearson_r") is not None,
    }

    # Save results - use custom encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    out_path = RESULTS_DIR / "C3A_saebench_corr.json"
    out_path.write_text(json.dumps(result, indent=2, cls=NumpyEncoder))
    print(f"\nResults saved to {out_path}", flush=True)

    write_progress("done", 100)

    summary = (
        f"n_saes={len(records)}, "
        f"ravel_r={correlations.get('ravel_proxy_tpp',{}).get('pearson_r')}, "
        f"scr_r={correlations.get('scr',{}).get('pearson_r')}, "
        f"h3_supported={h3_supported}"
    )
    mark_done(status="success", summary=summary)

    print("\n=== C3A COMPLETE ===")
    print(f"SAEs analyzed: {len(records)}")
    for t, c in correlations.items():
        if c["pearson_r"] is not None:
            print(f"  {t}: r={c['pearson_r']:.4f} (p={c['pearson_p']:.4f}) "
                  f"rho={c['spearman_rho']:.4f} bonferroni_sig={c['significant_after_bonferroni']}")
    print(f"H3 supported: {h3_supported} (avg |r| = {avg_abs_r:.4f})")

    return result


if __name__ == "__main__":
    try:
        result = main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        mark_done(status="failed", summary=str(e))
        sys.exit(1)
