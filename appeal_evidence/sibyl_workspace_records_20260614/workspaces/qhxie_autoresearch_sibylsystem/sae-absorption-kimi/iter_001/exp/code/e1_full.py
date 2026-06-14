#!/usr/bin/env python3
"""E1 Full: Downstream causal cost meta-analysis on Pythia-160M SAEBench data.

Uses real SAEBench metrics for absorption, core, sparse_probing, scr, and tpp.
Runs partial correlations and OLS regression with architecture-family controls.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASK_ID = "e1_full"
RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current/exp/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
E1_FULL_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current/exp/e1_full")
E1_FULL_DIR.mkdir(parents=True, exist_ok=True)

PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"

CACHE_DIR = Path("/home/qhxie/.cache/huggingface/hub/datasets--adamkarvonen--sae_bench_results_0125/snapshots/df1730fc7bd3ba7dc707b238e5fe098dba674caf")
HF_DATASET = "adamkarvonen/sae_bench_results_0125"

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Process tracking
# ---------------------------------------------------------------------------
PID_FILE.write_text(str(os.getpid()))

start_time_iso = datetime.now().isoformat()


def report_progress(epoch, total_epochs, step=0, total_steps=0, message=""):
    PROGRESS_FILE.write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "step": step,
                "total_steps": total_steps,
                "message": message,
                "updated_at": datetime.now().isoformat(),
            }
        )
    )


def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    progress = {}
    if PROGRESS_FILE.exists():
        try:
            progress = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    DONE_FILE.write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "status": status,
                "summary": summary,
                "final_progress": progress,
                "timestamp": datetime.now().isoformat(),
            }
        )
    )


# ---------------------------------------------------------------------------
# HF helpers
# ---------------------------------------------------------------------------
try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


def hf_download_file(repo_id, filename, repo_type="dataset", max_attempts=5):
    if not HF_AVAILABLE:
        return None
    for attempt in range(max_attempts):
        try:
            return hf_hub_download(repo_id, filename, repo_type=repo_type, local_dir_use_symlinks=False)
        except Exception as e:
            wait = 2 ** attempt + np.random.rand()
            print(f"  HF download attempt {attempt+1}/{max_attempts} failed for {filename}: {e}. Sleeping {wait:.1f}s...")
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def list_cached_json_files(eval_type):
    eval_dir = CACHE_DIR / eval_type
    if not eval_dir.exists():
        return []
    return sorted(eval_dir.rglob("*.json"))


def sae_id_from_path(path):
    basename = path.name
    basename = basename.replace("_eval_results.json", "")
    return basename


def parse_architecture(sid):
    parts = sid.split("_")
    for p in parts:
        if p in ["Standard", "TopK", "BatchTopK", "JumpRelu", "GatedSAE",
                 "MatryoshkaBatchTopK", "PAnneal", "OrtSAE", "Masked"]:
            return p
    return "Unknown"


def parse_model(sid):
    if "gemma-2-2b" in sid.lower():
        return "gemma-2-2b"
    if "pythia-160m" in sid.lower():
        return "pythia-160m"
    return "Unknown"


def parse_width(sid):
    m = re.search(r"width-(2pow\d+)", sid)
    if m:
        return 2 ** int(m.group(1).replace("2pow", ""))
    return None


def parse_layer(sid):
    m = re.search(r"layer_(\d+)", sid)
    if m:
        return int(m.group(1))
    return None


def parse_hook_point(sid):
    if "resid_pre" in sid:
        return "resid_pre"
    if "resid_post" in sid:
        return "resid_post"
    if "mlp_out" in sid:
        return "mlp_out"
    if "attn_out" in sid:
        return "attn_out"
    return "unknown"


def parse_release_dir(sid):
    """Extract release directory from SAE ID (everything up to and including date-XXXX)."""
    m = re.search(r"^(.*date-\d+)", sid)
    if m:
        return m.group(1)
    # Fallback: use architecture as delimiter
    arch = parse_architecture(sid)
    if arch != "Unknown" and f"_{arch}_" in sid:
        return sid.split(f"_{arch}_")[0]
    return sid


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------
def extract_absorption(path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    results = data.get("eval_result_metrics", data)
    if not isinstance(results, dict):
        return {}
    if "mean" in results and isinstance(results["mean"], dict):
        mean = results["mean"]
        mean_full = mean.get("mean_full_absorption_score")
        mean_frac = mean.get("mean_absorption_fraction_score")
        return {"absorption_mean": mean_full if mean_full is not None else mean_frac}
    mean_full = results.get("mean_full_absorption_score")
    mean_frac = results.get("mean_absorption_fraction_score")
    return {"absorption_mean": mean_full if mean_full is not None else mean_frac}


def extract_core(path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    results = data.get("eval_result_metrics", data)
    if not isinstance(results, dict):
        return {}
    sparsity = results.get("sparsity", {})
    recon = results.get("reconstruction_quality", {})
    perf = results.get("model_performance_preservation", {})
    return {
        "l0": sparsity.get("l0"),
        "ce_loss_recovered": perf.get("ce_loss_score"),
        "explained_variance": recon.get("explained_variance"),
    }


def extract_sparse_probing(path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    results = data.get("eval_result_metrics", {})
    sae = results.get("sae", {})
    # Use top-1 test accuracy as the most stringent sparse probing metric
    return {
        "sparse_probing_acc": sae.get("sae_test_accuracy"),
        "sparse_probing_top1_acc": sae.get("sae_top_1_test_accuracy"),
    }


def extract_scr(path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    results = data.get("eval_result_metrics", {})
    scr = results.get("scr_metrics", {})
    return {
        "scr_metric_10": scr.get("scr_metric_threshold_10"),
        "scr_metric_20": scr.get("scr_metric_threshold_20"),
        "scr_metric_50": scr.get("scr_metric_threshold_50"),
    }


def extract_tpp(path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    results = data.get("eval_result_metrics", {})
    tpp = results.get("tpp_metrics", {})
    return {
        "tpp_metric_10": tpp.get("tpp_threshold_10_total_metric"),
        "tpp_metric_20": tpp.get("tpp_threshold_20_total_metric"),
        "tpp_metric_50": tpp.get("tpp_threshold_50_total_metric"),
    }


# ---------------------------------------------------------------------------
# Download missing eval files
# ---------------------------------------------------------------------------
def get_eval_file(eval_type, sid):
    """Find eval file in cache or download from HF using correct release dir."""
    release_dir = parse_release_dir(sid)
    eval_dir = CACHE_DIR / eval_type
    # Check cache
    if eval_dir.exists():
        candidate = eval_dir / release_dir / f"{sid}_eval_results.json"
        if candidate.exists():
            return candidate
    # Download from HF
    hf_path = f"{eval_type}/{release_dir}/{sid}_eval_results.json"
    downloaded = hf_download_file(HF_DATASET, hf_path, repo_type="dataset")
    if downloaded:
        return Path(downloaded)
    return None


def download_missing_files(eval_type, sids, max_workers=8):
    """Download missing eval files for given SAE IDs."""
    missing = []
    for sid in sids:
        f = get_eval_file(eval_type, sid)
        if f is None or not f.exists():
            missing.append(sid)

    if not missing:
        return {}

    def try_download(sid):
        release_dir = parse_release_dir(sid)
        hf_path = f"{eval_type}/{release_dir}/{sid}_eval_results.json"
        downloaded = hf_download_file(HF_DATASET, hf_path, repo_type="dataset")
        if downloaded:
            return sid, Path(downloaded)
        return sid, None

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(try_download, sid): sid for sid in missing}
        for future in as_completed(futures):
            sid, path = future.result()
            if path:
                results[sid] = path
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    report_progress(1, 6, message="Loading cached absorption and core data")
    print("Loading cached absorption and core data...")

    abs_files = list_cached_json_files("absorption")
    core_files = list_cached_json_files("core")

    # Build maps
    def build_map(files):
        return {sae_id_from_path(f): f for f in files}

    abs_map = build_map(abs_files)
    core_map = build_map(core_files)

    # Focus on Pythia-160M
    common_ids = sorted(set(abs_map) & set(core_map))
    pythia_ids = [sid for sid in common_ids if parse_model(sid) == "pythia-160m"]
    print(f"Total common SAEs: {len(common_ids)}; Pythia-160M: {len(pythia_ids)}")

    sample_ids = pythia_ids
    print(f"Processing {len(sample_ids)} Pythia-160M SAEs")

    report_progress(2, 6, message="Downloading sparse_probing, scr, tpp data")
    print("Downloading missing sparse_probing files...")
    sp_downloaded = download_missing_files("sparse_probing", sample_ids)
    print(f"  Downloaded {len(sp_downloaded)} sparse_probing files")

    print("Downloading missing scr files...")
    scr_downloaded = download_missing_files("scr", sample_ids)
    print(f"  Downloaded {len(scr_downloaded)} scr files")

    print("Downloading missing tpp files...")
    tpp_downloaded = download_missing_files("tpp", sample_ids)
    print(f"  Downloaded {len(tpp_downloaded)} tpp files")

    # Rebuild maps after downloads
    sp_map = build_map(list_cached_json_files("sparse_probing"))
    scr_map = build_map(list_cached_json_files("scr"))
    tpp_map = build_map(list_cached_json_files("tpp"))

    report_progress(3, 6, step=0, total_steps=len(sample_ids), message="Parsing metrics")

    records = []
    for i, sid in enumerate(sample_ids):
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(sample_ids)}] ...")
            report_progress(3, 6, step=i+1, total_steps=len(sample_ids), message="Parsing metrics")

        metrics = {}
        metrics.update(extract_absorption(abs_map[sid]))
        metrics.update(extract_core(core_map[sid]))

        if sid in sp_map:
            metrics.update(extract_sparse_probing(sp_map[sid]))
        if sid in scr_map:
            metrics.update(extract_scr(scr_map[sid]))
        if sid in tpp_map:
            metrics.update(extract_tpp(tpp_map[sid]))

        metrics["sae_id"] = sid
        metrics["architecture"] = parse_architecture(sid)
        metrics["model"] = parse_model(sid)
        metrics["width"] = parse_width(sid)
        metrics["layer"] = parse_layer(sid)
        metrics["hook_point"] = parse_hook_point(sid)
        records.append(metrics)

    df = pd.DataFrame(records)
    print(f"Collected data for {len(df)} SAEs")

    # Save raw data
    raw_path = E1_FULL_DIR / "regression_raw_data.json"
    raw_path.write_text(df.to_json(orient="records", indent=2))

    report_progress(4, 6, message="Running correlations and regressions")

    # Prepare numeric columns
    numeric_cols = [
        "absorption_mean", "l0", "ce_loss_recovered", "explained_variance",
        "sparse_probing_acc", "sparse_probing_top1_acc",
        "scr_metric_10", "scr_metric_20", "scr_metric_50",
        "tpp_metric_10", "tpp_metric_20", "tpp_metric_50",
        "width", "layer",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_valid = df.dropna(subset=["absorption_mean"]).copy()
    print(f"Rows with absorption data: {len(df_valid)}")

    # Descriptive stats
    desc = df_valid[numeric_cols].describe().to_dict()

    # Correlations
    def safe_corr(x, y):
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 5:
            return None, None
        r, p = stats.pearsonr(x[mask], y[mask])
        return r, p

    downstream_outcomes = ["sparse_probing_acc", "sparse_probing_top1_acc", "scr_metric_10", "tpp_metric_10"]
    correlations = {}
    for outcome in downstream_outcomes:
        if outcome in df_valid.columns and df_valid[outcome].notna().sum() > 5:
            r, p = safe_corr(df_valid["absorption_mean"].values, df_valid[outcome].values)
            correlations[outcome] = {"pearson_r": r, "p_value": p, "n": int(df_valid[outcome].notna().sum())}

    # Partial correlations (controlling for L0 and CE loss recovered)
    partial_correlations = {}
    for outcome in downstream_outcomes:
        cols = ["absorption_mean", outcome, "l0", "ce_loss_recovered"]
        sub = df_valid[cols].dropna()
        if len(sub) >= 10:
            X_ctrl = sub[["l0", "ce_loss_recovered"]].values
            reg_abs = LinearRegression().fit(X_ctrl, sub["absorption_mean"].values)
            reg_out = LinearRegression().fit(X_ctrl, sub[outcome].values)
            resid_abs = sub["absorption_mean"].values - reg_abs.predict(X_ctrl)
            resid_out = sub[outcome].values - reg_out.predict(X_ctrl)
            r, p = stats.pearsonr(resid_abs, resid_out)
            partial_correlations[outcome] = {
                "partial_r": float(r),
                "p_value": float(p),
                "n": int(len(sub)),
            }

    # OLS regression with architecture dummies, width, layer
    regression_results = {}
    for outcome in downstream_outcomes:
        cols = [outcome, "absorption_mean", "l0", "ce_loss_recovered", "width", "layer", "architecture"]
        sub = df_valid[cols].dropna()
        if len(sub) >= 10:
            X = pd.get_dummies(sub[["absorption_mean", "l0", "ce_loss_recovered", "width", "layer", "architecture"]], drop_first=True)
            y = sub[outcome].values.astype(np.float64)
            X = X.astype(np.float64)
            cont_cols = [c for c in X.columns if not str(c).startswith("architecture_")]
            scaler = StandardScaler()
            X_scaled = X.copy()
            if len(cont_cols) > 0:
                X_scaled[cont_cols] = scaler.fit_transform(X[cont_cols])

            model = LinearRegression().fit(X_scaled, y)
            preds = model.predict(X_scaled)
            residuals = y - preds
            mse = np.mean(residuals ** 2)
            XtX = X_scaled.values.T @ X_scaled.values
            XtX_inv = np.linalg.pinv(XtX)
            se = np.sqrt(mse * np.diag(XtX_inv))
            t_stats = model.coef_ / (se + 1e-12)
            df_resid = len(y) - X_scaled.shape[1] - 1
            p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), max(1, df_resid)))

            coef_table = []
            for name, coef, se_val, t_val, p_val in zip(X_scaled.columns, model.coef_, se, t_stats, p_vals):
                coef_table.append({
                    "variable": str(name),
                    "coefficient": float(coef),
                    "se": float(se_val),
                    "t_stat": float(t_val),
                    "p_value": float(p_val),
                })

            regression_results[outcome] = {
                "r2": float(model.score(X_scaled, y)),
                "n": int(len(y)),
                "coefficients": coef_table,
            }

    report_progress(5, 6, message="Saving results")

    results = {
        "task_id": TASK_ID,
        "n_saes_total": int(len(df)),
        "n_saes_with_absorption": int(len(df_valid)),
        "sample_ids": sample_ids,
        "descriptive_stats": desc,
        "correlations": correlations,
        "partial_correlations": partial_correlations,
        "regression_results": regression_results,
        "data_source": "SAEBench HF dataset (adamkarvonen/sae_bench_results_0125)",
        "note": "Real SAEBench metrics used for absorption, core, sparse_probing, scr, and tpp on Pythia-160M. RAVEL unavailable for Pythia-160M; SCR and TPP used as additional downstream interpretability proxies.",
    }

    out_json = E1_FULL_DIR / "regression_results.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"Results saved to {out_json}")

    # Summary markdown
    md_lines = [
        "# E1 Full: Downstream Causal Cost Meta-Analysis (Pythia-160M)",
        "",
        f"**Task:** {TASK_ID}  ",
        f"**SAEs analyzed:** {len(df_valid)} (Pythia-160M from SAEBench)  ",
        f"**Data source:** {results['data_source']}  ",
        "",
        "## Descriptive Statistics",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|--------|------|-----|-----|-----|",
    ]
    for metric, vals in desc.items():
        md_lines.append(f"| {metric} | {vals.get('mean', 0):.4f} | {vals.get('std', 0):.4f} | {vals.get('min', 0):.4f} | {vals.get('max', 0):.4f} |")

    md_lines.extend([
        "",
        "## Correlations (absorption vs downstream)",
        "",
        "| Outcome | Pearson r | p-value | N |",
        "|---------|-----------|---------|---|",
    ])
    for outcome, vals in correlations.items():
        r_str = f"{vals['pearson_r']:.3f}" if vals['pearson_r'] is not None else "N/A"
        p_str = f"{vals['p_value']:.3f}" if vals['p_value'] is not None else "N/A"
        md_lines.append(f"| {outcome} | {r_str} | {p_str} | {vals['n']} |")

    md_lines.extend([
        "",
        "## Partial Correlations (controlling for L0 and CE loss recovered)",
        "",
        "| Outcome | Partial r | p-value | N |",
        "|---------|-----------|---------|---|",
    ])
    for outcome, vals in partial_correlations.items():
        md_lines.append(f"| {outcome} | {vals['partial_r']:.3f} | {vals['p_value']:.3f} | {vals['n']} |")

    md_lines.extend([
        "",
        "## OLS Regression Summaries",
        "",
    ])
    for outcome, reg in regression_results.items():
        md_lines.extend([
            f"### {outcome} (R² = {reg['r2']:.3f}, N = {reg['n']})",
            "",
            "| Variable | Coefficient | SE | t-stat | p-value |",
            "|----------|-------------|----|--------|---------|",
        ])
        for row in reg["coefficients"]:
            md_lines.append(
                f"| {row['variable']} | {row['coefficient']:.4f} | {row['se']:.4f} | {row['t_stat']:.3f} | {row['p_value']:.3f} |"
            )
        md_lines.append("")

    md_path = E1_FULL_DIR / "summary.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Summary saved to {md_path}")

    # Update gpu_progress
    gp_path = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current/exp/gpu_progress.json")
    if gp_path.exists():
        gp = json.loads(gp_path.read_text())
    else:
        gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

    start_time_dt = datetime.fromisoformat(start_time_iso)
    end_time_dt = datetime.now()
    actual_min = max(1, int((end_time_dt - start_time_dt).total_seconds() / 60))

    gp["completed"] = list(dict.fromkeys(gp.get("completed", []) + [TASK_ID]))
    gp["running"].pop(TASK_ID, None)
    gp["timings"][TASK_ID] = {
        "planned_min": 10,
        "actual_min": actual_min,
        "start_time": start_time_iso,
        "end_time": end_time_dt.isoformat(),
        "config_snapshot": {
            "sample_size": int(len(df_valid)),
            "task": "meta_analysis",
            "gpu_count": 0,
            "model": "pythia-160m",
        },
    }
    gp_path.write_text(json.dumps(gp, indent=2))

    # Update experiment_state
    exp_state_path = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current/exp/experiment_state.json")
    if exp_state_path.exists():
        exp_state = json.loads(exp_state_path.read_text())
        if TASK_ID in exp_state.get("tasks", {}):
            exp_state["tasks"][TASK_ID]["status"] = "completed"
            exp_state["tasks"][TASK_ID]["completed_at"] = end_time_dt.isoformat()
        exp_state_path.write_text(json.dumps(exp_state, indent=2))

    report_progress(6, 6, message="Done")
    mark_done(
        status="success",
        summary=f"E1 full meta-analysis completed on {len(df_valid)} Pythia-160M SAEBench SAEs. Partial correlations and OLS regressions saved."
    )
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        mark_done(status="failed", summary=str(e))
        raise
