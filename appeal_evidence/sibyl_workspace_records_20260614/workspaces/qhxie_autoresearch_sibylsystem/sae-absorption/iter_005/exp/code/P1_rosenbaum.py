"""
P1_rosenbaum: Rosenbaum Sensitivity Analysis with Propensity Matching

Multiple matching strategies to handle the width/absorption confound:
1. Exact width + NN L0 (strictest, may have few pairs)
2. Median-split within width stratum (uses all data within each width)
3. Propensity score matching (most flexible, matches on propensity of being high-absorption)
4. Mahalanobis distance matching on (log_width, log_L0, layer)

For each matching strategy, runs Wilcoxon signed-rank tests and Rosenbaum sensitivity.

Pilot mode: full 48-SAE dataset (with known L0).
Pass criteria: At least 5 matched pairs formed. Gamma > 1.0.
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

TASK_ID = "P1_rosenbaum"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
DATA_PATH = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/iter_004/exp/results/full/C3A_saebench_corr.json")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
SEED = 42

np.random.seed(SEED)


def write_progress(task_id, results_dir, epoch, total_epochs, step=0,
                   total_steps=0, loss=None, metric=None):
    """Write progress file for system monitor to track."""
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id, results_dir, status="success", summary=""):
    """Write DONE marker file for system monitor to detect."""
    pid_file = Path(results_dir) / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
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


def load_data():
    """Load and prepare the 54-SAE dataset, filtering to those with known L0."""
    with open(DATA_PATH) as f:
        data = json.load(f)

    records = data["raw_records"]
    df = pd.DataFrame(records)

    # Filter to SAEs with known L0
    df = df[df["l0"].notna()].copy()
    df["log_l0"] = np.log(df["l0"])
    df["log_width"] = np.log(df["width_int"])
    df.reset_index(drop=True, inplace=True)

    return df


# ============================================================
# MATCHING STRATEGIES
# ============================================================

def strategy_exact_width_nn_l0(df, high_thresh=0.3, low_thresh=0.1, caliper_sd=0.5):
    """Strategy 1: Exact match on width, nearest neighbor on log(L0)."""
    high = df[df["absorption_score"] > high_thresh].copy()
    low = df[df["absorption_score"] < low_thresh].copy()

    l0_std = df["log_l0"].std()
    caliper = caliper_sd * l0_std

    matched = []
    used_low = set()

    for _, h in high.iterrows():
        cands = low[(low["width_str"] == h["width_str"]) &
                     (~low.index.isin(used_low))]
        if len(cands) == 0:
            continue
        dists = np.abs(cands["log_l0"] - h["log_l0"])
        best = dists.idxmin()
        if dists[best] <= caliper:
            matched.append(_make_pair(h, cands.loc[best], dists[best], "exact_width"))
            used_low.add(best)

    return matched, {
        "strategy": "exact_width_nn_l0",
        "high_threshold": high_thresh,
        "low_threshold": low_thresh,
        "caliper_sd": caliper_sd,
        "caliper_log_l0": float(caliper),
        "n_high": len(high),
        "n_low": len(low),
        "n_matched": len(matched),
    }


def strategy_within_width_median_split(df):
    """
    Strategy 2: Within each width stratum, split by median absorption,
    then match above-median to below-median using NN on L0.
    This maximizes sample size by using all data within each width group.
    """
    matched = []
    stratum_info = {}

    for width in df["width_str"].unique():
        subset = df[df["width_str"] == width].copy()
        median_abs = subset["absorption_score"].median()

        above = subset[subset["absorption_score"] > median_abs].copy()
        below = subset[subset["absorption_score"] <= median_abs].copy()

        stratum_info[width] = {
            "n_total": len(subset),
            "median_absorption": float(median_abs),
            "n_above": len(above),
            "n_below": len(below),
        }

        used_below = set()
        for _, h in above.iterrows():
            cands = below[~below.index.isin(used_below)]
            if len(cands) == 0:
                break
            dists = np.abs(cands["log_l0"] - h["log_l0"])
            best = dists.idxmin()
            matched.append(_make_pair(h, cands.loc[best], dists[best], "median_split"))
            used_below.add(best)

    return matched, {
        "strategy": "within_width_median_split",
        "strata": stratum_info,
        "n_matched": len(matched),
    }


def strategy_propensity_score(df, high_thresh=0.3, low_thresh=0.1, caliper_sd=0.25):
    """
    Strategy 3: Propensity score matching.
    Model P(high_absorption) from width, L0, layer.
    Match on propensity scores without requiring exact width match.
    """
    df_filt = df[(df["absorption_score"] > high_thresh) |
                 (df["absorption_score"] < low_thresh)].copy()
    if len(df_filt) < 5:
        return [], {"strategy": "propensity_score", "error": "Too few SAEs", "n": len(df_filt)}

    df_filt["treatment"] = (df_filt["absorption_score"] > high_thresh).astype(int)

    X = df_filt[["log_width", "log_l0", "layer"]].values
    y = df_filt["treatment"].values

    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X, y)
    df_filt["propensity"] = lr.predict_proba(X)[:, 1]

    treated = df_filt[df_filt["treatment"] == 1]
    control = df_filt[df_filt["treatment"] == 0]

    p_std = df_filt["propensity"].std()
    caliper = caliper_sd * p_std if p_std > 0 else 0.1

    matched = []
    used_ctrl = set()

    for _, t in treated.iterrows():
        cands = control[~control.index.isin(used_ctrl)]
        if len(cands) == 0:
            break
        dists = np.abs(cands["propensity"] - t["propensity"])
        best = dists.idxmin()
        if dists[best] <= caliper:
            pair = _make_pair(t, cands.loc[best], dists[best], "propensity")
            pair["high_propensity"] = float(t["propensity"])
            pair["low_propensity"] = float(cands.loc[best]["propensity"])
            matched.append(pair)
            used_ctrl.add(best)

    return matched, {
        "strategy": "propensity_score",
        "high_threshold": high_thresh,
        "low_threshold": low_thresh,
        "n_treated": len(treated),
        "n_control": len(control),
        "n_matched": len(matched),
        "caliper_propensity": float(caliper),
        "lr_coefs": {
            "log_width": float(lr.coef_[0][0]),
            "log_l0": float(lr.coef_[0][1]),
            "layer": float(lr.coef_[0][2]),
            "intercept": float(lr.intercept_[0]),
        },
    }


def strategy_mahalanobis(df, high_thresh=0.3, low_thresh=0.1, caliper_factor=2.0):
    """
    Strategy 4: Mahalanobis distance matching on (log_width, log_L0, layer).
    """
    high = df[df["absorption_score"] > high_thresh].copy()
    low = df[df["absorption_score"] < low_thresh].copy()

    if len(high) < 2 or len(low) < 2:
        return [], {"strategy": "mahalanobis", "error": "Too few in groups",
                    "n_high": len(high), "n_low": len(low)}

    features = ["log_width", "log_l0", "layer"]
    combined = pd.concat([high, low])[features]
    cov_mat = combined.cov().values
    try:
        cov_inv = np.linalg.inv(cov_mat)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_mat)

    # Compute median Mahalanobis distance as caliper reference
    all_dists = []
    for _, h in high.iterrows():
        for _, l in low.iterrows():
            d = mahalanobis(h[features].values, l[features].values, cov_inv)
            all_dists.append(d)
    median_dist = np.median(all_dists)
    caliper = caliper_factor * median_dist

    matched = []
    used_low = set()

    for _, h in high.iterrows():
        cands = low[~low.index.isin(used_low)]
        if len(cands) == 0:
            break
        dists = cands.apply(
            lambda row: mahalanobis(h[features].values, row[features].values, cov_inv),
            axis=1
        )
        best = dists.idxmin()
        if dists[best] <= caliper:
            matched.append(_make_pair(h, cands.loc[best], dists[best], "mahalanobis"))
            used_low.add(best)

    return matched, {
        "strategy": "mahalanobis",
        "high_threshold": high_thresh,
        "low_threshold": low_thresh,
        "n_high": len(high),
        "n_low": len(low),
        "n_matched": len(matched),
        "median_mahalanobis_dist": float(median_dist),
        "caliper": float(caliper),
    }


def strategy_tertile_within_width(df):
    """
    Strategy 5: Tertile split within each width stratum.
    Compare top third vs bottom third of absorption within each width.
    """
    matched = []
    stratum_info = {}

    for width in df["width_str"].unique():
        subset = df[df["width_str"] == width].copy()
        if len(subset) < 6:
            stratum_info[width] = {"n_total": len(subset), "skipped": True,
                                    "reason": "too few SAEs"}
            continue

        q33 = subset["absorption_score"].quantile(0.33)
        q67 = subset["absorption_score"].quantile(0.67)

        high = subset[subset["absorption_score"] >= q67].copy()
        low = subset[subset["absorption_score"] <= q33].copy()

        stratum_info[width] = {
            "n_total": len(subset),
            "q33": float(q33),
            "q67": float(q67),
            "n_high": len(high),
            "n_low": len(low),
        }

        used_low = set()
        for _, h in high.iterrows():
            cands = low[~low.index.isin(used_low)]
            if len(cands) == 0:
                break
            dists = np.abs(cands["log_l0"] - h["log_l0"])
            best = dists.idxmin()
            matched.append(_make_pair(h, cands.loc[best], dists[best], "tertile"))
            used_low.add(best)

    return matched, {
        "strategy": "tertile_within_width",
        "strata": stratum_info,
        "n_matched": len(matched),
    }


def _make_pair(h_row, l_row, distance, method):
    """Create a pair dict from high-absorption and low-absorption rows."""
    return {
        "method": method,
        "distance": float(distance),
        "high_sae": h_row["sae_key"],
        "low_sae": l_row["sae_key"],
        "high_width": h_row["width_str"],
        "low_width": l_row["width_str"],
        "high_absorption": float(h_row["absorption_score"]),
        "low_absorption": float(l_row["absorption_score"]),
        "absorption_diff": float(h_row["absorption_score"] - l_row["absorption_score"]),
        "high_l0": float(h_row["l0"]),
        "low_l0": float(l_row["l0"]),
        "high_layer": int(h_row["layer"]),
        "low_layer": int(l_row["layer"]),
        "high_sparse_probing_f1": _safe_float(h_row.get("sparse_probing_f1")),
        "low_sparse_probing_f1": _safe_float(l_row.get("sparse_probing_f1")),
        "high_scr_score": _safe_float(h_row.get("scr_score")),
        "low_scr_score": _safe_float(l_row.get("scr_score")),
        "high_tpp_score": _safe_float(h_row.get("tpp_score")),
        "low_tpp_score": _safe_float(l_row.get("tpp_score")),
        "high_unlearning_score": _safe_float(h_row.get("unlearning_score")),
        "low_unlearning_score": _safe_float(l_row.get("unlearning_score")),
    }


def _safe_float(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return float(val)


# ============================================================
# STATISTICAL TESTS
# ============================================================

def wilcoxon_test(pairs, metric_name):
    """
    Wilcoxon signed-rank test on quality differences.
    Positive diff = low-absorption SAE has better quality (expected direction).
    """
    valid = [(p[f"low_{metric_name}"], p[f"high_{metric_name}"])
             for p in pairs
             if p[f"low_{metric_name}"] is not None and p[f"high_{metric_name}"] is not None]

    if len(valid) < 2:
        return {"metric": metric_name, "n_pairs": len(valid),
                "error": f"Too few valid pairs ({len(valid)})"}

    diffs = np.array([lo - hi for lo, hi in valid])

    # Check if all diffs are zero
    if np.all(diffs == 0):
        return {"metric": metric_name, "n_pairs": len(valid),
                "error": "All differences are zero"}

    try:
        stat, p_val = stats.wilcoxon(diffs, alternative="two-sided")
    except ValueError as e:
        return {"metric": metric_name, "n_pairs": len(valid), "error": str(e)}

    return {
        "metric": metric_name,
        "n_pairs": len(valid),
        "mean_diff": float(np.mean(diffs)),
        "median_diff": float(np.median(diffs)),
        "std_diff": float(np.std(diffs, ddof=1)) if len(valid) > 1 else 0.0,
        "wilcoxon_stat": float(stat),
        "p_value": float(p_val),
        "significant_005": bool(p_val < 0.05),
        "direction": "low_absorption_better" if np.mean(diffs) > 0 else "high_absorption_better",
        "diffs": [float(d) for d in diffs],
    }


def rosenbaum_sensitivity(pairs, metric_name, gamma_max=5.0, gamma_step=0.05):
    """
    Rosenbaum sensitivity analysis.

    For matched pairs with quality difference d_i, under hidden bias Gamma,
    the probability that pair i is assigned to treatment (high absorption)
    lies in [1/(1+Gamma), Gamma/(1+Gamma)].

    We compute the worst-case p-value at each Gamma and find the critical Gamma
    where the result first becomes non-significant at alpha=0.05.
    """
    valid = [(p[f"low_{metric_name}"], p[f"high_{metric_name}"])
             for p in pairs
             if p[f"low_{metric_name}"] is not None and p[f"high_{metric_name}"] is not None]

    diffs = np.array([lo - hi for lo, hi in valid])
    n = len(diffs)

    if n < 2:
        return {"metric": metric_name, "error": "Too few pairs", "n": n}

    # Zero diffs are excluded from Wilcoxon
    nonzero = diffs[diffs != 0]
    if len(nonzero) < 1:
        return {"metric": metric_name, "error": "All diffs are zero", "n": n}

    abs_diffs = np.abs(nonzero)
    signs = np.sign(nonzero)
    ranks = stats.rankdata(abs_diffs)

    T_plus = float(np.sum(ranks[signs > 0]))
    sum_ranks = float(np.sum(ranks))
    sum_ranks_sq = float(np.sum(ranks ** 2))

    gammas = np.arange(1.0, gamma_max + gamma_step, gamma_step)
    gamma_p_values = []
    critical_gamma = None

    for gamma in gammas:
        p_pos = gamma / (1.0 + gamma)

        E_T = sum_ranks * p_pos
        Var_T = sum_ranks_sq * p_pos * (1 - p_pos)

        if Var_T > 0:
            z = (T_plus - E_T) / np.sqrt(Var_T)
            # Upper bound p-value (testing if positive diffs could be explained by bias)
            p_upper = 1.0 - stats.norm.cdf(z)
        else:
            p_upper = 1.0

        gamma_p_values.append({
            "gamma": float(round(gamma, 2)),
            "p_value": float(p_upper),
            "significant_005": bool(p_upper < 0.05),
        })

        if critical_gamma is None and p_upper >= 0.05:
            critical_gamma = float(round(gamma, 2))

    if critical_gamma is None:
        critical_gamma = float(gammas[-1])
        at_boundary = True
    else:
        at_boundary = False

    return {
        "metric": metric_name,
        "n_pairs": n,
        "n_nonzero": len(nonzero),
        "T_plus": T_plus,
        "sum_ranks": sum_ranks,
        "critical_gamma": critical_gamma,
        "at_boundary": at_boundary,
        "interpretation": _interpret_gamma(critical_gamma, at_boundary),
        "gamma_p_values": gamma_p_values,
    }


def _interpret_gamma(gamma, at_boundary):
    if at_boundary:
        return f"Robust to Gamma >= {gamma:.1f} (exceeded test range). Very strong evidence."
    elif gamma < 1.05:
        return f"Gamma={gamma:.2f}: Not significant even without hidden bias."
    elif gamma < 1.2:
        return f"Gamma={gamma:.2f}: Fragile. Small hidden bias could explain result."
    elif gamma < 1.5:
        return f"Gamma={gamma:.2f}: Moderate sensitivity."
    elif gamma < 2.0:
        return f"Gamma={gamma:.2f}: Moderate robustness."
    else:
        return f"Gamma={gamma:.2f}: Strong robustness to hidden bias."


def matching_quality(pairs, strategy_name):
    """Compute balance diagnostics for matched pairs."""
    if not pairs:
        return {"strategy": strategy_name, "n": 0, "error": "No pairs"}

    df_pairs = pd.DataFrame(pairs)
    diag = {"strategy": strategy_name, "n_pairs": len(pairs)}

    # L0 balance
    high_l0 = df_pairs["high_l0"].values
    low_l0 = df_pairs["low_l0"].values
    pooled_sd = np.sqrt((np.var(high_l0, ddof=1) + np.var(low_l0, ddof=1)) / 2) if len(high_l0) > 1 else 1.0
    smd = (np.mean(high_l0) - np.mean(low_l0)) / pooled_sd if pooled_sd > 0 else float("inf")
    diag["l0_balance"] = {
        "mean_high": float(np.mean(high_l0)),
        "mean_low": float(np.mean(low_l0)),
        "smd": float(smd),
        "balanced_025": bool(abs(smd) < 0.25),
    }

    # Width balance
    width_cross = pd.crosstab(df_pairs["high_width"], df_pairs["low_width"])
    exact_width_match = sum(1 for p in pairs if p["high_width"] == p["low_width"])
    diag["width_balance"] = {
        "exact_match_count": exact_width_match,
        "exact_match_pct": float(exact_width_match / len(pairs) * 100),
    }

    # Absorption gap
    diag["absorption_gap"] = {
        "mean_high": float(df_pairs["high_absorption"].mean()),
        "mean_low": float(df_pairs["low_absorption"].mean()),
        "mean_diff": float(df_pairs["absorption_diff"].mean()),
    }

    # Layer balance
    diag["layer_balance"] = {
        "same_layer_count": sum(1 for p in pairs if p["high_layer"] == p["low_layer"]),
        "same_layer_pct": float(sum(1 for p in pairs if p["high_layer"] == p["low_layer"]) / len(pairs) * 100),
    }

    return diag


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()
    start_dt = datetime.now()

    # Write PID file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

    write_progress(TASK_ID, RESULTS_DIR, 0, 6,
                   metric={"status": "loading_data"})

    print(f"[{TASK_ID}] Starting Rosenbaum Sensitivity Analysis (Multi-Strategy)")
    df = load_data()
    print(f"[{TASK_ID}] Loaded {len(df)} SAEs with known L0")

    quality_metrics = ["sparse_probing_f1", "scr_score", "tpp_score", "unlearning_score"]
    all_strategy_results = {}

    # ---- Strategy 1: Exact width + NN L0 ----
    write_progress(TASK_ID, RESULTS_DIR, 1, 6,
                   metric={"status": "strategy_1_exact_width"})
    print(f"\n[{TASK_ID}] Strategy 1: Exact width + NN L0 matching")

    pairs1, diag1 = strategy_exact_width_nn_l0(df, high_thresh=0.3, low_thresh=0.1, caliper_sd=1.0)
    print(f"  {diag1['n_matched']} pairs (thresh >0.3 / <0.1)")

    # Try relaxed thresholds
    if len(pairs1) < 3:
        pairs1b, diag1b = strategy_exact_width_nn_l0(df, high_thresh=0.25, low_thresh=0.15, caliper_sd=1.5)
        print(f"  Relaxed: {diag1b['n_matched']} pairs (thresh >0.25 / <0.15)")
        if len(pairs1b) > len(pairs1):
            pairs1, diag1 = pairs1b, diag1b

    mq1 = matching_quality(pairs1, "exact_width_nn_l0")
    w1 = {m: wilcoxon_test(pairs1, m) for m in quality_metrics} if len(pairs1) >= 2 else {}
    r1 = {m: rosenbaum_sensitivity(pairs1, m) for m in quality_metrics} if len(pairs1) >= 2 else {}

    all_strategy_results["exact_width_nn_l0"] = {
        "diagnostics": diag1, "matching_quality": mq1,
        "pairs": pairs1, "wilcoxon": w1, "rosenbaum": r1,
    }

    # ---- Strategy 2: Within-width median split ----
    write_progress(TASK_ID, RESULTS_DIR, 2, 6,
                   metric={"status": "strategy_2_median_split"})
    print(f"\n[{TASK_ID}] Strategy 2: Within-width median split")

    pairs2, diag2 = strategy_within_width_median_split(df)
    print(f"  {diag2['n_matched']} pairs")

    mq2 = matching_quality(pairs2, "median_split")
    w2 = {m: wilcoxon_test(pairs2, m) for m in quality_metrics}
    r2 = {m: rosenbaum_sensitivity(pairs2, m) for m in quality_metrics}

    all_strategy_results["within_width_median_split"] = {
        "diagnostics": diag2, "matching_quality": mq2,
        "pairs": pairs2, "wilcoxon": w2, "rosenbaum": r2,
    }

    for m in quality_metrics:
        if "p_value" in w2.get(m, {}):
            gamma = r2[m].get("critical_gamma", "N/A")
            print(f"  {m}: diff={w2[m]['mean_diff']:.4f}, p={w2[m]['p_value']:.4f}, Gamma={gamma}")

    # ---- Strategy 3: Propensity score matching ----
    write_progress(TASK_ID, RESULTS_DIR, 3, 6,
                   metric={"status": "strategy_3_propensity"})
    print(f"\n[{TASK_ID}] Strategy 3: Propensity score matching")

    pairs3, diag3 = strategy_propensity_score(df, high_thresh=0.3, low_thresh=0.1, caliper_sd=0.5)
    print(f"  {diag3.get('n_matched', 0)} pairs")

    # Try relaxed caliper
    if diag3.get("n_matched", 0) < 5:
        pairs3b, diag3b = strategy_propensity_score(df, high_thresh=0.3, low_thresh=0.1, caliper_sd=1.0)
        print(f"  Relaxed caliper: {diag3b.get('n_matched', 0)} pairs")
        if diag3b.get("n_matched", 0) > diag3.get("n_matched", 0):
            pairs3, diag3 = pairs3b, diag3b

    # Try even more relaxed thresholds
    if diag3.get("n_matched", 0) < 5:
        pairs3c, diag3c = strategy_propensity_score(df, high_thresh=0.25, low_thresh=0.15, caliper_sd=1.0)
        print(f"  Relaxed thresholds: {diag3c.get('n_matched', 0)} pairs (>0.25/<0.15)")
        if diag3c.get("n_matched", 0) > diag3.get("n_matched", 0):
            pairs3, diag3 = pairs3c, diag3c

    mq3 = matching_quality(pairs3, "propensity_score") if pairs3 else {"n": 0}
    w3 = {m: wilcoxon_test(pairs3, m) for m in quality_metrics} if len(pairs3) >= 2 else {}
    r3 = {m: rosenbaum_sensitivity(pairs3, m) for m in quality_metrics} if len(pairs3) >= 2 else {}

    all_strategy_results["propensity_score"] = {
        "diagnostics": diag3, "matching_quality": mq3,
        "pairs": pairs3, "wilcoxon": w3, "rosenbaum": r3,
    }

    for m in quality_metrics:
        if "p_value" in w3.get(m, {}):
            gamma = r3[m].get("critical_gamma", "N/A")
            print(f"  {m}: diff={w3[m]['mean_diff']:.4f}, p={w3[m]['p_value']:.4f}, Gamma={gamma}")

    # ---- Strategy 4: Mahalanobis matching ----
    write_progress(TASK_ID, RESULTS_DIR, 4, 6,
                   metric={"status": "strategy_4_mahalanobis"})
    print(f"\n[{TASK_ID}] Strategy 4: Mahalanobis distance matching")

    pairs4, diag4 = strategy_mahalanobis(df, high_thresh=0.3, low_thresh=0.1, caliper_factor=3.0)
    print(f"  {diag4.get('n_matched', 0)} pairs")

    # Relaxed thresholds
    if diag4.get("n_matched", 0) < 5:
        pairs4b, diag4b = strategy_mahalanobis(df, high_thresh=0.25, low_thresh=0.15, caliper_factor=3.0)
        print(f"  Relaxed: {diag4b.get('n_matched', 0)} pairs (>0.25/<0.15)")
        if diag4b.get("n_matched", 0) > diag4.get("n_matched", 0):
            pairs4, diag4 = pairs4b, diag4b

    mq4 = matching_quality(pairs4, "mahalanobis") if pairs4 else {"n": 0}
    w4 = {m: wilcoxon_test(pairs4, m) for m in quality_metrics} if len(pairs4) >= 2 else {}
    r4 = {m: rosenbaum_sensitivity(pairs4, m) for m in quality_metrics} if len(pairs4) >= 2 else {}

    all_strategy_results["mahalanobis"] = {
        "diagnostics": diag4, "matching_quality": mq4,
        "pairs": pairs4, "wilcoxon": w4, "rosenbaum": r4,
    }

    for m in quality_metrics:
        if "p_value" in w4.get(m, {}):
            gamma = r4[m].get("critical_gamma", "N/A")
            print(f"  {m}: diff={w4[m]['mean_diff']:.4f}, p={w4[m]['p_value']:.4f}, Gamma={gamma}")

    # ---- Strategy 5: Tertile within width ----
    write_progress(TASK_ID, RESULTS_DIR, 5, 6,
                   metric={"status": "strategy_5_tertile"})
    print(f"\n[{TASK_ID}] Strategy 5: Tertile split within width")

    pairs5, diag5 = strategy_tertile_within_width(df)
    print(f"  {diag5['n_matched']} pairs")

    mq5 = matching_quality(pairs5, "tertile_within_width")
    w5 = {m: wilcoxon_test(pairs5, m) for m in quality_metrics} if len(pairs5) >= 2 else {}
    r5 = {m: rosenbaum_sensitivity(pairs5, m) for m in quality_metrics} if len(pairs5) >= 2 else {}

    all_strategy_results["tertile_within_width"] = {
        "diagnostics": diag5, "matching_quality": mq5,
        "pairs": pairs5, "wilcoxon": w5, "rosenbaum": r5,
    }

    for m in quality_metrics:
        if "p_value" in w5.get(m, {}):
            gamma = r5[m].get("critical_gamma", "N/A")
            print(f"  {m}: diff={w5[m]['mean_diff']:.4f}, p={w5[m]['p_value']:.4f}, Gamma={gamma}")

    # ---- Compile summary ----
    write_progress(TASK_ID, RESULTS_DIR, 6, 6,
                   metric={"status": "compiling_results"})

    # Find best strategy by number of pairs and significance
    strategy_summary = {}
    best_gamma_overall = 0.0
    best_gamma_metric = None
    best_gamma_strategy = None
    max_pairs = 0

    for sname, sres in all_strategy_results.items():
        n_pairs = len(sres.get("pairs", []))
        max_pairs = max(max_pairs, n_pairs)

        gammas = {}
        for m, r in sres.get("rosenbaum", {}).items():
            if isinstance(r, dict) and "critical_gamma" in r:
                gammas[m] = r["critical_gamma"]
                if r["critical_gamma"] > best_gamma_overall:
                    best_gamma_overall = r["critical_gamma"]
                    best_gamma_metric = m
                    best_gamma_strategy = sname

        sig_metrics = [m for m, w in sres.get("wilcoxon", {}).items()
                       if isinstance(w, dict) and w.get("significant_005", False)]

        strategy_summary[sname] = {
            "n_pairs": n_pairs,
            "rosenbaum_gammas": gammas,
            "max_gamma": max(gammas.values()) if gammas else 0.0,
            "significant_metrics": sig_metrics,
        }

    # Pilot pass assessment
    pilot_pass = max_pairs >= 5 and best_gamma_overall > 1.0

    summary = {
        "strategy_comparison": strategy_summary,
        "best_gamma_overall": float(best_gamma_overall),
        "best_gamma_metric": best_gamma_metric,
        "best_gamma_strategy": best_gamma_strategy,
        "max_pairs_any_strategy": max_pairs,
        "pilot_pass": pilot_pass,
        "pilot_pass_reason": (
            f"{'PASS' if pilot_pass else 'FAIL'}: "
            f"max pairs={max_pairs} ({'>=5' if max_pairs >= 5 else '<5'}), "
            f"best Gamma={best_gamma_overall:.2f} ({'>1.0' if best_gamma_overall > 1.0 else '<=1.0'})"
        ),
        "key_finding": _key_finding(all_strategy_results, quality_metrics),
    }

    # ---- Save results ----
    end_time = time.time()
    elapsed = end_time - start_time

    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "timestamp": datetime.now().isoformat(),
        "data_source": str(DATA_PATH),
        "n_saes_total": len(df),
        "seed": SEED,
        "strategies": all_strategy_results,
        "summary": summary,
        "pilot_pass_criteria": "At least 5 matched pairs formed. Gamma > 1.0.",
        "pilot_pass": pilot_pass,
        "runtime_seconds": float(elapsed),
    }

    output_path = RESULTS_DIR / f"{TASK_ID}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"[{TASK_ID}] RESULTS SUMMARY")
    print(f"{'='*60}")
    for sname, ssum in strategy_summary.items():
        print(f"  {sname}: {ssum['n_pairs']} pairs, max Gamma={ssum['max_gamma']:.2f}, "
              f"sig metrics: {ssum['significant_metrics']}")
    print(f"\n  Best overall: strategy={best_gamma_strategy}, metric={best_gamma_metric}, "
          f"Gamma={best_gamma_overall:.2f}")
    print(f"  Pilot pass: {pilot_pass}")
    print(f"  Runtime: {elapsed:.1f}s")
    print(f"  Results saved to: {output_path}")

    # Write DONE marker
    mark_task_done(TASK_ID, RESULTS_DIR,
                   status="success" if pilot_pass else "completed_below_threshold",
                   summary=summary["pilot_pass_reason"])

    return results


def _key_finding(strategies, metrics):
    """Generate a human-readable key finding from results."""
    findings = []

    # Check if within-width strategies show significance
    for sname in ["within_width_median_split", "tertile_within_width"]:
        sres = strategies.get(sname, {})
        for m, w in sres.get("wilcoxon", {}).items():
            if isinstance(w, dict) and w.get("significant_005", False):
                r = sres.get("rosenbaum", {}).get(m, {})
                gamma = r.get("critical_gamma", 0)
                findings.append(
                    f"Within-width {sname}: {m} shows significant quality difference "
                    f"(p={w['p_value']:.4f}, mean diff={w['mean_diff']:.4f}), "
                    f"Rosenbaum Gamma={gamma:.2f}."
                )

    if not findings:
        findings.append(
            "No within-width matching strategy produced significant Wilcoxon results. "
            "This may indicate that absorption's association with quality is primarily "
            "driven by the width/L0 confound, or that n is too small for matched-pair analysis."
        )

    return " | ".join(findings)


if __name__ == "__main__":
    results = main()
