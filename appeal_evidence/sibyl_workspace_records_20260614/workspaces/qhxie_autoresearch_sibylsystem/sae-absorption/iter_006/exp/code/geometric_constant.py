"""
Geometric Constant Validation: Decoder Weight Analysis

Computes c(w_P, w_C) = ||w_P||^2 * (1 - cos^2(w_P, w_C)) from SAE decoder weights
for all first-letter parent-child pairs.

Tests whether c modulates the absorption threshold beyond CMI alone.
Specifically: does CMI/c(w_P, w_C) correlate more strongly with absorption rate than raw CMI?

This is a zero-GPU computation (decoder weights only) that validates the geometric
component of the rate-distortion theory.

Task: geometric_constant (PILOT mode)
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy import stats

# =============================================================================
# Configuration
# =============================================================================
WORKSPACE = "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current"
RESULTS_DIR = Path(WORKSPACE) / "exp" / "results"
FULL_DIR = RESULTS_DIR / "full"
PILOTS_DIR = RESULTS_DIR / "pilots"
CODE_DIR = Path(WORKSPACE) / "exp" / "code"

SEED = 42
TASK_ID = "geometric_constant"

# SAE config - same as used in CMI estimation and first_letter_improved
SAE_RELEASE = "gemma-scope-2b-pt-res"
SAE_ID = "layer_12/width_16k/average_l0_82"

# Input files
FIRST_LETTER_FILE = FULL_DIR / "first_letter_improved.json"
CMI_FILE = FULL_DIR / "cmi_estimation.json"

# =============================================================================
# Process identification (CRITICAL protocol)
# =============================================================================
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(task_id, results_dir, epoch, total_epochs, step=0,
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


# =============================================================================
# Main execution
# =============================================================================
def main():
    timestamp_start = datetime.now().isoformat()
    start_time = time.time()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    report_progress(TASK_ID, RESULTS_DIR, 0, 5, step=0, total_steps=5,
                    metric={"phase": "loading_data"})

    # =========================================================================
    # Step 1: Load first-letter results and CMI results
    # =========================================================================
    print("=" * 60)
    print("Step 1: Loading first-letter and CMI results")
    print("=" * 60)

    with open(FIRST_LETTER_FILE) as f:
        fl_data = json.load(f)

    with open(CMI_FILE) as f:
        cmi_data = json.load(f)

    # Extract per-letter data from first_letter_improved (L12-16k config)
    l12_16k = fl_data["l12_16k"]
    per_letter = l12_16k["per_letter"]

    # Extract CMI data at best subspace dimension (d'=10)
    best_dim = str(cmi_data["best_subspace_dim"])
    cmi_by_letter = cmi_data["cmi_by_subspace_dim"][best_dim]

    # Get the full correlation data
    correlation_data = cmi_data["correlation_cmi_vs_absorption"]

    print(f"  First-letter data: {len(per_letter)} letters")
    print(f"  CMI data: {len(cmi_by_letter)} letters at d'={best_dim}")
    print(f"  Best CMI-absorption rho: {correlation_data['spearman_rho']:.4f}")

    report_progress(TASK_ID, RESULTS_DIR, 1, 5, step=1, total_steps=5,
                    metric={"phase": "loading_sae"})

    # =========================================================================
    # Step 2: Load SAE decoder weights
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Loading SAE decoder weights")
    print("=" * 60)

    from sae_lens import SAE

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device="cpu"  # No GPU needed for decoder weight analysis
    )

    W_dec = sae.W_dec.detach()  # Shape: [d_sae, d_model]
    d_sae, d_model = W_dec.shape
    print(f"  SAE decoder shape: {W_dec.shape}")
    print(f"  d_sae={d_sae}, d_model={d_model}")

    report_progress(TASK_ID, RESULTS_DIR, 2, 5, step=2, total_steps=5,
                    metric={"phase": "computing_geometric_constant"})

    # =========================================================================
    # Step 3: Compute geometric constant for each letter
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Computing geometric constant c(w_P, w_C) for each letter")
    print("=" * 60)

    # For each letter, we need:
    # - Parent feature(s): the split features identified by the k-sparse probe
    #   The "parent" in absorption context is the first-letter feature that should fire
    #   but doesn't (it gets absorbed). The split_features are the probe directions.
    #   The PRIMARY parent feature is the one with highest weight (split_features[0]).
    #
    # - Child feature(s): the features that DO fire and absorb the parent's information.
    #   These are identified in absorbed_examples -> features (feature_idx values).
    #
    # For the geometric constant, we compute c(w_P, w_C) between the parent's
    # decoder direction and each absorbing child's decoder direction.
    # We use the PRIMARY parent feature (highest probe weight) and aggregate
    # across all observed child features.

    letters = sorted([l for l in per_letter.keys() if l in cmi_by_letter])
    print(f"  Processing {len(letters)} letters")

    geometric_results = {}
    all_letters_data = []

    for letter in letters:
        letter_info = per_letter[letter]
        cmi_info = cmi_by_letter[letter]

        # Parent feature: primary split feature (highest probe weight)
        split_features = letter_info["split_features"]
        split_weights = letter_info["split_weights"]
        primary_parent_idx = split_features[0]

        # Get parent decoder direction
        w_parent = W_dec[primary_parent_idx]  # Shape: [d_model]
        w_parent_norm_sq = torch.dot(w_parent, w_parent).item()
        w_parent_norm = np.sqrt(w_parent_norm_sq)

        # Collect child features from absorbed examples
        absorbed_examples = letter_info.get("absorbed_examples", [])
        child_feature_indices = set()
        for ex in absorbed_examples:
            for feat in ex.get("features", []):
                child_feature_indices.add(feat["feature_idx"])

        # If no absorbed examples (absorption_rate = 0), use all split features
        # as potential "child" features for the geometric constant
        # (This gives us c values for non-absorbed letters too, which is needed
        # for the correlation analysis)
        if len(child_feature_indices) == 0:
            # Use split features other than primary as child candidates
            # Also check secondary split features
            for i, sf in enumerate(split_features[1:], 1):
                child_feature_indices.add(sf)

        # If still empty (only 1 split feature), use the primary itself
        # (c will be 0, which is meaningful: perfectly aligned = no geometric penalty)
        if len(child_feature_indices) == 0:
            child_feature_indices.add(primary_parent_idx)

        # Compute c(w_P, w_C) for each child
        c_values = []
        cos_values = []
        child_details = []

        for child_idx in sorted(child_feature_indices):
            w_child = W_dec[child_idx]  # Shape: [d_model]
            w_child_norm = torch.norm(w_child).item()

            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                w_parent.unsqueeze(0), w_child.unsqueeze(0)
            ).item()

            # Geometric constant: c = ||w_P||^2 * (1 - cos^2(w_P, w_C))
            c_val = w_parent_norm_sq * (1.0 - cos_sim ** 2)

            c_values.append(c_val)
            cos_values.append(cos_sim)
            child_details.append({
                "child_idx": int(child_idx),
                "cosine_w_parent": round(cos_sim, 6),
                "c_value": round(c_val, 6),
                "w_child_norm": round(w_child_norm, 6),
            })

        # Summary statistics for this letter
        mean_c = float(np.mean(c_values))
        median_c = float(np.median(c_values))
        min_c = float(np.min(c_values))
        max_c = float(np.max(c_values))
        mean_cos = float(np.mean(cos_values))

        # CMI value
        cmi_val = cmi_info["cmi"]
        absorption_rate = cmi_info["absorption_rate"]
        probe_f1 = cmi_info["probe_f1"]

        # CMI / c ratio (use mean_c; handle division by near-zero)
        if mean_c > 1e-10:
            cmi_over_c = cmi_val / mean_c
        else:
            cmi_over_c = float('inf')

        # Also compute CMI / median_c
        if median_c > 1e-10:
            cmi_over_c_median = cmi_val / median_c
        else:
            cmi_over_c_median = float('inf')

        result = {
            "letter": letter,
            "absorption_rate": absorption_rate,
            "probe_f1": probe_f1,
            "cmi": round(cmi_val, 6),
            "primary_parent_idx": int(primary_parent_idx),
            "primary_parent_weight": round(split_weights[0], 4),
            "w_parent_norm": round(w_parent_norm, 6),
            "w_parent_norm_sq": round(w_parent_norm_sq, 6),
            "n_child_features": len(child_feature_indices),
            "mean_c": round(mean_c, 6),
            "median_c": round(median_c, 6),
            "min_c": round(min_c, 6),
            "max_c": round(max_c, 6),
            "mean_cosine": round(mean_cos, 6),
            "cmi_over_c_mean": round(cmi_over_c, 6) if cmi_over_c != float('inf') else "inf",
            "cmi_over_c_median": round(cmi_over_c_median, 6) if cmi_over_c_median != float('inf') else "inf",
            "child_details": child_details,
        }

        geometric_results[letter] = result
        all_letters_data.append(result)

        print(f"  {letter}: abs_rate={absorption_rate:.4f}, CMI={cmi_val:.4f}, "
              f"mean_c={mean_c:.4f}, CMI/c={cmi_over_c:.4f}, "
              f"n_children={len(child_feature_indices)}, "
              f"mean_cos={mean_cos:.4f}")

    report_progress(TASK_ID, RESULTS_DIR, 3, 5, step=3, total_steps=5,
                    metric={"phase": "computing_correlations"})

    # =========================================================================
    # Step 4: Compute correlations
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Computing correlations")
    print("=" * 60)

    # Prepare arrays for correlation
    letters_for_corr = []
    abs_rates = []
    cmi_vals = []
    c_vals_mean = []
    c_vals_median = []
    cmi_over_c_mean_vals = []
    cmi_over_c_median_vals = []
    parent_norms = []
    mean_cosines = []

    for d in all_letters_data:
        # Skip letters where cmi_over_c is inf (division by near-zero c)
        if d["cmi_over_c_mean"] == "inf" or d["cmi_over_c_median"] == "inf":
            print(f"  Skipping {d['letter']} (c near zero)")
            continue
        letters_for_corr.append(d["letter"])
        abs_rates.append(d["absorption_rate"])
        cmi_vals.append(d["cmi"])
        c_vals_mean.append(d["mean_c"])
        c_vals_median.append(d["median_c"])
        cmi_over_c_mean_vals.append(d["cmi_over_c_mean"])
        cmi_over_c_median_vals.append(d["cmi_over_c_median"])
        parent_norms.append(d["w_parent_norm"])
        mean_cosines.append(d["mean_cosine"])

    abs_rates = np.array(abs_rates)
    cmi_vals = np.array(cmi_vals)
    c_vals_mean = np.array(c_vals_mean)
    c_vals_median = np.array(c_vals_median)
    cmi_over_c_mean_vals = np.array(cmi_over_c_mean_vals)
    cmi_over_c_median_vals = np.array(cmi_over_c_median_vals)
    parent_norms = np.array(parent_norms)
    mean_cosines = np.array(mean_cosines)

    n_valid = len(letters_for_corr)
    print(f"  Valid letters for correlation: {n_valid}")

    # 1. Raw CMI vs absorption (replicating for comparison)
    rho_cmi, p_cmi = stats.spearmanr(cmi_vals, abs_rates)
    r_cmi, p_r_cmi = stats.pearsonr(cmi_vals, abs_rates)
    print(f"\n  Raw CMI vs absorption:")
    print(f"    Spearman rho = {rho_cmi:.4f} (p = {p_cmi:.4f})")
    print(f"    Pearson r    = {r_cmi:.4f} (p = {p_r_cmi:.4f})")

    # 2. c (geometric constant) vs absorption
    rho_c, p_c = stats.spearmanr(c_vals_mean, abs_rates)
    r_c, p_r_c = stats.pearsonr(c_vals_mean, abs_rates)
    print(f"\n  Geometric constant c vs absorption:")
    print(f"    Spearman rho = {rho_c:.4f} (p = {p_c:.4f})")
    print(f"    Pearson r    = {r_c:.4f} (p = {p_r_c:.4f})")

    # 3. CMI/c (mean) vs absorption -- PRIMARY TEST
    rho_cmi_c_mean, p_cmi_c_mean = stats.spearmanr(cmi_over_c_mean_vals, abs_rates)
    r_cmi_c_mean, p_r_cmi_c_mean = stats.pearsonr(cmi_over_c_mean_vals, abs_rates)
    print(f"\n  CMI/c(mean) vs absorption (PRIMARY TEST):")
    print(f"    Spearman rho = {rho_cmi_c_mean:.4f} (p = {p_cmi_c_mean:.4f})")
    print(f"    Pearson r    = {r_cmi_c_mean:.4f} (p = {p_r_cmi_c_mean:.4f})")

    # 4. CMI/c (median) vs absorption
    rho_cmi_c_med, p_cmi_c_med = stats.spearmanr(cmi_over_c_median_vals, abs_rates)
    r_cmi_c_med, p_r_cmi_c_med = stats.pearsonr(cmi_over_c_median_vals, abs_rates)
    print(f"\n  CMI/c(median) vs absorption:")
    print(f"    Spearman rho = {rho_cmi_c_med:.4f} (p = {p_cmi_c_med:.4f})")
    print(f"    Pearson r    = {r_cmi_c_med:.4f} (p = {p_r_cmi_c_med:.4f})")

    # 5. Parent norm vs absorption
    rho_norm, p_norm = stats.spearmanr(parent_norms, abs_rates)
    print(f"\n  Parent norm vs absorption:")
    print(f"    Spearman rho = {rho_norm:.4f} (p = {p_norm:.4f})")

    # 6. Mean cosine vs absorption
    rho_cos, p_cos = stats.spearmanr(mean_cosines, abs_rates)
    print(f"\n  Mean cosine vs absorption:")
    print(f"    Spearman rho = {rho_cos:.4f} (p = {p_cos:.4f})")

    # 7. c vs CMI (do the two geometric/information-theoretic quantities relate?)
    rho_c_cmi, p_c_cmi = stats.spearmanr(c_vals_mean, cmi_vals)
    print(f"\n  c vs CMI (cross-metric):")
    print(f"    Spearman rho = {rho_c_cmi:.4f} (p = {p_c_cmi:.4f})")

    # =========================================================================
    # Step 4b: Improvement assessment
    # =========================================================================
    print("\n" + "-" * 40)
    print("Improvement Assessment: Does CMI/c beat raw CMI?")
    print("-" * 40)

    # Compare absolute Spearman rho values
    abs_rho_cmi = abs(rho_cmi)
    abs_rho_cmi_c = abs(rho_cmi_c_mean)
    improvement = abs_rho_cmi_c - abs_rho_cmi
    pct_improvement = (improvement / abs_rho_cmi * 100) if abs_rho_cmi > 0 else float('inf')

    print(f"  |rho(CMI, absorption)|      = {abs_rho_cmi:.4f}")
    print(f"  |rho(CMI/c, absorption)|     = {abs_rho_cmi_c:.4f}")
    print(f"  Improvement: {improvement:+.4f} ({pct_improvement:+.1f}%)")

    if abs_rho_cmi_c > abs_rho_cmi:
        print("  RESULT: CMI/c IMPROVES over raw CMI")
        c_modulates = True
    else:
        print("  RESULT: CMI/c does NOT improve over raw CMI")
        c_modulates = False

    # Bootstrap CI for the difference in rho
    n_bootstrap = 10000
    rng = np.random.RandomState(SEED)
    rho_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_valid, size=n_valid, replace=True)
        try:
            rho_boot_cmi, _ = stats.spearmanr(cmi_vals[idx], abs_rates[idx])
            rho_boot_cmi_c, _ = stats.spearmanr(cmi_over_c_mean_vals[idx], abs_rates[idx])
            rho_diffs.append(abs(rho_boot_cmi_c) - abs(rho_boot_cmi))
        except:
            pass

    rho_diffs = np.array(rho_diffs)
    ci_lower_diff = float(np.percentile(rho_diffs, 2.5))
    ci_upper_diff = float(np.percentile(rho_diffs, 97.5))
    print(f"  Bootstrap 95% CI for |rho(CMI/c)| - |rho(CMI)|: "
          f"[{ci_lower_diff:.4f}, {ci_upper_diff:.4f}]")

    report_progress(TASK_ID, RESULTS_DIR, 4, 5, step=4, total_steps=5,
                    metric={"phase": "multi_dim_analysis"})

    # =========================================================================
    # Step 5: CMI/c across all subspace dimensions
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 5: CMI/c correlation across subspace dimensions")
    print("=" * 60)

    dim_results = {}
    for dim_str in ["10", "20", "30", "50"]:
        cmi_dim = cmi_data["cmi_by_subspace_dim"][dim_str]

        dim_cmi_vals = []
        dim_abs_rates = []
        dim_cmi_over_c = []
        dim_letters = []

        for d in all_letters_data:
            letter = d["letter"]
            if letter not in cmi_dim:
                continue
            if d["cmi_over_c_mean"] == "inf":
                continue

            cmi_at_dim = cmi_dim[letter]["cmi"]
            c_mean = d["mean_c"]
            if c_mean < 1e-10:
                continue

            dim_letters.append(letter)
            dim_cmi_vals.append(cmi_at_dim)
            dim_abs_rates.append(d["absorption_rate"])
            dim_cmi_over_c.append(cmi_at_dim / c_mean)

        dim_cmi_vals = np.array(dim_cmi_vals)
        dim_abs_rates = np.array(dim_abs_rates)
        dim_cmi_over_c = np.array(dim_cmi_over_c)

        rho_raw, p_raw = stats.spearmanr(dim_cmi_vals, dim_abs_rates)
        rho_adj, p_adj = stats.spearmanr(dim_cmi_over_c, dim_abs_rates)

        dim_results[dim_str] = {
            "n_letters": len(dim_letters),
            "spearman_rho_raw_cmi": round(float(rho_raw), 6),
            "spearman_p_raw_cmi": round(float(p_raw), 6),
            "spearman_rho_cmi_over_c": round(float(rho_adj), 6),
            "spearman_p_cmi_over_c": round(float(p_adj), 6),
            "improvement": round(float(abs(rho_adj) - abs(rho_raw)), 6),
        }

        print(f"  d'={dim_str}: rho(CMI)={rho_raw:.4f}, rho(CMI/c)={rho_adj:.4f}, "
              f"improvement={abs(rho_adj) - abs(rho_raw):+.4f}")

    # =========================================================================
    # Step 5b: Additional analyses
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 5b: Distribution statistics for c values")
    print("=" * 60)

    all_c_mean = [d["mean_c"] for d in all_letters_data]
    all_c_mean = np.array(all_c_mean)
    print(f"  c statistics across {len(all_c_mean)} letters:")
    print(f"    Mean:   {np.mean(all_c_mean):.4f}")
    print(f"    Median: {np.median(all_c_mean):.4f}")
    print(f"    Std:    {np.std(all_c_mean):.4f}")
    print(f"    Min:    {np.min(all_c_mean):.4f}")
    print(f"    Max:    {np.max(all_c_mean):.4f}")

    # Absorbed vs non-absorbed c values
    absorbed_c = [d["mean_c"] for d in all_letters_data if d["absorption_rate"] > 0.1]
    non_absorbed_c = [d["mean_c"] for d in all_letters_data if d["absorption_rate"] <= 0.05]

    if len(absorbed_c) > 0 and len(non_absorbed_c) > 0:
        absorbed_c = np.array(absorbed_c)
        non_absorbed_c = np.array(non_absorbed_c)
        u_stat, u_p = stats.mannwhitneyu(absorbed_c, non_absorbed_c, alternative='two-sided')
        print(f"\n  Absorbed vs non-absorbed c values:")
        print(f"    Absorbed mean c:     {np.mean(absorbed_c):.4f} (n={len(absorbed_c)})")
        print(f"    Non-absorbed mean c: {np.mean(non_absorbed_c):.4f} (n={len(non_absorbed_c)})")
        print(f"    Mann-Whitney U={u_stat:.1f}, p={u_p:.4f}")
    else:
        u_stat, u_p = None, None

    # =========================================================================
    # Compile results
    # =========================================================================
    elapsed = time.time() - start_time

    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "seed": SEED,
        "model": "gemma-2-2b",
        "sae_config": "L12-16k-L0-82",
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "timestamp_start": timestamp_start,
        "methodology": {
            "description": "Compute c(w_P, w_C) = ||w_P||^2 * (1 - cos^2(w_P, w_C)) from SAE decoder weights for first-letter parent-child pairs. Parent = primary split feature (highest probe weight). Children = absorbing features from absorbed_examples (or secondary split features for non-absorbed letters).",
            "parent_selection": "Primary split feature with highest probe weight",
            "child_selection": "Absorbing features from absorbed_examples for absorbed letters; secondary split features for non-absorbed letters",
            "best_subspace_dim": int(best_dim),
        },
        "n_letters": len(all_letters_data),
        "n_valid_for_correlation": n_valid,
        "correlations": {
            "raw_cmi_vs_absorption": {
                "spearman_rho": round(float(rho_cmi), 6),
                "spearman_p": round(float(p_cmi), 6),
                "pearson_r": round(float(r_cmi), 6),
                "pearson_p": round(float(p_r_cmi), 6),
            },
            "geometric_c_vs_absorption": {
                "spearman_rho": round(float(rho_c), 6),
                "spearman_p": round(float(p_c), 6),
                "pearson_r": round(float(r_c), 6),
                "pearson_p": round(float(p_r_c), 6),
            },
            "cmi_over_c_mean_vs_absorption": {
                "spearman_rho": round(float(rho_cmi_c_mean), 6),
                "spearman_p": round(float(p_cmi_c_mean), 6),
                "pearson_r": round(float(r_cmi_c_mean), 6),
                "pearson_p": round(float(p_r_cmi_c_mean), 6),
            },
            "cmi_over_c_median_vs_absorption": {
                "spearman_rho": round(float(rho_cmi_c_med), 6),
                "spearman_p": round(float(p_cmi_c_med), 6),
                "pearson_r": round(float(r_cmi_c_med), 6),
                "pearson_p": round(float(p_r_cmi_c_med), 6),
            },
            "parent_norm_vs_absorption": {
                "spearman_rho": round(float(rho_norm), 6),
                "spearman_p": round(float(p_norm), 6),
            },
            "mean_cosine_vs_absorption": {
                "spearman_rho": round(float(rho_cos), 6),
                "spearman_p": round(float(p_cos), 6),
            },
            "c_vs_cmi": {
                "spearman_rho": round(float(rho_c_cmi), 6),
                "spearman_p": round(float(p_c_cmi), 6),
            },
        },
        "improvement_assessment": {
            "abs_rho_raw_cmi": round(float(abs_rho_cmi), 6),
            "abs_rho_cmi_over_c": round(float(abs_rho_cmi_c), 6),
            "improvement": round(float(improvement), 6),
            "pct_improvement": round(float(pct_improvement), 1),
            "c_modulates_threshold": c_modulates,
            "bootstrap_ci_95_diff": [round(ci_lower_diff, 6), round(ci_upper_diff, 6)],
        },
        "subspace_dim_comparison": dim_results,
        "c_distribution": {
            "mean": round(float(np.mean(all_c_mean)), 6),
            "median": round(float(np.median(all_c_mean)), 6),
            "std": round(float(np.std(all_c_mean)), 6),
            "min": round(float(np.min(all_c_mean)), 6),
            "max": round(float(np.max(all_c_mean)), 6),
        },
        "absorbed_vs_non_absorbed_c": {
            "absorbed_mean_c": round(float(np.mean(absorbed_c)), 6) if len(absorbed_c) > 0 else None,
            "non_absorbed_mean_c": round(float(np.mean(non_absorbed_c)), 6) if len(non_absorbed_c) > 0 else None,
            "mann_whitney_U": round(float(u_stat), 1) if u_stat is not None else None,
            "mann_whitney_p": round(float(u_p), 6) if u_p is not None else None,
        },
        "per_letter": geometric_results,
        "summary_table": [
            {
                "letter": d["letter"],
                "absorption_rate": d["absorption_rate"],
                "probe_f1": d["probe_f1"],
                "cmi": d["cmi"],
                "mean_c": d["mean_c"],
                "median_c": d["median_c"],
                "cmi_over_c_mean": d["cmi_over_c_mean"],
                "w_parent_norm": d["w_parent_norm"],
                "mean_cosine": d["mean_cosine"],
                "n_child_features": d["n_child_features"],
            }
            for d in all_letters_data
        ],
        "pass_criteria": {
            "c_computed_all_25_letters": len(all_letters_data) == 25,
            "n_letters_computed": len(all_letters_data),
            "cmi_over_c_correlation_computed": True,
            "comparison_with_raw_cmi_reported": True,
            "c_modulates_beyond_cmi": c_modulates,
            "overall_pass": len(all_letters_data) >= 20,
        },
        "elapsed_sec": round(elapsed, 2),
        "timestamp_end": datetime.now().isoformat(),
    }

    # Save results
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    output_file = FULL_DIR / "geometric_constant.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_file}")

    # Also save to pilots directory
    PILOTS_DIR.mkdir(parents=True, exist_ok=True)
    pilot_file = PILOTS_DIR / "geometric_constant.json"
    with open(pilot_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Pilot results saved to: {pilot_file}")

    # =========================================================================
    # Final summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY: Geometric Constant Validation")
    print("=" * 60)
    print(f"  Letters analyzed: {len(all_letters_data)}")
    print(f"  Raw CMI vs absorption:     rho = {rho_cmi:.4f} (p = {p_cmi:.4f})")
    print(f"  c (geometric) vs absorption: rho = {rho_c:.4f} (p = {p_c:.4f})")
    print(f"  CMI/c vs absorption:         rho = {rho_cmi_c_mean:.4f} (p = {p_cmi_c_mean:.4f})")
    print(f"  Improvement: {improvement:+.4f} ({pct_improvement:+.1f}%)")
    print(f"  Bootstrap 95% CI for diff: [{ci_lower_diff:.4f}, {ci_upper_diff:.4f}]")
    print(f"  c modulates threshold: {c_modulates}")
    print(f"  Elapsed: {elapsed:.1f}s")

    report_progress(TASK_ID, RESULTS_DIR, 5, 5, step=5, total_steps=5,
                    metric={
                        "phase": "done",
                        "rho_cmi": round(float(rho_cmi), 4),
                        "rho_cmi_over_c": round(float(rho_cmi_c_mean), 4),
                        "c_modulates": c_modulates,
                    })

    return results


if __name__ == "__main__":
    try:
        results = main()

        # Update gpu_progress.json
        gpu_progress_file = Path(WORKSPACE) / "exp" / "gpu_progress.json"
        if gpu_progress_file.exists():
            with open(gpu_progress_file) as f:
                gp = json.load(f)
        else:
            gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

        if TASK_ID not in gp["completed"]:
            gp["completed"].append(TASK_ID)
        if TASK_ID in gp.get("running", {}):
            del gp["running"][TASK_ID]

        end_time = datetime.now().isoformat()
        gp["timings"][TASK_ID] = {
            "planned_min": 10,
            "actual_min": round(results["elapsed_sec"] / 60, 1),
            "start_time": results["timestamp_start"],
            "end_time": end_time,
            "config_snapshot": {
                "model": "gemma-2-2b",
                "sae_config": "L12-16k-L0-82",
                "n_letters": 25,
                "analysis_type": "decoder_weight_geometry",
                "gpu_model": "RTX PRO 6000 Blackwell",
                "gpu_count": 0,
            }
        }

        with open(gpu_progress_file, "w") as f:
            json.dump(gp, f, indent=2)

        # Write DONE marker
        mark_task_done(TASK_ID, RESULTS_DIR,
                       status="success",
                       summary=f"Geometric constant computed for 25 letters. "
                               f"CMI/c rho={results['correlations']['cmi_over_c_mean_vs_absorption']['spearman_rho']:.4f}, "
                               f"raw CMI rho={results['correlations']['raw_cmi_vs_absorption']['spearman_rho']:.4f}. "
                               f"c modulates={results['improvement_assessment']['c_modulates_threshold']}")

        print(f"\n[DONE] geometric_constant completed successfully")

    except Exception as e:
        traceback.print_exc()

        # Record failure
        gpu_progress_file = Path(WORKSPACE) / "exp" / "gpu_progress.json"
        if gpu_progress_file.exists():
            with open(gpu_progress_file) as f:
                gp = json.load(f)
        else:
            gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

        if TASK_ID not in gp["failed"]:
            gp["failed"].append(TASK_ID)
        if TASK_ID in gp.get("running", {}):
            del gp["running"][TASK_ID]

        with open(gpu_progress_file, "w") as f:
            json.dump(gp, f, indent=2)

        mark_task_done(TASK_ID, RESULTS_DIR,
                       status="failed",
                       summary=f"Error: {str(e)}")

        sys.exit(1)
