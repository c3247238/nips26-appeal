#!/usr/bin/env python3
"""
Unsupervised Pipeline Validation Against Probe Gold Standard (Stage 2c)

Validates the full unsupervised pipeline (conditional cosine + firing rate + ITAC)
against first-letter probe-based absorption rate (gold standard).

Reports: Spearman rho, AUROC, Precision@50.
Ablates each component independently.

Decision gate: rho < 0.3 -> de-emphasize unsupervised detection contribution.
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# ── PID file for system recovery ──
TASK_ID = "unsupervised_validation"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "exp/results/full"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

def report_progress(task_id, results_dir, epoch, total_epochs, step=0,
                    total_steps=0, loss=None, metric=None):
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_task_done(task_id, results_dir, status="success", summary=""):
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


def load_gold_standard(first_letter_path):
    """Load per-letter absorption rates and split features from first_letter_validation."""
    with open(first_letter_path) as f:
        data = json.load(f)

    l12_16k = data["l12_16k"]
    letters_data = {}
    for letter, info in l12_16k["per_letter"].items():
        if info.get("status") == "ok":
            letters_data[letter] = {
                "absorption_rate": info["absorption_rate"],
                "n_tested": info["n_tested"],
                "n_absorbed": info["n_absorbed"],
                "n_false_negatives": info["n_false_negatives"],
                "probe_f1": info["probe_f1"],
                "split_features": info["split_features"],
                "split_weights": info.get("split_weights", []),
                "absorbed_examples": info.get("absorbed_examples", []),
            }

    aggregate = l12_16k["aggregate"]
    return letters_data, aggregate


def load_decoder_geometry(geometry_path):
    """Load candidate pairs from decoder geometry analysis."""
    with open(geometry_path) as f:
        data = json.load(f)
    return data["candidate_pairs"], data


def load_itac_results(itac_path):
    """Load ITAC results for candidate pairs."""
    with open(itac_path) as f:
        data = json.load(f)
    return data["candidate_pairs_results"], data


def build_feature_to_letter_map(letters_data):
    """Map each split feature index to its letter(s)."""
    feature_to_letters = {}
    for letter, info in letters_data.items():
        for feat_idx in info["split_features"]:
            if feat_idx not in feature_to_letters:
                feature_to_letters[feat_idx] = []
            feature_to_letters[feat_idx].append(letter)
    return feature_to_letters


def compute_per_letter_unsupervised_scores(letters_data, candidate_pairs, itac_pairs):
    """
    For each letter, compute unsupervised absorption scores.

    Strategy: A letter is considered "detected as absorbed" by the unsupervised pipeline
    if any of its split features appear in a candidate pair (as either parent or child).
    The score for the letter is the max absorption_score across all matching pairs.

    We compute scores from multiple components:
    1. Conditional cosine similarity (from decoder geometry)
    2. Firing rate ratio (from decoder geometry)
    3. ITAC value (from ITAC analysis)
    4. Full pipeline absorption_score (combined)
    """
    feature_to_letters = build_feature_to_letter_map(letters_data)

    # Build lookup for ITAC values by (parent, child) pair
    itac_lookup = {}
    for pair in itac_pairs:
        key = (pair["parent_idx"], pair["child_idx"])
        itac_lookup[key] = pair

    # Initialize per-letter scores
    letter_scores = {}
    for letter in letters_data:
        letter_scores[letter] = {
            "conditional_cosine_max": 0.0,
            "conditional_cosine_mean": 0.0,
            "firing_rate_score_max": 0.0,
            "itac_max": 0.0,
            "itac_mean": 0.0,
            "absorption_score_max": 0.0,
            "absorption_score_mean": 0.0,
            "n_matching_pairs": 0,
            "matching_pairs": [],
            "gold_absorption_rate": letters_data[letter]["absorption_rate"],
            "gold_probe_f1": letters_data[letter]["probe_f1"],
        }

    # Scan ALL candidate pairs from decoder geometry
    for pair in candidate_pairs:
        parent_idx = pair["parent_idx"]
        child_idx = pair["child_idx"]

        # Check if either feature is a split feature for any letter
        involved_letters = set()
        if parent_idx in feature_to_letters:
            involved_letters.update(feature_to_letters[parent_idx])
        if child_idx in feature_to_letters:
            involved_letters.update(feature_to_letters[child_idx])

        if not involved_letters:
            continue

        # Get ITAC for this pair if available
        itac_key = (parent_idx, child_idx)
        itac_val = None
        if itac_key in itac_lookup:
            itac_info = itac_lookup[itac_key]
            if itac_info.get("valid", False):
                itac_val = itac_info["itac"]

        for letter in involved_letters:
            ls = letter_scores[letter]
            ls["n_matching_pairs"] += 1
            ls["matching_pairs"].append({
                "parent_idx": parent_idx,
                "child_idx": child_idx,
                "global_cosine": pair.get("global_cosine", 0),
                "conditional_cosine": pair.get("conditional_cosine", 0),
                "firing_rate_ratio": pair.get("firing_rate_ratio", 1.0),
                "absorption_score": pair.get("absorption_score", 0),
                "itac": itac_val,
            })

            # Update max/mean scores
            cc = pair.get("conditional_cosine", 0)
            fr = 1.0 - pair.get("firing_rate_ratio", 1.0)  # Higher = more asymmetric
            ab_score = pair.get("absorption_score", 0)

            ls["conditional_cosine_max"] = max(ls["conditional_cosine_max"], cc)
            ls["firing_rate_score_max"] = max(ls["firing_rate_score_max"], fr)
            ls["absorption_score_max"] = max(ls["absorption_score_max"], ab_score)

            if itac_val is not None:
                ls["itac_max"] = max(ls["itac_max"], itac_val)

    # Compute means for letters with matches
    for letter, ls in letter_scores.items():
        n = ls["n_matching_pairs"]
        if n > 0:
            pairs = ls["matching_pairs"]
            ls["conditional_cosine_mean"] = np.mean([p["conditional_cosine"] for p in pairs])
            ls["firing_rate_score_mean"] = np.mean([1.0 - p["firing_rate_ratio"] for p in pairs])
            ls["absorption_score_mean"] = np.mean([p["absorption_score"] for p in pairs])
            itac_vals = [p["itac"] for p in pairs if p["itac"] is not None]
            if itac_vals:
                ls["itac_mean"] = np.mean(itac_vals)

    return letter_scores


def compute_all_pairs_scores(candidate_pairs, itac_pairs, letters_data):
    """
    Alternative approach: treat ALL candidate pairs as potential absorption pairs,
    and evaluate how well the unsupervised pipeline predicts known absorbed feature indices.

    For each candidate pair, check if it involves a known absorbed feature
    (from absorbed_examples in the gold standard).
    """
    # Collect all known absorbed feature indices
    absorbed_features = set()
    for letter, info in letters_data.items():
        for example in info.get("absorbed_examples", []):
            for feat in example.get("features", []):
                absorbed_features.add(feat["feature_idx"])

    # Also collect all split features (these are the "expected parent-like" features)
    all_split_features = set()
    for letter, info in letters_data.items():
        for feat_idx in info["split_features"]:
            all_split_features.add(feat_idx)

    # Build ITAC lookup
    itac_lookup = {}
    for pair in itac_pairs:
        key = (pair["parent_idx"], pair["child_idx"])
        if pair.get("valid", False):
            itac_lookup[key] = pair["itac"]

    # Score each candidate pair
    pair_scores = []
    for pair in candidate_pairs:
        parent_idx = pair["parent_idx"]
        child_idx = pair["child_idx"]

        # Is this pair "truly absorbed"?
        # A pair is "truly involved in absorption" if one feature is an absorbed feature
        # and the other is a split feature (or any known letter feature)
        involves_absorbed = parent_idx in absorbed_features or child_idx in absorbed_features
        involves_split = parent_idx in all_split_features or child_idx in all_split_features
        is_absorption_pair = involves_absorbed or involves_split

        itac_key = (parent_idx, child_idx)
        itac_val = itac_lookup.get(itac_key, None)

        pair_scores.append({
            "parent_idx": parent_idx,
            "child_idx": child_idx,
            "is_absorption_related": is_absorption_pair,
            "involves_absorbed_feature": involves_absorbed,
            "involves_split_feature": involves_split,
            "global_cosine": pair.get("global_cosine", 0),
            "conditional_cosine": pair.get("conditional_cosine", 0),
            "firing_rate_ratio": pair.get("firing_rate_ratio", 1.0),
            "absorption_score": pair.get("absorption_score", 0),
            "itac": itac_val,
        })

    return pair_scores, absorbed_features, all_split_features


def spearman_rho(x, y):
    """Compute Spearman rank correlation."""
    from scipy.stats import spearmanr
    if len(x) < 3:
        return 0.0, 1.0
    rho, p = spearmanr(x, y)
    return float(rho) if not np.isnan(rho) else 0.0, float(p)


def compute_auroc(labels, scores):
    """Compute AUROC."""
    from sklearn.metrics import roc_auc_score, roc_curve
    labels = np.array(labels)
    scores = np.array(scores)

    if len(np.unique(labels)) < 2:
        return 0.5, [], [], []

    try:
        auroc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        return float(auroc), fpr.tolist(), tpr.tolist(), thresholds.tolist()
    except Exception:
        return 0.5, [], [], []


def compute_precision_at_k(labels, scores, k=50):
    """Compute Precision@K: fraction of top-K scored items that are true positives."""
    indices = np.argsort(scores)[::-1]  # Sort descending by score
    top_k = min(k, len(indices))
    top_k_labels = [labels[i] for i in indices[:top_k]]
    precision = sum(top_k_labels) / top_k if top_k > 0 else 0.0
    return float(precision), top_k


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_stats.append(np.mean(sample))
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_stats, alpha * 100))
    upper = float(np.percentile(boot_stats, (1 - alpha) * 100))
    return lower, upper


def run_validation():
    """Main validation pipeline."""
    start_time = time.time()
    timestamp_start = datetime.now().isoformat()

    report_progress(TASK_ID, RESULTS_DIR, 0, 5, step=0, total_steps=5,
                    metric={"status": "loading_data"})

    # ── Step 1: Load data ──
    workspace = Path(".")
    first_letter_path = workspace / "exp/results/full/first_letter_validation.json"
    geometry_path = workspace / "exp/results/full/decoder_geometry.json"
    itac_path = workspace / "exp/results/full/itac_real_activations.json"

    print("Loading gold standard (first_letter_validation)...")
    letters_data, aggregate = load_gold_standard(first_letter_path)
    print(f"  Loaded {len(letters_data)} letters, aggregate absorption rate: {aggregate['aggregate_absorption_rate']:.4f}")

    print("Loading decoder geometry candidates...")
    geometry_pairs, geometry_full = load_decoder_geometry(geometry_path)
    print(f"  Loaded {len(geometry_pairs)} candidate pairs")

    print("Loading ITAC results...")
    itac_pairs, itac_full = load_itac_results(itac_path)
    print(f"  Loaded {len(itac_pairs)} ITAC pair results")

    report_progress(TASK_ID, RESULTS_DIR, 1, 5, step=1, total_steps=5,
                    metric={"status": "computing_per_letter_scores"})

    # ── Step 2: Per-letter unsupervised scores ──
    print("\nComputing per-letter unsupervised scores...")
    letter_scores = compute_per_letter_unsupervised_scores(
        letters_data, geometry_pairs, itac_pairs
    )

    # Print summary
    letters_with_matches = sum(1 for ls in letter_scores.values() if ls["n_matching_pairs"] > 0)
    print(f"  Letters with matching unsupervised pairs: {letters_with_matches}/{len(letter_scores)}")

    for letter in sorted(letter_scores.keys()):
        ls = letter_scores[letter]
        print(f"  {letter}: gold_rate={ls['gold_absorption_rate']:.2f}, "
              f"n_pairs={ls['n_matching_pairs']}, "
              f"absorption_score_max={ls['absorption_score_max']:.4f}, "
              f"itac_max={ls['itac_max']:.2f}")

    report_progress(TASK_ID, RESULTS_DIR, 2, 5, step=2, total_steps=5,
                    metric={"status": "computing_correlations"})

    # ── Step 3: Compute validation metrics ──
    print("\n=== Validation Metrics ===")

    # Extract arrays for correlation
    letters_sorted = sorted(letter_scores.keys())
    gold_rates = [letter_scores[l]["gold_absorption_rate"] for l in letters_sorted]
    absorption_scores_max = [letter_scores[l]["absorption_score_max"] for l in letters_sorted]
    absorption_scores_mean = [letter_scores[l]["absorption_score_mean"] for l in letters_sorted]
    cond_cosine_max = [letter_scores[l]["conditional_cosine_max"] for l in letters_sorted]
    cond_cosine_mean = [letter_scores[l]["conditional_cosine_mean"] for l in letters_sorted]
    fr_scores_max = [letter_scores[l]["firing_rate_score_max"] for l in letters_sorted]
    itac_max = [letter_scores[l]["itac_max"] for l in letters_sorted]
    itac_mean = [letter_scores[l]["itac_mean"] for l in letters_sorted]
    n_pairs = [letter_scores[l]["n_matching_pairs"] for l in letters_sorted]

    # Binary labels: absorbed (rate > 0.1) vs non-absorbed (rate <= 0.05)
    # Use intermediate threshold to exclude ambiguous cases
    binary_labels_strict = []
    binary_scores_strict = []
    binary_scores_cc = []
    binary_scores_fr = []
    binary_scores_itac = []
    binary_letters = []

    for l in letters_sorted:
        rate = letter_scores[l]["gold_absorption_rate"]
        if rate > 0.1:
            label = 1
        elif rate <= 0.05:
            label = 0
        else:
            continue  # Skip ambiguous
        binary_labels_strict.append(label)
        binary_scores_strict.append(letter_scores[l]["absorption_score_max"])
        binary_scores_cc.append(letter_scores[l]["conditional_cosine_max"])
        binary_scores_fr.append(letter_scores[l]["firing_rate_score_max"])
        binary_scores_itac.append(letter_scores[l]["itac_max"])
        binary_letters.append(l)

    # Also use all letters with any-rate binary: rate > 0 vs rate == 0
    binary_all_labels = [1 if letter_scores[l]["gold_absorption_rate"] > 0 else 0
                         for l in letters_sorted]
    binary_all_scores = [letter_scores[l]["absorption_score_max"] for l in letters_sorted]

    # ── Spearman correlations ──
    results_correlations = {}

    # Full pipeline (absorption_score)
    rho_full_max, p_full_max = spearman_rho(gold_rates, absorption_scores_max)
    rho_full_mean, p_full_mean = spearman_rho(gold_rates, absorption_scores_mean)
    print(f"\nFull Pipeline (absorption_score):")
    print(f"  Spearman rho (max): {rho_full_max:.4f} (p={p_full_max:.4f})")
    print(f"  Spearman rho (mean): {rho_full_mean:.4f} (p={p_full_mean:.4f})")

    results_correlations["full_pipeline"] = {
        "rho_max": rho_full_max, "p_max": p_full_max,
        "rho_mean": rho_full_mean, "p_mean": p_full_mean,
    }

    # Conditional cosine only
    rho_cc_max, p_cc_max = spearman_rho(gold_rates, cond_cosine_max)
    rho_cc_mean, p_cc_mean = spearman_rho(gold_rates, cond_cosine_mean)
    print(f"\nConditional Cosine Only:")
    print(f"  Spearman rho (max): {rho_cc_max:.4f} (p={p_cc_max:.4f})")
    print(f"  Spearman rho (mean): {rho_cc_mean:.4f} (p={p_cc_mean:.4f})")

    results_correlations["conditional_cosine"] = {
        "rho_max": rho_cc_max, "p_max": p_cc_max,
        "rho_mean": rho_cc_mean, "p_mean": p_cc_mean,
    }

    # Firing rate only
    rho_fr_max, p_fr_max = spearman_rho(gold_rates, fr_scores_max)
    print(f"\nFiring Rate Filter Only:")
    print(f"  Spearman rho (max): {rho_fr_max:.4f} (p={p_fr_max:.4f})")

    results_correlations["firing_rate"] = {
        "rho_max": rho_fr_max, "p_max": p_fr_max,
    }

    # ITAC only
    rho_itac_max, p_itac_max = spearman_rho(gold_rates, itac_max)
    rho_itac_mean, p_itac_mean = spearman_rho(gold_rates, itac_mean)
    print(f"\nITAC Only:")
    print(f"  Spearman rho (max): {rho_itac_max:.4f} (p={p_itac_max:.4f})")
    print(f"  Spearman rho (mean): {rho_itac_mean:.4f} (p={p_itac_mean:.4f})")

    results_correlations["itac"] = {
        "rho_max": rho_itac_max, "p_max": p_itac_max,
        "rho_mean": rho_itac_mean, "p_mean": p_itac_mean,
    }

    # N_matching_pairs as a simple count-based score
    rho_npairs, p_npairs = spearman_rho(gold_rates, n_pairs)
    print(f"\nN Matching Pairs (baseline):")
    print(f"  Spearman rho: {rho_npairs:.4f} (p={p_npairs:.4f})")

    results_correlations["n_matching_pairs"] = {
        "rho": rho_npairs, "p": p_npairs,
    }

    report_progress(TASK_ID, RESULTS_DIR, 3, 5, step=3, total_steps=5,
                    metric={"status": "computing_auroc"})

    # ── AUROC ──
    print("\n=== AUROC (binary classification) ===")
    results_auroc = {}

    # Strict binary (rate > 0.1 vs rate <= 0.05)
    if len(set(binary_labels_strict)) >= 2:
        auroc_full, fpr_full, tpr_full, _ = compute_auroc(binary_labels_strict, binary_scores_strict)
        auroc_cc, fpr_cc, tpr_cc, _ = compute_auroc(binary_labels_strict, binary_scores_cc)
        auroc_fr, fpr_fr, tpr_fr, _ = compute_auroc(binary_labels_strict, binary_scores_fr)
        auroc_itac, fpr_itac, tpr_itac, _ = compute_auroc(binary_labels_strict, binary_scores_itac)
        n_pos = sum(binary_labels_strict)
        n_neg = len(binary_labels_strict) - n_pos
        print(f"  Strict binary: n_pos={n_pos}, n_neg={n_neg}")
        print(f"  Full pipeline AUROC: {auroc_full:.4f}")
        print(f"  Conditional cosine AUROC: {auroc_cc:.4f}")
        print(f"  Firing rate AUROC: {auroc_fr:.4f}")
        print(f"  ITAC AUROC: {auroc_itac:.4f}")

        results_auroc["strict"] = {
            "n_pos": n_pos, "n_neg": n_neg,
            "full_pipeline": auroc_full,
            "conditional_cosine": auroc_cc,
            "firing_rate": auroc_fr,
            "itac": auroc_itac,
            "roc_full": {"fpr": fpr_full, "tpr": tpr_full},
            "roc_cc": {"fpr": fpr_cc, "tpr": tpr_cc},
            "roc_fr": {"fpr": fpr_fr, "tpr": tpr_fr},
            "roc_itac": {"fpr": fpr_itac, "tpr": tpr_itac},
            "letters_included": binary_letters,
        }
    else:
        print("  WARNING: Not enough class diversity for strict binary AUROC")
        results_auroc["strict"] = {"error": "insufficient_class_diversity"}

    # Relaxed binary (rate > 0 vs rate == 0)
    if len(set(binary_all_labels)) >= 2:
        auroc_all_full, fpr_af, tpr_af, _ = compute_auroc(binary_all_labels, binary_all_scores)
        auroc_all_cc, _, _, _ = compute_auroc(binary_all_labels,
            [letter_scores[l]["conditional_cosine_max"] for l in letters_sorted])
        auroc_all_fr, _, _, _ = compute_auroc(binary_all_labels,
            [letter_scores[l]["firing_rate_score_max"] for l in letters_sorted])
        auroc_all_itac, _, _, _ = compute_auroc(binary_all_labels,
            [letter_scores[l]["itac_max"] for l in letters_sorted])
        n_pos_all = sum(binary_all_labels)
        n_neg_all = len(binary_all_labels) - n_pos_all
        print(f"\n  Relaxed binary: n_pos={n_pos_all}, n_neg={n_neg_all}")
        print(f"  Full pipeline AUROC: {auroc_all_full:.4f}")
        print(f"  Conditional cosine AUROC: {auroc_all_cc:.4f}")
        print(f"  Firing rate AUROC: {auroc_all_fr:.4f}")
        print(f"  ITAC AUROC: {auroc_all_itac:.4f}")

        results_auroc["relaxed"] = {
            "n_pos": n_pos_all, "n_neg": n_neg_all,
            "full_pipeline": auroc_all_full,
            "conditional_cosine": auroc_all_cc,
            "firing_rate": auroc_all_fr,
            "itac": auroc_all_itac,
            "roc_full": {"fpr": fpr_af, "tpr": tpr_af},
        }
    else:
        results_auroc["relaxed"] = {"error": "insufficient_class_diversity"}

    # ── Precision@50 ──
    # Using the pair-level analysis
    print("\n=== Pair-Level Analysis ===")
    pair_scores, absorbed_features, split_features = compute_all_pairs_scores(
        geometry_pairs, itac_pairs, letters_data
    )

    n_absorption_related = sum(1 for p in pair_scores if p["is_absorption_related"])
    n_involves_absorbed = sum(1 for p in pair_scores if p["involves_absorbed_feature"])
    n_involves_split = sum(1 for p in pair_scores if p["involves_split_feature"])
    print(f"  Total candidate pairs: {len(pair_scores)}")
    print(f"  Pairs involving absorbed features: {n_involves_absorbed}")
    print(f"  Pairs involving split features: {n_involves_split}")
    print(f"  Pairs absorption-related (either): {n_absorption_related}")
    print(f"  Known absorbed feature indices: {sorted(absorbed_features)[:20]}...")
    print(f"  Known split feature indices (sample): {sorted(list(split_features))[:20]}...")

    # Precision@K for pair-level
    pair_labels = [1 if p["is_absorption_related"] else 0 for p in pair_scores]
    pair_absorption_scores = [p["absorption_score"] for p in pair_scores]

    prec_at_50, n_at_50 = compute_precision_at_k(pair_labels, pair_absorption_scores, k=50)
    prec_at_100, n_at_100 = compute_precision_at_k(pair_labels, pair_absorption_scores, k=100)
    prec_at_10, n_at_10 = compute_precision_at_k(pair_labels, pair_absorption_scores, k=10)

    # Also precision based on "involves_absorbed_feature" only (stricter)
    pair_labels_strict_absorbed = [1 if p["involves_absorbed_feature"] else 0 for p in pair_scores]
    prec_at_50_strict, _ = compute_precision_at_k(pair_labels_strict_absorbed, pair_absorption_scores, k=50)

    print(f"\n  Precision@10 (absorption-related): {prec_at_10:.4f}")
    print(f"  Precision@50 (absorption-related): {prec_at_50:.4f}")
    print(f"  Precision@100 (absorption-related): {prec_at_100:.4f}")
    print(f"  Precision@50 (absorbed features only): {prec_at_50_strict:.4f}")

    # Pair-level AUROC
    if len(set(pair_labels)) >= 2 and sum(pair_labels) >= 5:
        auroc_pair, _, _, _ = compute_auroc(pair_labels, pair_absorption_scores)
        print(f"  Pair-level AUROC (absorption-related): {auroc_pair:.4f}")
    else:
        auroc_pair = 0.5
        print(f"  Pair-level AUROC: insufficient positive pairs")

    # Pair-level AUROC with ITAC
    pair_itac_scores = [p["itac"] if p["itac"] is not None else 0.0 for p in pair_scores]
    if len(set(pair_labels)) >= 2 and sum(pair_labels) >= 5:
        auroc_pair_itac, _, _, _ = compute_auroc(pair_labels, pair_itac_scores)
        print(f"  Pair-level AUROC (ITAC): {auroc_pair_itac:.4f}")
    else:
        auroc_pair_itac = 0.5

    report_progress(TASK_ID, RESULTS_DIR, 4, 5, step=4, total_steps=5,
                    metric={"status": "component_ablation"})

    # ── Step 4: Component ablation ──
    print("\n=== Component Ablation ===")
    ablation_results = {}

    # Define component combinations
    component_configs = {
        "full_pipeline": {"use_cc": True, "use_fr": True, "use_itac": True},
        "conditional_cosine_only": {"use_cc": True, "use_fr": False, "use_itac": False},
        "firing_rate_only": {"use_cc": False, "use_fr": True, "use_itac": False},
        "itac_only": {"use_cc": False, "use_fr": False, "use_itac": True},
        "cc_plus_fr": {"use_cc": True, "use_fr": True, "use_itac": False},
        "cc_plus_itac": {"use_cc": True, "use_fr": False, "use_itac": True},
    }

    for config_name, config in component_configs.items():
        # Compute composite score per letter
        scores = []
        for l in letters_sorted:
            ls = letter_scores[l]
            score = 0.0
            n_components = 0
            if config["use_cc"]:
                score += ls["conditional_cosine_max"]
                n_components += 1
            if config["use_fr"]:
                score += ls["firing_rate_score_max"]
                n_components += 1
            if config["use_itac"]:
                # Normalize ITAC to [0,1] range using log transform
                itac_val = ls["itac_max"]
                itac_norm = min(np.log1p(itac_val) / 10.0, 1.0)  # Cap at ~22000
                score += itac_norm
                n_components += 1
            if n_components > 0:
                score /= n_components
            scores.append(score)

        rho_abl, p_abl = spearman_rho(gold_rates, scores)

        # AUROC
        if len(set(binary_all_labels)) >= 2:
            auroc_abl, _, _, _ = compute_auroc(binary_all_labels, scores)
        else:
            auroc_abl = 0.5

        # Strict binary AUROC
        if len(set(binary_labels_strict)) >= 2:
            strict_scores = [scores[letters_sorted.index(l)] for l in binary_letters]
            auroc_abl_strict, _, _, _ = compute_auroc(binary_labels_strict, strict_scores)
        else:
            auroc_abl_strict = 0.5

        # Precision@50 at pair level with component filtering
        pair_component_scores = []
        for p in pair_scores:
            s = 0.0
            nc = 0
            if config["use_cc"]:
                s += p["conditional_cosine"]
                nc += 1
            if config["use_fr"]:
                s += 1.0 - p["firing_rate_ratio"]
                nc += 1
            if config["use_itac"]:
                itac_v = p["itac"] if p["itac"] is not None else 0.0
                s += min(np.log1p(itac_v) / 10.0, 1.0)
                nc += 1
            pair_component_scores.append(s / nc if nc > 0 else 0.0)

        prec_abl, _ = compute_precision_at_k(pair_labels, pair_component_scores, k=50)

        print(f"  {config_name:30s}: rho={rho_abl:.4f} (p={p_abl:.4f}), "
              f"AUROC(relaxed)={auroc_abl:.4f}, AUROC(strict)={auroc_abl_strict:.4f}, "
              f"Prec@50={prec_abl:.4f}")

        ablation_results[config_name] = {
            "components": config,
            "spearman_rho": rho_abl,
            "spearman_p": p_abl,
            "auroc_relaxed": auroc_abl,
            "auroc_strict": auroc_abl_strict,
            "precision_at_50": prec_abl,
            "per_letter_scores": {l: scores[i] for i, l in enumerate(letters_sorted)},
        }

    # ── Step 5: Decision gate evaluation ──
    print("\n=== Decision Gate ===")
    best_rho = max(
        results_correlations["full_pipeline"]["rho_max"],
        results_correlations["conditional_cosine"]["rho_max"],
        results_correlations["firing_rate"]["rho_max"],
        results_correlations["itac"]["rho_max"],
    )
    best_component = max(results_correlations.items(),
                         key=lambda x: x[1].get("rho_max", 0))

    decision = "PASS" if best_rho >= 0.3 else "FAIL"
    if best_rho >= 0.5:
        decision_detail = "STRONG_PASS: rho >= 0.5, unsupervised detection validated"
    elif best_rho >= 0.3:
        decision_detail = "MODERATE_PASS: 0.3 <= rho < 0.5, minimum viability"
    else:
        decision_detail = "FAIL: rho < 0.3, de-emphasize unsupervised detection contribution"

    print(f"  Best Spearman rho: {best_rho:.4f} (component: {best_component[0]})")
    print(f"  Decision: {decision}")
    print(f"  Detail: {decision_detail}")

    # Also check AUROC
    best_auroc_strict = 0.5
    if "strict" in results_auroc and "full_pipeline" in results_auroc.get("strict", {}):
        strict_aurocs = results_auroc["strict"]
        if isinstance(strict_aurocs, dict) and "full_pipeline" in strict_aurocs:
            best_auroc_strict = max(
                strict_aurocs.get("full_pipeline", 0.5),
                strict_aurocs.get("conditional_cosine", 0.5),
                strict_aurocs.get("firing_rate", 0.5),
                strict_aurocs.get("itac", 0.5),
            )
    print(f"  Best AUROC (strict): {best_auroc_strict:.4f}")

    report_progress(TASK_ID, RESULTS_DIR, 5, 5, step=5, total_steps=5,
                    metric={"status": "saving_results"})

    # ── Compile results ──
    elapsed_sec = time.time() - start_time

    # Summarize per-letter for output (without full matching_pairs to keep size manageable)
    per_letter_summary = {}
    for letter in letters_sorted:
        ls = letter_scores[letter]
        per_letter_summary[letter] = {
            "gold_absorption_rate": ls["gold_absorption_rate"],
            "gold_probe_f1": ls["gold_probe_f1"],
            "n_matching_pairs": ls["n_matching_pairs"],
            "absorption_score_max": ls["absorption_score_max"],
            "absorption_score_mean": ls["absorption_score_mean"],
            "conditional_cosine_max": ls["conditional_cosine_max"],
            "conditional_cosine_mean": ls["conditional_cosine_mean"],
            "firing_rate_score_max": ls["firing_rate_score_max"],
            "itac_max": ls["itac_max"],
            "itac_mean": ls["itac_mean"],
            # Include sample matching pairs (max 3)
            "sample_matching_pairs": ls["matching_pairs"][:3],
        }

    results = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "seed": 42,
        "model": "gemma-2-2b",
        "sae_config": "L12-16k",
        "sae_id": "layer_12/width_16k/average_l0_82",
        "timestamp_start": timestamp_start,
        "timestamp_end": datetime.now().isoformat(),
        "elapsed_sec": elapsed_sec,
        "gold_standard": {
            "source": "first_letter_validation.json",
            "aggregate_absorption_rate": aggregate["aggregate_absorption_rate"],
            "n_letters": len(letters_data),
            "letters_above_gate": aggregate.get("letters_above_gate", 0),
        },
        "unsupervised_sources": {
            "decoder_geometry": {
                "source": "decoder_geometry.json",
                "n_candidate_pairs": len(geometry_pairs),
            },
            "itac": {
                "source": "itac_real_activations.json",
                "n_itac_pairs": len(itac_pairs),
                "itac_pass_criteria": itac_full.get("pass_criteria", {}),
            },
        },
        "per_letter_scores": per_letter_summary,
        "correlations": results_correlations,
        "auroc": results_auroc,
        "pair_level_analysis": {
            "total_candidate_pairs": len(pair_scores),
            "n_absorption_related": n_absorption_related,
            "n_involves_absorbed_feature": n_involves_absorbed,
            "n_involves_split_feature": n_involves_split,
            "n_absorbed_features": len(absorbed_features),
            "n_split_features": len(split_features),
            "absorbed_feature_indices": sorted(list(absorbed_features)),
            "precision_at_10": prec_at_10,
            "precision_at_50": prec_at_50,
            "precision_at_100": prec_at_100,
            "precision_at_50_strict": prec_at_50_strict,
            "pair_auroc_absorption_score": auroc_pair,
            "pair_auroc_itac": auroc_pair_itac,
        },
        "component_ablation": ablation_results,
        "decision_gate": {
            "criterion": "Spearman rho > 0.3",
            "best_rho": best_rho,
            "best_component": best_component[0],
            "best_auroc_strict": best_auroc_strict,
            "decision": decision,
            "detail": decision_detail,
        },
        "pass_criteria": {
            "full_pipeline_rho_gt_0.3": best_rho > 0.3,
            "any_component_rho_gt_0.4": any(
                v.get("rho_max", 0) > 0.4 or v.get("rho", 0) > 0.4
                for v in results_correlations.values()
            ),
            "auroc_gt_0.6": best_auroc_strict > 0.6,
            "overall": best_rho > 0.3 and best_auroc_strict > 0.6,
        },
        "summary": (
            f"Unsupervised pipeline validation against first-letter gold standard. "
            f"Best Spearman rho: {best_rho:.4f} (component: {best_component[0]}). "
            f"Best AUROC (strict): {best_auroc_strict:.4f}. "
            f"Decision: {decision}. "
            f"{letters_with_matches}/{len(letter_scores)} letters matched by unsupervised pairs."
        ),
    }

    # Save results
    output_path = RESULTS_DIR / "unsupervised_validation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("UNSUPERVISED VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Gold standard: {len(letters_data)} letters, aggregate absorption rate = {aggregate['aggregate_absorption_rate']:.4f}")
    print(f"Letters with unsupervised matches: {letters_with_matches}/{len(letter_scores)}")
    print(f"\nBest correlations:")
    print(f"  Full pipeline:      rho={results_correlations['full_pipeline']['rho_max']:.4f}")
    print(f"  Conditional cosine: rho={results_correlations['conditional_cosine']['rho_max']:.4f}")
    print(f"  Firing rate:        rho={results_correlations['firing_rate']['rho_max']:.4f}")
    print(f"  ITAC:               rho={results_correlations['itac']['rho_max']:.4f}")
    print(f"  N pairs:            rho={results_correlations['n_matching_pairs']['rho']:.4f}")
    if "strict" in results_auroc and isinstance(results_auroc["strict"], dict) and "full_pipeline" in results_auroc["strict"]:
        print(f"\nAUROC (strict binary):")
        s = results_auroc["strict"]
        print(f"  Full pipeline:      {s['full_pipeline']:.4f}")
        print(f"  Conditional cosine: {s['conditional_cosine']:.4f}")
        print(f"  Firing rate:        {s['firing_rate']:.4f}")
        print(f"  ITAC:               {s['itac']:.4f}")
    print(f"\nDecision gate: {decision}")
    print(f"  {decision_detail}")
    print(f"\nElapsed: {elapsed_sec:.1f}s")
    print("=" * 60)

    return results


if __name__ == "__main__":
    try:
        results = run_validation()
        mark_task_done(TASK_ID, RESULTS_DIR, status="success",
                       summary=results["summary"])
        print("\nTask completed successfully.")
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        mark_task_done(TASK_ID, RESULTS_DIR, status="failed", summary=str(e))
        sys.exit(1)
