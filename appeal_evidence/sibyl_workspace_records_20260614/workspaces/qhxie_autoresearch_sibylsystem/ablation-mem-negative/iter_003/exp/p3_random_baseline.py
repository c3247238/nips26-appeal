#!/usr/bin/env python3
"""
Pilot P3: Random Baseline Validation for UAD

Computes comprehensive random baselines to validate whether UAD's performance
is significantly better than random chance.

Two baseline types:
1. Global random: Randomly sample pairs from all C(N,2) possible pairs
2. Same-cluster random: Randomly sample pairs from UAD-discovered clusters

Also computes empirical random baseline by actual random sampling.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tqdm import tqdm

from sae_lens import SAE
from transformer_lens import HookedTransformer

SEED = 42
MODEL_NAME = "gpt2"
SAE_ID = "gpt2-small-res-jb"
LAYER = 8
HOOK_NAME = f"blocks.{LAYER}.hook_resid_pre"
DATASET_SAMPLES = 1000
FEATURES_ANALYZED = 500
N_CLUSTERS = 50

RESULTS_DIR = Path("exp/results/pilots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "p3_random_baseline"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"
NUMBER_WORDS = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']


def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = {
        "task_id": TASK_ID, "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(progress))


def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    final_progress = {}
    if PROGRESS_FILE.exists():
        try:
            final_progress = json.loads(PROGRESS_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress, "timestamp": datetime.now().isoformat(),
    }))


def detect_number_features(model, sae, device="cuda"):
    """Detect number word features and absorption pairs (same as P2)."""
    number_acts = {}
    for word in NUMBER_WORDS:
        tokens = model.to_tokens(word)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[HOOK_NAME])
            resid = cache[HOOK_NAME].squeeze(0)
            acts = sae.encode(resid)
        if acts.shape[0] > 1:
            number_acts[word] = acts[1].cpu().numpy()

    number_primary = {}
    for word in NUMBER_WORDS:
        acts = number_acts[word]
        sorted_feats = np.argsort(acts)[::-1]
        best_feat = int(sorted_feats[0])
        if best_feat == 13586 and len(sorted_feats) > 1:
            best_feat = int(sorted_feats[1])
        number_primary[word] = {'feature': best_feat, 'activation': float(acts[best_feat])}

    threshold = 5.0
    absorption_features = {}
    for feat in range(sae.cfg.d_sae):
        if feat == 13586:
            continue
        nums_for_feat = []
        for word in NUMBER_WORDS:
            if number_acts[word][feat] > threshold:
                nums_for_feat.append(word)
        if len(nums_for_feat) > 1:
            absorption_features[feat] = nums_for_feat

    absorption_pairs = []
    for feat, nums in absorption_features.items():
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pair = tuple(sorted([nums[i], nums[j]]))
                absorption_pairs.append(pair)
    absorption_pairs = list(set(absorption_pairs))

    feature_pairs = []
    for w1, w2 in absorption_pairs:
        f1 = number_primary[w1]['feature']
        f2 = number_primary[w2]['feature']
        feature_pairs.append(tuple(sorted([f1, f2])))
    feature_pairs = list(set(feature_pairs))

    return {
        'number_primary': number_primary,
        'absorption_features': absorption_features,
        'absorption_pairs': absorption_pairs,
        'feature_pairs': feature_pairs,
    }


def collect_feature_activations(model, sae, texts, device="cuda", max_features=500, force_features=None):
    """Collect feature activations on dataset (same as P2)."""
    all_activations = []
    feature_counts = None

    model.eval()
    with torch.no_grad():
        for text in tqdm(texts, desc="Collecting activations"):
            tokens = model.to_tokens(text)
            if tokens.shape[1] < 2:
                continue
            _, cache = model.run_with_cache(tokens, names_filter=[HOOK_NAME])
            resid = cache[HOOK_NAME].squeeze(0)
            sae_acts = sae.encode(resid)
            binary_acts = (sae_acts > 0).float().cpu().numpy()
            all_activations.append(binary_acts)
            if feature_counts is None:
                feature_counts = binary_acts.sum(axis=0)
            else:
                feature_counts += binary_acts.sum(axis=0)

    if feature_counts is None:
        return None, None

    top_indices = np.argsort(feature_counts)[-max_features:]
    top_indices = set(top_indices.tolist())

    if force_features:
        top_indices.update(force_features)

    top_indices = sorted(top_indices)

    filtered_activations = []
    for acts in all_activations:
        filtered_activations.append(acts[:, top_indices])

    return filtered_activations, top_indices


def compute_cooccurrence_matrix(activations, feature_indices):
    """Compute phi coefficient matrix (same as P2)."""
    n_features = len(feature_indices)
    cooccur = np.zeros((n_features, n_features))
    marginal = np.zeros(n_features)
    total = 0

    for acts in activations:
        for pos in range(acts.shape[0]):
            feat_vec = acts[pos]
            cooccur += np.outer(feat_vec, feat_vec)
            marginal += feat_vec
            total += 1

    phi_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            a = cooccur[i, j]
            b = marginal[i] - cooccur[i, j]
            c = marginal[j] - cooccur[i, j]
            d = total - marginal[i] - marginal[j] + cooccur[i, j]
            denom = np.sqrt((a + b) * (a + c) * (b + d) * (c + d))
            if denom > 0:
                phi_matrix[i, j] = (a * d - b * c) / denom
            else:
                phi_matrix[i, j] = 0.0

    return phi_matrix, marginal / total, cooccur


def uad_detect_absorption(phi_matrix, feature_indices, n_clusters=50):
    """Run UAD clustering (same as P2)."""
    dist_matrix = 1 - np.abs(phi_matrix)
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="ward")
    clusters = fcluster(Z, n_clusters, criterion="maxclust")

    cluster_map = {}
    for idx, clust_id in enumerate(clusters):
        if clust_id not in cluster_map:
            cluster_map[clust_id] = []
        cluster_map[clust_id].append(idx)

    candidate_pairs = []
    for clust_id, members in cluster_map.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                idx_i = members[i]
                idx_j = members[j]
                feat_i = feature_indices[idx_i]
                feat_j = feature_indices[idx_j]
                phi = phi_matrix[idx_i, idx_j]
                candidate_pairs.append((feat_i, feat_j, float(phi), int(clust_id)))

    return candidate_pairs, clusters, cluster_map


def filter_dead_features(candidate_pairs, feature_indices, marginal_freqs, threshold=0.001):
    """Filter dead features (same as P2)."""
    alive_features = set()
    for idx, freq in enumerate(marginal_freqs):
        if freq >= threshold:
            alive_features.add(feature_indices[idx])

    filtered = []
    for feat_i, feat_j, phi, clust_id in candidate_pairs:
        if feat_i in alive_features and feat_j in alive_features:
            filtered.append((feat_i, feat_j, phi, clust_id))
    return filtered


def evaluate_pairs(detected_pairs, ground_truth_pairs):
    """Evaluate detected pairs against ground truth."""
    detected_set = set()
    for feat_i, feat_j, phi, clust_id in detected_pairs:
        detected_set.add(tuple(sorted([feat_i, feat_j])))

    gt_set = set(tuple(p) for p in ground_truth_pairs)

    tp = len(detected_set & gt_set)
    fp = len(detected_set - gt_set)
    fn = len(gt_set - detected_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "detected_pairs": len(detected_set), "ground_truth_pairs": len(gt_set),
    }


def compute_global_random_baseline(n_features, n_gt_pairs, n_detected, n_trials=100):
    """
    Global random baseline: randomly sample n_detected pairs from all C(n_features, 2) pairs.
    Uses analytical expectation (no need for actual sampling).
    """
    n_all_pairs = n_features * (n_features - 1) // 2
    f1_scores = []
    precisions = []
    recalls = []

    for _ in range(n_trials):
        # Expected true positives when sampling n_detected pairs uniformly
        expected_tp = n_detected * n_gt_pairs / n_all_pairs
        expected_fp = n_detected - expected_tp
        expected_fn = n_gt_pairs - expected_tp

        p = expected_tp / (expected_tp + expected_fp) if (expected_tp + expected_fp) > 0 else 0
        r = expected_tp / (expected_tp + expected_fn) if (expected_tp + expected_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        f1_scores.append(f1)
        precisions.append(p)
        recalls.append(r)

    return {
        "mean_f1": float(np.mean(f1_scores)),
        "std_f1": float(np.std(f1_scores)),
        "mean_precision": float(np.mean(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "n_trials": n_trials,
        "baseline_type": "global_random",
        "n_all_pairs": n_all_pairs,
        "n_detected": n_detected,
        "n_gt_pairs": n_gt_pairs,
    }


def compute_empirical_global_random_baseline(feature_indices, gt_pairs, n_detected, n_trials=100, seed=42):
    """
    Empirical global random: actually sample random pairs and evaluate.
    """
    rng = np.random.RandomState(seed)
    all_pairs = []
    for i in range(len(feature_indices)):
        for j in range(i + 1, len(feature_indices)):
            all_pairs.append(tuple(sorted([feature_indices[i], feature_indices[j]])))

    gt_set = set(tuple(p) for p in gt_pairs)
    f1_scores = []
    precisions = []
    recalls = []
    tp_counts = []

    for _ in range(n_trials):
        if n_detected >= len(all_pairs):
            sampled = all_pairs
        else:
            sampled_idx = rng.choice(len(all_pairs), size=n_detected, replace=False)
            sampled = [all_pairs[i] for i in sampled_idx]

        sampled_set = set(sampled)
        tp = len(sampled_set & gt_set)
        fp = len(sampled_set - gt_set)
        fn = len(gt_set - sampled_set)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        f1_scores.append(f1)
        precisions.append(p)
        recalls.append(r)
        tp_counts.append(tp)

    return {
        "mean_f1": float(np.mean(f1_scores)),
        "std_f1": float(np.std(f1_scores)),
        "mean_precision": float(np.mean(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "mean_tp": float(np.mean(tp_counts)),
        "n_trials": n_trials,
        "baseline_type": "empirical_global_random",
    }


def compute_same_cluster_random_baseline(cluster_map, feature_indices, gt_pairs, n_detected, n_trials=100, seed=42):
    """
    Same-cluster random baseline: sample pairs only from within UAD-discovered clusters.
    This tests whether the clustering step itself provides value.
    """
    rng = np.random.RandomState(seed)

    # Build list of all within-cluster pairs
    cluster_pairs = []
    for clust_id, members in cluster_map.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                feat_i = feature_indices[members[i]]
                feat_j = feature_indices[members[j]]
                cluster_pairs.append(tuple(sorted([feat_i, feat_j])))

    gt_set = set(tuple(p) for p in gt_pairs)
    f1_scores = []
    precisions = []
    recalls = []
    tp_counts = []

    for _ in range(n_trials):
        if n_detected >= len(cluster_pairs):
            sampled = cluster_pairs
        else:
            sampled_idx = rng.choice(len(cluster_pairs), size=n_detected, replace=False)
            sampled = [cluster_pairs[i] for i in sampled_idx]

        sampled_set = set(sampled)
        tp = len(sampled_set & gt_set)
        fp = len(sampled_set - gt_set)
        fn = len(gt_set - sampled_set)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        f1_scores.append(f1)
        precisions.append(p)
        recalls.append(r)
        tp_counts.append(tp)

    return {
        "mean_f1": float(np.mean(f1_scores)),
        "std_f1": float(np.std(f1_scores)),
        "mean_precision": float(np.mean(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "mean_tp": float(np.mean(tp_counts)),
        "n_trials": n_trials,
        "baseline_type": "same_cluster_random",
        "n_cluster_pairs": len(cluster_pairs),
    }


def compute_frequency_weighted_baseline(feature_indices, marginal_freqs, gt_pairs, n_detected, n_trials=100, seed=42):
    """
    Frequency-weighted random: sample pairs with probability proportional to
    the product of feature frequencies (more likely to co-occur).
    """
    rng = np.random.RandomState(seed)

    # Compute sampling weights as product of marginal frequencies
    weights = []
    pair_list = []
    for i in range(len(feature_indices)):
        for j in range(i + 1, len(feature_indices)):
            w = marginal_freqs[i] * marginal_freqs[j]
            weights.append(w)
            pair_list.append(tuple(sorted([feature_indices[i], feature_indices[j]])))

    weights = np.array(weights)
    weights = weights / weights.sum()

    gt_set = set(tuple(p) for p in gt_pairs)
    f1_scores = []
    precisions = []
    recalls = []
    tp_counts = []

    for _ in range(n_trials):
        sampled_idx = rng.choice(len(pair_list), size=n_detected, replace=False, p=weights)
        sampled = [pair_list[i] for i in sampled_idx]

        sampled_set = set(sampled)
        tp = len(sampled_set & gt_set)
        fp = len(sampled_set - gt_set)
        fn = len(gt_set - sampled_set)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        f1_scores.append(f1)
        precisions.append(p)
        recalls.append(r)
        tp_counts.append(tp)

    return {
        "mean_f1": float(np.mean(f1_scores)),
        "std_f1": float(np.std(f1_scores)),
        "mean_precision": float(np.mean(precisions)),
        "mean_recall": float(np.mean(recalls)),
        "mean_tp": float(np.mean(tp_counts)),
        "n_trials": n_trials,
        "baseline_type": "frequency_weighted_random",
    }


def main():
    start_time = time.time()
    PID_FILE.write_text(str(os.getpid()))
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    report_progress(0, 5, metric={"stage": "loading_model"})
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    print(f"Model loaded. Layers: {model.cfg.n_layers}")

    report_progress(1, 5, metric={"stage": "loading_sae"})
    print(f"Loading SAE: {SAE_ID}, layer {LAYER}...")
    sae = SAE.from_pretrained(release=SAE_ID, sae_id=f"blocks.{LAYER}.hook_resid_pre", device=device)
    print(f"SAE loaded. d_sae: {sae.cfg.d_sae}, d_in: {sae.cfg.d_in}")

    report_progress(2, 5, metric={"stage": "ground_truth"})
    print("Building ground truth (number word features)...")
    gt = detect_number_features(model, sae, device)
    print(f"Detected {len(gt['number_primary'])} number features")
    print(f"Absorption features: {list(gt['absorption_features'].keys())}")
    print(f"Absorption pairs: {len(gt['absorption_pairs'])}")
    print(f"Feature absorption pairs: {len(gt['feature_pairs'])}")

    gt_feats = set()
    for p in gt['feature_pairs']:
        gt_feats.update(p)
    print(f"GT feature indices: {sorted(gt_feats)}")

    report_progress(3, 5, metric={"stage": "loading_dataset"})
    print("\nLoading OpenWebText samples...")
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    ds = ds.shuffle(seed=SEED)
    texts = []
    for i, example in enumerate(ds):
        if i >= DATASET_SAMPLES:
            break
        texts.append(example["text"])
    print(f"Loaded {len(texts)} texts")

    report_progress(4, 5, metric={"stage": "uad_analysis"})
    print("\nRunning UAD to get detected pairs...")
    print("  Collecting feature activations...")
    activations, feature_indices = collect_feature_activations(
        model, sae, texts, device, max_features=FEATURES_ANALYZED, force_features=gt_feats
    )
    print(f"  Analyzing {len(feature_indices)} features")

    print("  Computing co-occurrence matrix...")
    phi_matrix, marginal_freqs, cooccur = compute_cooccurrence_matrix(activations, feature_indices)

    print("  Running hierarchical clustering...")
    candidate_pairs, clusters, cluster_map = uad_detect_absorption(phi_matrix, feature_indices, n_clusters=N_CLUSTERS)
    print(f"  Raw candidate pairs: {len(candidate_pairs)}")

    print("  Filtering dead features...")
    filtered_pairs = filter_dead_features(candidate_pairs, feature_indices, marginal_freqs, threshold=0.001)
    print(f"  After dead feature filtering: {len(filtered_pairs)}")

    # Evaluate UAD
    print("\nEvaluating UAD...")
    uad_results = evaluate_pairs(filtered_pairs, gt['feature_pairs'])
    print(f"UAD F1: {uad_results['f1']:.3f}, Precision: {uad_results['precision']:.3f}, Recall: {uad_results['recall']:.3f}")

    # Compute all baselines
    n_features = len(feature_indices)
    n_gt = len(gt['feature_pairs'])
    n_detected = len(filtered_pairs)

    print("\n" + "=" * 60)
    print("COMPUTING RANDOM BASELINES")
    print("=" * 60)

    # 1. Analytical global random
    print("\n1. Analytical global random baseline...")
    global_random = compute_global_random_baseline(n_features, n_gt, n_detected, n_trials=100)
    print(f"   Mean F1: {global_random['mean_f1']:.4f} (+/- {global_random['std_f1']:.4f})")
    print(f"   Mean Precision: {global_random['mean_precision']:.4f}")
    print(f"   Mean Recall: {global_random['mean_recall']:.4f}")

    # 2. Empirical global random
    print("\n2. Empirical global random baseline...")
    empirical_global = compute_empirical_global_random_baseline(
        feature_indices, gt['feature_pairs'], n_detected, n_trials=100, seed=SEED
    )
    print(f"   Mean F1: {empirical_global['mean_f1']:.4f} (+/- {empirical_global['std_f1']:.4f})")
    print(f"   Mean Precision: {empirical_global['mean_precision']:.4f}")
    print(f"   Mean Recall: {empirical_global['mean_recall']:.4f}")
    print(f"   Mean TP: {empirical_global['mean_tp']:.2f}")

    # 3. Same-cluster random
    print("\n3. Same-cluster random baseline...")
    same_cluster = compute_same_cluster_random_baseline(
        cluster_map, feature_indices, gt['feature_pairs'], n_detected, n_trials=100, seed=SEED
    )
    print(f"   Mean F1: {same_cluster['mean_f1']:.4f} (+/- {same_cluster['std_f1']:.4f})")
    print(f"   Mean Precision: {same_cluster['mean_precision']:.4f}")
    print(f"   Mean Recall: {same_cluster['mean_recall']:.4f}")
    print(f"   Mean TP: {same_cluster['mean_tp']:.2f}")
    print(f"   Cluster pairs available: {same_cluster['n_cluster_pairs']}")

    # 4. Frequency-weighted random
    print("\n4. Frequency-weighted random baseline...")
    freq_weighted = compute_frequency_weighted_baseline(
        feature_indices, marginal_freqs, gt['feature_pairs'], n_detected, n_trials=100, seed=SEED
    )
    print(f"   Mean F1: {freq_weighted['mean_f1']:.4f} (+/- {freq_weighted['std_f1']:.4f})")
    print(f"   Mean Precision: {freq_weighted['mean_precision']:.4f}")
    print(f"   Mean Recall: {freq_weighted['mean_recall']:.4f}")
    print(f"   Mean TP: {freq_weighted['mean_tp']:.2f}")

    # Compute differences
    uad_f1 = uad_results['f1']
    uad_precision = uad_results['precision']
    uad_recall = uad_results['recall']

    output = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": MODEL_NAME, "sae_id": SAE_ID, "layer": LAYER,
            "dataset_samples": DATASET_SAMPLES, "features_analyzed": len(feature_indices),
            "n_clusters": N_CLUSTERS, "seed": SEED,
        },
        "ground_truth": {
            "n_absorption_pairs": n_gt,
            "absorption_pairs": [list(p) for p in gt['feature_pairs']],
        },
        "uad_results": uad_results,
        "baselines": {
            "analytical_global_random": global_random,
            "empirical_global_random": empirical_global,
            "same_cluster_random": same_cluster,
            "frequency_weighted_random": freq_weighted,
        },
        "comparisons": {
            "uad_vs_global_random": {
                "uad_f1": uad_f1,
                "global_random_f1": global_random['mean_f1'],
                "f1_difference": uad_f1 - global_random['mean_f1'],
                "uad_better": uad_f1 > global_random['mean_f1'],
            },
            "uad_vs_empirical_global": {
                "uad_f1": uad_f1,
                "empirical_global_f1": empirical_global['mean_f1'],
                "f1_difference": uad_f1 - empirical_global['mean_f1'],
                "uad_better": uad_f1 > empirical_global['mean_f1'],
            },
            "uad_vs_same_cluster": {
                "uad_f1": uad_f1,
                "same_cluster_f1": same_cluster['mean_f1'],
                "f1_difference": uad_f1 - same_cluster['mean_f1'],
                "uad_better": uad_f1 > same_cluster['mean_f1'],
            },
            "uad_vs_frequency_weighted": {
                "uad_f1": uad_f1,
                "freq_weighted_f1": freq_weighted['mean_f1'],
                "f1_difference": uad_f1 - freq_weighted['mean_f1'],
                "uad_better": uad_f1 > freq_weighted['mean_f1'],
            },
        },
        "pass_criteria": {
            "global_random_f1_lt_0.05": global_random['mean_f1'] < 0.05,
            "uad_minus_global_ge_0.3": (uad_f1 - global_random['mean_f1']) >= 0.3,
            "all_pass": global_random['mean_f1'] < 0.05 and (uad_f1 - global_random['mean_f1']) >= 0.3,
        },
        "runtime_seconds": time.time() - start_time,
    }

    results_file = RESULTS_DIR / "p3_random_baseline_results.json"
    results_file.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {results_file}")

    print("\n" + "=" * 60)
    print("RANDOM BASELINE VALIDATION RESULTS")
    print("=" * 60)
    print(f"UAD F1:              {uad_f1:.4f}")
    print(f"Global random F1:    {global_random['mean_f1']:.4f} (+/- {global_random['std_f1']:.4f})")
    print(f"Empirical global F1: {empirical_global['mean_f1']:.4f} (+/- {empirical_global['std_f1']:.4f})")
    print(f"Same-cluster F1:     {same_cluster['mean_f1']:.4f} (+/- {same_cluster['std_f1']:.4f})")
    print(f"Freq-weighted F1:    {freq_weighted['mean_f1']:.4f} (+/- {freq_weighted['std_f1']:.4f})")
    print("-" * 60)
    print(f"UAD - Global random:     {uad_f1 - global_random['mean_f1']:+.4f}")
    print(f"UAD - Same-cluster:      {uad_f1 - same_cluster['mean_f1']:+.4f}")
    print(f"UAD - Freq-weighted:     {uad_f1 - freq_weighted['mean_f1']:+.4f}")
    print("-" * 60)

    # Pass criteria
    crit1 = global_random['mean_f1'] < 0.05
    crit2 = (uad_f1 - global_random['mean_f1']) >= 0.3
    print(f"\nPass criteria:")
    print(f"  Global random F1 < 0.05:     {global_random['mean_f1']:.4f} < 0.05 -> {'PASS' if crit1 else 'FAIL'}")
    print(f"  UAD F1 - random F1 >= 0.3:   {uad_f1 - global_random['mean_f1']:.4f} >= 0.3 -> {'PASS' if crit2 else 'FAIL'}")
    print(f"  OVERALL: {'PASS' if (crit1 and crit2) else 'FAIL'}")
    print("=" * 60)

    mark_done(
        status="success" if (crit1 and crit2) else "partial",
        summary=f"UAD F1={uad_f1:.4f}, Global random F1={global_random['mean_f1']:.4f}, Diff={uad_f1 - global_random['mean_f1']:+.4f}"
    )
    return output


if __name__ == "__main__":
    main()
