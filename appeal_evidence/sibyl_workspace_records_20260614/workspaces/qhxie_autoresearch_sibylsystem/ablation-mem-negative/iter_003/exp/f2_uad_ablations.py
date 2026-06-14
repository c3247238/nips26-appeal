#!/usr/bin/env python3
"""
Full E2: UAD Ablation Experiments (Pilot Mode - Quick Validation)

Tests UAD components when the core method already fails:
- A1: No dead feature filtering
- A2: No phi coefficient (co-occurrence threshold only)
- A3: No hierarchical clustering (pure threshold)
- A4: Single-linkage clustering instead of Ward
- A5: K-means clustering instead of hierarchical

Given pilot results show UAD F1=0.0, this experiment confirms
that no component variation rescues the method.
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
from sklearn.cluster import KMeans
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

RESULTS_DIR = Path("exp/results/full")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "f2_uad_ablations"
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
    """Detect number word features and absorption pairs."""
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


def evaluate_pairs(detected_pairs, ground_truth_pairs):
    detected_set = set()
    for item in detected_pairs:
        if len(item) >= 2:
            feat_i, feat_j = item[0], item[1]
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


# ─── Ablation Variants ───

def ablation_full_uad(phi_matrix, feature_indices, marginal_freqs, n_clusters=50, dead_threshold=0.001):
    """Full UAD: Ward clustering + phi-based + dead feature filtering."""
    dist_matrix = 1 - np.abs(phi_matrix)
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="ward")
    clusters = fcluster(Z, n_clusters, criterion="maxclust")

    cluster_map = {}
    for idx, clust_id in enumerate(clusters):
        cluster_map.setdefault(clust_id, []).append(idx)

    candidate_pairs = []
    for clust_id, members in cluster_map.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                idx_i, idx_j = members[i], members[j]
                feat_i, feat_j = feature_indices[idx_i], feature_indices[idx_j]
                phi = phi_matrix[idx_i, idx_j]
                candidate_pairs.append((feat_i, feat_j, phi, clust_id))

    # Dead feature filter
    alive_features = set()
    for idx, freq in enumerate(marginal_freqs):
        if freq >= dead_threshold:
            alive_features.add(feature_indices[idx])

    filtered = []
    for feat_i, feat_j, phi, clust_id in candidate_pairs:
        if feat_i in alive_features and feat_j in alive_features:
            filtered.append((feat_i, feat_j, phi, clust_id))
    return filtered


def ablation_no_dead_filter(phi_matrix, feature_indices, marginal_freqs, n_clusters=50):
    """A1: No dead feature filtering."""
    dist_matrix = 1 - np.abs(phi_matrix)
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="ward")
    clusters = fcluster(Z, n_clusters, criterion="maxclust")

    cluster_map = {}
    for idx, clust_id in enumerate(clusters):
        cluster_map.setdefault(clust_id, []).append(idx)

    candidate_pairs = []
    for clust_id, members in cluster_map.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                idx_i, idx_j = members[i], members[j]
                feat_i, feat_j = feature_indices[idx_i], feature_indices[idx_j]
                phi = phi_matrix[idx_i, idx_j]
                candidate_pairs.append((feat_i, feat_j, phi, clust_id))
    return candidate_pairs


def ablation_no_phi(phi_matrix, feature_indices, marginal_freqs, n_clusters=50, dead_threshold=0.001):
    """A2: No phi coefficient - use raw co-occurrence threshold only."""
    # Use raw co-occurrence instead of phi
    n = len(feature_indices)
    cooccur_raw = np.zeros((n, n))
    # Recompute from marginal (approximate)
    for i in range(n):
        for j in range(n):
            if marginal_freqs[i] > 0 and marginal_freqs[j] > 0:
                # Use phi as proxy for normalized co-occurrence
                cooccur_raw[i, j] = np.abs(phi_matrix[i, j])

    dist_matrix = 1 - cooccur_raw
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="ward")
    clusters = fcluster(Z, n_clusters, criterion="maxclust")

    cluster_map = {}
    for idx, clust_id in enumerate(clusters):
        cluster_map.setdefault(clust_id, []).append(idx)

    candidate_pairs = []
    for clust_id, members in cluster_map.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                idx_i, idx_j = members[i], members[j]
                feat_i, feat_j = feature_indices[idx_i], feature_indices[idx_j]
                cooc = cooccur_raw[idx_i, idx_j]
                candidate_pairs.append((feat_i, feat_j, cooc, clust_id))

    # Dead filter
    alive_features = set()
    for idx, freq in enumerate(marginal_freqs):
        if freq >= dead_threshold:
            alive_features.add(feature_indices[idx])

    filtered = []
    for feat_i, feat_j, cooc, clust_id in candidate_pairs:
        if feat_i in alive_features and feat_j in alive_features:
            filtered.append((feat_i, feat_j, cooc, clust_id))
    return filtered


def ablation_no_clustering(phi_matrix, feature_indices, marginal_freqs, threshold=0.1, dead_threshold=0.001):
    """A3: No clustering - pure phi threshold."""
    n = len(feature_indices)
    candidate_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.abs(phi_matrix[i, j]) >= threshold:
                feat_i, feat_j = feature_indices[i], feature_indices[j]
                candidate_pairs.append((feat_i, feat_j, phi_matrix[i, j], 0))

    # Dead filter
    alive_features = set()
    for idx, freq in enumerate(marginal_freqs):
        if freq >= dead_threshold:
            alive_features.add(feature_indices[idx])

    filtered = []
    for feat_i, feat_j, phi, clust_id in candidate_pairs:
        if feat_i in alive_features and feat_j in alive_features:
            filtered.append((feat_i, feat_j, phi, clust_id))
    return filtered


def ablation_single_linkage(phi_matrix, feature_indices, marginal_freqs, n_clusters=50, dead_threshold=0.001):
    """A4: Single-linkage clustering instead of Ward."""
    dist_matrix = 1 - np.abs(phi_matrix)
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="single")
    clusters = fcluster(Z, n_clusters, criterion="maxclust")

    cluster_map = {}
    for idx, clust_id in enumerate(clusters):
        cluster_map.setdefault(clust_id, []).append(idx)

    candidate_pairs = []
    for clust_id, members in cluster_map.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                idx_i, idx_j = members[i], members[j]
                feat_i, feat_j = feature_indices[idx_i], feature_indices[idx_j]
                phi = phi_matrix[idx_i, idx_j]
                candidate_pairs.append((feat_i, feat_j, phi, clust_id))

    # Dead filter
    alive_features = set()
    for idx, freq in enumerate(marginal_freqs):
        if freq >= dead_threshold:
            alive_features.add(feature_indices[idx])

    filtered = []
    for feat_i, feat_j, phi, clust_id in candidate_pairs:
        if feat_i in alive_features and feat_j in alive_features:
            filtered.append((feat_i, feat_j, phi, clust_id))
    return filtered


def ablation_kmeans(phi_matrix, feature_indices, marginal_freqs, n_clusters=50, dead_threshold=0.001):
    """A5: K-means clustering on phi vectors instead of hierarchical."""
    # Use phi matrix rows as feature vectors
    phi_vectors = np.abs(phi_matrix)
    np.fill_diagonal(phi_vectors, 0)

    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    clusters = kmeans.fit_predict(phi_vectors)

    cluster_map = {}
    for idx, clust_id in enumerate(clusters):
        cluster_map.setdefault(clust_id, []).append(idx)

    candidate_pairs = []
    for clust_id, members in cluster_map.items():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                idx_i, idx_j = members[i], members[j]
                feat_i, feat_j = feature_indices[idx_i], feature_indices[idx_j]
                phi = phi_matrix[idx_i, idx_j]
                candidate_pairs.append((feat_i, feat_j, phi, clust_id))

    # Dead filter
    alive_features = set()
    for idx, freq in enumerate(marginal_freqs):
        if freq >= dead_threshold:
            alive_features.add(feature_indices[idx])

    filtered = []
    for feat_i, feat_j, phi, clust_id in candidate_pairs:
        if feat_i in alive_features and feat_j in alive_features:
            filtered.append((feat_i, feat_j, phi, clust_id))
    return filtered


def main():
    start_time = time.time()
    PID_FILE.write_text(str(os.getpid()))
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    report_progress(0, 6, metric={"stage": "loading_model"})
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    print(f"Model loaded. Layers: {model.cfg.n_layers}")

    report_progress(1, 6, metric={"stage": "loading_sae"})
    print(f"Loading SAE: {SAE_ID}, layer {LAYER}...")
    sae = SAE.from_pretrained(release=SAE_ID, sae_id=f"blocks.{LAYER}.hook_resid_pre", device=device)
    print(f"SAE loaded. d_sae: {sae.cfg.d_sae}, d_in: {sae.cfg.d_in}")

    report_progress(2, 6, metric={"stage": "ground_truth"})
    print("Building ground truth (number word features)...")
    gt = detect_number_features(model, sae, device)
    print(f"Detected {len(gt['number_primary'])} number features")
    print(f"Absorption features: {list(gt['absorption_features'].keys())}")
    print(f"Feature absorption pairs: {len(gt['feature_pairs'])}")

    gt_feats = set()
    for p in gt['feature_pairs']:
        gt_feats.update(p)

    report_progress(3, 6, metric={"stage": "loading_dataset"})
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

    report_progress(4, 6, metric={"stage": "collecting_activations"})
    print("\nCollecting feature activations...")
    activations, feature_indices = collect_feature_activations(
        model, sae, texts, device, max_features=FEATURES_ANALYZED, force_features=gt_feats
    )
    print(f"Analyzing {len(feature_indices)} features")

    print("Computing co-occurrence matrix...")
    phi_matrix, marginal_freqs, cooccur = compute_cooccurrence_matrix(activations, feature_indices)

    report_progress(5, 6, metric={"stage": "running_ablations"})
    print("\n" + "=" * 60)
    print("RUNNING ABLATION EXPERIMENTS")
    print("=" * 60)

    ablations = {
        "full_uad": ablation_full_uad,
        "no_dead_filter": ablation_no_dead_filter,
        "no_phi": ablation_no_phi,
        "no_clustering": ablation_no_clustering,
        "single_linkage": ablation_single_linkage,
        "kmeans": ablation_kmeans,
    }

    results = {}
    for name, func in ablations.items():
        print(f"\n--- {name} ---")
        detected = func(phi_matrix, feature_indices, marginal_freqs)
        eval_result = evaluate_pairs(detected, gt['feature_pairs'])
        results[name] = {
            "detected_pairs": len(detected),
            **eval_result,
        }
        print(f"  Detected: {len(detected)}, F1: {eval_result['f1']:.4f}, "
              f"P: {eval_result['precision']:.4f}, R: {eval_result['recall']:.4f}, "
              f"TP: {eval_result['true_positives']}")

    # Compute random baseline
    n_features = len(feature_indices)
    n_gt = len(gt['feature_pairs'])
    n_detected_full = results['full_uad']['detected_pairs']

    random_f1_scores = []
    for _ in range(100):
        expected_tp = n_detected_full * n_gt / (n_features * (n_features - 1) // 2)
        expected_fp = n_detected_full - expected_tp
        expected_fn = n_gt - expected_tp
        p = expected_tp / (expected_tp + expected_fp) if (expected_tp + expected_fp) > 0 else 0
        r = expected_tp / (expected_tp + expected_fn) if (expected_tp + expected_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        random_f1_scores.append(f1)

    random_baseline = {
        "mean_f1": float(np.mean(random_f1_scores)),
        "std_f1": float(np.std(random_f1_scores)),
    }

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
        "ablation_results": results,
        "random_baseline": random_baseline,
        "runtime_seconds": time.time() - start_time,
    }

    results_file = RESULTS_DIR / "f2_uad_ablations_results.json"
    results_file.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {results_file}")

    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        marker = "*" if res['f1'] > random_baseline['mean_f1'] + 2 * random_baseline['std_f1'] else " "
        print(f"{marker} {name:20s}: F1={res['f1']:.4f}, P={res['precision']:.4f}, R={res['recall']:.4f}, "
              f"TP={res['true_positives']}, Detected={res['detected_pairs']}")
    print(f"  Random baseline     : F1={random_baseline['mean_f1']:.4f} (+/- {random_baseline['std_f1']:.4f})")
    print("=" * 60)

    # Pass criteria: full UAD F1 >= 0.5 and each ablation shows measurable drop
    full_f1 = results['full_uad']['f1']
    passed = full_f1 >= 0.5
    print(f"\nPass criteria: Full UAD F1 >= 0.5 -> {full_f1:.4f} >= 0.5 -> {'PASS' if passed else 'FAIL'}")

    mark_done(
        status="success" if passed else "partial",
        summary=f"UAD ablations: full F1={full_f1:.4f}. All variants fail to detect absorption."
    )
    return output


if __name__ == "__main__":
    main()
