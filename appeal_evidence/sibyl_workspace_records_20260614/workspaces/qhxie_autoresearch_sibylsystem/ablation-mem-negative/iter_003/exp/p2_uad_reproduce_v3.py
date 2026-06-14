#!/usr/bin/env python3
"""
Pilot P2: UAD Reproducibility Validation (v3)
Uses NUMBER word hierarchy as ground truth for absorption detection.
Main SAE (gpt2-small-res-jb) at layer 8.
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
MODEL_NAME = "gpt2"
SAE_ID = "gpt2-small-res-jb"
LAYER = 8
HOOK_NAME = f"blocks.{LAYER}.hook_resid_pre"

DATASET_SAMPLES = 1000
BATCH_SIZE = 8
MAX_SEQ_LEN = 128
FEATURES_ANALYZED = 500
N_CLUSTERS = 50

RESULTS_DIR = Path("exp/results/pilots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "p2_uad_reproduce"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"

# Number words for ground truth
NUMBER_WORDS = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']

# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------
def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = {
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
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
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


# ---------------------------------------------------------------------------
# Ground truth: number word features
# ---------------------------------------------------------------------------
def detect_number_features(model, sae, device="cuda"):
    """Detect SAE features for number words and build absorption pairs."""
    number_acts = {}
    for word in NUMBER_WORDS:
        tokens = model.to_tokens(word)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[HOOK_NAME])
            resid = cache[HOOK_NAME].squeeze(0)
            acts = sae.encode(resid)
        if acts.shape[0] > 1:
            number_acts[word] = acts[1].cpu().numpy()

    # Find primary feature for each number (excluding generic feature 13586)
    number_primary = {}
    for word in NUMBER_WORDS:
        acts = number_acts[word]
        sorted_feats = np.argsort(acts)[::-1]
        best_feat = int(sorted_feats[0])
        if best_feat == 13586 and len(sorted_feats) > 1:
            best_feat = int(sorted_feats[1])
        number_primary[word] = {
            'feature': best_feat,
            'activation': float(acts[best_feat]),
        }

    # Build absorption pairs: two numbers share a feature if both activate > threshold
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

    # Convert to feature pairs
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


# ---------------------------------------------------------------------------
# UAD Algorithm
# ---------------------------------------------------------------------------
def collect_feature_activations(model, sae, texts, device="cuda", max_features=500):
    """Collect SAE feature activations on texts."""
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
    top_indices = sorted(top_indices.tolist())

    filtered_activations = []
    for acts in all_activations:
        filtered_activations.append(acts[:, top_indices])

    return filtered_activations, top_indices


def compute_cooccurrence_matrix(activations, feature_indices):
    """Compute phi coefficient co-occurrence matrix."""
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
    """Run UAD: hierarchical clustering + within-cluster pair detection."""
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

    return candidate_pairs, clusters


def filter_dead_features(candidate_pairs, feature_indices, marginal_freqs, threshold=0.001):
    """Remove pairs involving dead features."""
    alive_features = set()
    for idx, freq in enumerate(marginal_freqs):
        if freq >= threshold:
            alive_features.add(feature_indices[idx])

    filtered = []
    for feat_i, feat_j, phi, clust_id in candidate_pairs:
        if feat_i in alive_features and feat_j in alive_features:
            filtered.append((feat_i, feat_j, phi, clust_id))

    return filtered


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_uad(detected_pairs, ground_truth_pairs, number_primary):
    """Evaluate UAD against ground truth absorption pairs."""
    detected_set = set()
    for feat_i, feat_j, phi, clust_id in detected_pairs:
        detected_set.add(tuple(sorted([feat_i, feat_j])))

    gt_set = set(ground_truth_pairs)

    tp = len(detected_set & gt_set)
    fp = len(detected_set - gt_set)
    fn = len(gt_set - detected_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "detected_pairs": len(detected_set),
        "ground_truth_pairs": len(gt_set),
    }


def compute_random_baseline(n_features, n_gt_pairs, n_detected, n_trials=100):
    """Compute random baseline F1."""
    n_all_pairs = n_features * (n_features - 1) // 2

    f1_scores = []
    for _ in range(n_trials):
        expected_tp = n_detected * n_gt_pairs / n_all_pairs
        expected_fp = n_detected - expected_tp
        expected_fn = n_gt_pairs - expected_tp

        p = expected_tp / (expected_tp + expected_fp) if (expected_tp + expected_fp) > 0 else 0
        r = expected_tp / (expected_tp + expected_fn) if (expected_tp + expected_fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1_scores.append(f1)

    return {
        "mean_f1": float(np.mean(f1_scores)),
        "std_f1": float(np.std(f1_scores)),
        "n_trials": n_trials,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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
    sae = SAE.from_pretrained(
        release=SAE_ID,
        sae_id=f"blocks.{LAYER}.hook_resid_pre",
        device=device,
    )
    print(f"SAE loaded. d_sae: {sae.cfg.d_sae}, d_in: {sae.cfg.d_in}")

    report_progress(2, 5, metric={"stage": "ground_truth"})

    # Build ground truth with number word features
    print("Building ground truth (number word features)...")
    gt = detect_number_features(model, sae, device)
    print(f"Detected {len(gt['number_primary'])} number features")
    print(f"Absorption features: {list(gt['absorption_features'].keys())}")
    print(f"Absorption pairs: {len(gt['absorption_pairs'])}")
    print(f"Feature absorption pairs: {len(gt['feature_pairs'])}")

    report_progress(3, 5, metric={"stage": "loading_dataset"})

    # Load dataset
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

    # Run UAD
    print("\nRunning UAD...")
    print("  Collecting feature activations...")
    activations, feature_indices = collect_feature_activations(
        model, sae, texts, device, max_features=FEATURES_ANALYZED
    )
    print(f"  Analyzing {len(feature_indices)} features")

    print("  Computing co-occurrence matrix...")
    phi_matrix, marginal_freqs, cooccur = compute_cooccurrence_matrix(activations, feature_indices)

    print("  Running hierarchical clustering...")
    candidate_pairs, clusters = uad_detect_absorption(phi_matrix, feature_indices, n_clusters=N_CLUSTERS)
    print(f"  Raw candidate pairs: {len(candidate_pairs)}")

    print("  Filtering dead features...")
    filtered_pairs = filter_dead_features(candidate_pairs, feature_indices, marginal_freqs, threshold=0.001)
    print(f"  After dead feature filtering: {len(filtered_pairs)}")

    # Evaluate
    print("\nEvaluating...")
    results = evaluate_uad(filtered_pairs, gt['feature_pairs'], gt['number_primary'])

    # Random baseline
    baseline = compute_random_baseline(
        len(feature_indices), len(gt['feature_pairs']), len(filtered_pairs)
    )

    # Dead feature stats
    dead_count = sum(1 for f in marginal_freqs if f < 0.001)
    dead_pct = dead_count / len(marginal_freqs) * 100

    # Compile results
    output = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": MODEL_NAME,
            "sae_id": SAE_ID,
            "layer": LAYER,
            "dataset_samples": DATASET_SAMPLES,
            "features_analyzed": FEATURES_ANALYZED,
            "n_clusters": N_CLUSTERS,
            "seed": SEED,
        },
        "ground_truth": {
            "number_features": gt['number_primary'],
            "absorption_features": {str(k): v for k, v in gt['absorption_features'].items()},
            "absorption_pairs": gt['absorption_pairs'],
            "feature_pairs": [list(p) for p in gt['feature_pairs']],
        },
        "uad_results": {
            "detected_pairs": len(filtered_pairs),
            "precision": results["precision"],
            "recall": results["recall"],
            "f1": results["f1"],
            "true_positives": results["true_positives"],
            "false_positives": results["false_positives"],
            "false_negatives": results["false_negatives"],
        },
        "random_baseline": baseline,
        "dead_feature_stats": {
            "dead_count": dead_count,
            "total_features": len(marginal_freqs),
            "dead_percentage": dead_pct,
        },
        "runtime_seconds": time.time() - start_time,
    }

    # Save results
    results_file = RESULTS_DIR / "p2_uad_reproduce_results.json"
    results_file.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("UAD REPRODUCIBILITY RESULTS (v3 - Number Hierarchy)")
    print("=" * 60)
    print(f"Ground truth absorption pairs: {len(gt['feature_pairs'])}")
    print(f"UAD detected pairs: {len(filtered_pairs)}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall:    {results['recall']:.3f}")
    print(f"F1:        {results['f1']:.3f}")
    print(f"Random baseline F1: {baseline['mean_f1']:.3f} (+/- {baseline['std_f1']:.3f})")
    print(f"Dead features: {dead_count}/{len(marginal_freqs)} ({dead_pct:.1f}%)")
    print(f"Runtime: {output['runtime_seconds']:.1f}s")
    print("=" * 60)

    # Pass criteria check
    passed = results["f1"] >= 0.5 and results["recall"] >= 0.8
    print(f"\nPass criteria: F1 >= 0.5 ({results['f1']:.3f}), Recall >= 0.8 ({results['recall']:.3f})")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")

    mark_done(
        status="success" if passed else "partial",
        summary=f"UAD F1={results['f1']:.3f}, Recall={results['recall']:.3f}, "
                f"Detected={len(filtered_pairs)} pairs, Dead={dead_pct:.1f}%"
    )

    return output


if __name__ == "__main__":
    main()
