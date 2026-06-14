#!/usr/bin/env python3
"""
Pilot P2: UAD Reproducibility Validation
Reproduce UAD results on GPT-2 Small layer 8 with pre-trained SAE.
Compare against first-letter ground truth (a-z, vowel/consonant hierarchy).
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
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

# SAELens imports
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
# Ground truth: first-letter features
# ---------------------------------------------------------------------------
def build_ground_truth_letter_features(model, sae, texts, device="cuda"):
    """
    Detect SAE features that respond to first letters (a-z).
    Returns: dict mapping letter -> list of (feature_idx, mean_activation)
    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    letter_activations = {letter: [] for letter in letters}

    model.eval()
    with torch.no_grad():
        for text in tqdm(texts, desc="Scanning first-letter features"):
            tokens = model.to_tokens(text)
            if tokens.shape[1] < 2:
                continue

            # Get first letter of each token (after BOS)
            token_strs = [model.tokenizer.decode([t]) for t in tokens[0, 1:]]
            first_letters = [s[0].lower() if s else "" for s in token_strs]

            # Run model and get SAE activations
            _, cache = model.run_with_cache(tokens, names_filter=[HOOK_NAME])
            resid = cache[HOOK_NAME]  # [1, seq_len, d_model]

            # Encode with SAE
            sae_acts = sae.encode(resid.squeeze(0))  # [seq_len, d_sae]

            # For each position (skip BOS), record activation for its first letter
            for pos, letter in enumerate(first_letters, start=1):
                if letter in letters and pos < sae_acts.shape[0]:
                    letter_activations[letter].append(sae_acts[pos].cpu().numpy())

    # Find top feature for each letter
    letter_top_features = {}
    for letter in letters:
        if len(letter_activations[letter]) == 0:
            continue
        acts = np.stack(letter_activations[letter])  # [n_samples, d_sae]
        mean_acts = acts.mean(axis=0)
        top_idx = int(np.argmax(mean_acts))
        letter_top_features[letter] = (top_idx, float(mean_acts[top_idx]))

    return letter_top_features


def build_absorption_pairs(letter_top_features):
    """
    Build ground truth absorption pairs based on vowel/consonant hierarchy.
    Parent = vowel concept (a, e, i, o, u share a parent feature).
    Child = individual letter features.
    Absorption = a single SAE feature responds to multiple vowels.
    """
    vowels = set("aeiou")
    consonants = set("abcdefghijklmnopqrstuvwxyz") - vowels

    # Find features that respond to multiple vowels
    vowel_features = {}
    for v in vowels:
        if v in letter_top_features:
            feat_idx = letter_top_features[v][0]
            if feat_idx not in vowel_features:
                vowel_features[feat_idx] = []
            vowel_features[feat_idx].append(v)

    # Absorption pairs: (parent_letter, child_letter) where parent and child
    # share the same SAE feature (the feature "absorbs" multiple concepts)
    absorption_pairs = []
    for feat_idx, letters in vowel_features.items():
        if len(letters) > 1:
            # Each pair of letters sharing a feature is an absorption pair
            for i in range(len(letters)):
                for j in range(i + 1, len(letters)):
                    absorption_pairs.append((letters[i], letters[j]))

    # Also: if a consonant feature responds to multiple consonants
    consonant_features = {}
    for c in consonants:
        if c in letter_top_features:
            feat_idx = letter_top_features[c][0]
            if feat_idx not in consonant_features:
                consonant_features[feat_idx] = []
            consonant_features[feat_idx].append(c)

    for feat_idx, letters in consonant_features.items():
        if len(letters) > 1:
            for i in range(len(letters)):
                for j in range(i + 1, len(letters)):
                    absorption_pairs.append((letters[i], letters[j]))

    return absorption_pairs


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
            resid = cache[HOOK_NAME].squeeze(0)  # [seq_len, d_model]
            sae_acts = sae.encode(resid)  # [seq_len, d_sae]

            # Convert to binary activation (feature fires if activation > 0)
            binary_acts = (sae_acts > 0).float().cpu().numpy()  # [seq_len, d_sae]
            all_activations.append(binary_acts)

            if feature_counts is None:
                feature_counts = binary_acts.sum(axis=0)
            else:
                feature_counts += binary_acts.sum(axis=0)

    # Find most active features
    if feature_counts is None:
        return None, None

    top_indices = np.argsort(feature_counts)[-max_features:]
    top_indices = sorted(top_indices.tolist())

    # Extract activations for top features only
    filtered_activations = []
    for acts in all_activations:
        filtered_activations.append(acts[:, top_indices])

    return filtered_activations, top_indices


def compute_cooccurrence_matrix(activations, feature_indices):
    """Compute phi coefficient (Yule's phi) co-occurrence matrix."""
    n_features = len(feature_indices)
    cooccur = np.zeros((n_features, n_features))
    marginal = np.zeros(n_features)
    total = 0

    for acts in activations:
        # acts: [seq_len, n_features]
        for pos in range(acts.shape[0]):
            feat_vec = acts[pos]
            cooccur += np.outer(feat_vec, feat_vec)
            marginal += feat_vec
            total += 1

    # Compute phi coefficient
    phi_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            a = cooccur[i, j]  # both active
            b = marginal[i] - cooccur[i, j]  # i active, j not
            c = marginal[j] - cooccur[i, j]  # j active, i not
            d = total - marginal[i] - marginal[j] + cooccur[i, j]  # neither

            denom = np.sqrt((a + b) * (a + c) * (b + d) * (c + d))
            if denom > 0:
                phi_matrix[i, j] = (a * d - b * c) / denom
            else:
                phi_matrix[i, j] = 0.0

    return phi_matrix, marginal / total, cooccur


def uad_detect_absorption(phi_matrix, feature_indices, n_clusters=50):
    """Run UAD: hierarchical clustering + within-cluster pair detection."""
    # Distance = 1 - phi
    dist_matrix = 1 - np.abs(phi_matrix)
    np.fill_diagonal(dist_matrix, 0)

    # Ensure symmetry
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    # Hierarchical clustering
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="ward")
    clusters = fcluster(Z, n_clusters, criterion="maxclust")

    # Find pairs within same cluster
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
    """Remove pairs involving dead features (very low activation frequency)."""
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
def evaluate_uad(detected_pairs, ground_truth_pairs, all_letter_features):
    """Evaluate UAD against ground truth absorption pairs."""
    # Convert to sets of tuples (sorted)
    detected_set = set()
    for feat_i, feat_j, phi, clust_id in detected_pairs:
        detected_set.add(tuple(sorted([feat_i, feat_j])))

    gt_set = set()
    for l1, l2 in ground_truth_pairs:
        if l1 in all_letter_features and l2 in all_letter_features:
            feat1 = all_letter_features[l1][0]
            feat2 = all_letter_features[l2][0]
            gt_set.add(tuple(sorted([feat1, feat2])))

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


def compute_random_baseline(all_feature_indices, ground_truth_pairs, n_detected, n_trials=100):
    """Compute random baseline F1."""
    gt_set = set()
    for l1, l2 in ground_truth_pairs:
        # We need the actual feature indices - this is tricky without the letter mapping
        # For now, compute a simple random expectation
        pass

    # Simplified: random selection from all possible pairs
    n_features = len(all_feature_indices)
    n_all_pairs = n_features * (n_features - 1) // 2
    n_gt = len(ground_truth_pairs)

    f1_scores = []
    for _ in range(n_trials):
        # Randomly select n_detected pairs
        selected = np.random.choice(n_all_pairs, size=min(n_detected, n_all_pairs), replace=False)
        # Expected overlap
        expected_tp = n_detected * n_gt / n_all_pairs
        expected_fp = n_detected - expected_tp
        expected_fn = n_gt - expected_tp

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

    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    report_progress(0, 5, metric={"stage": "loading_model"})

    # Load model
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    print(f"Model loaded. Layers: {model.cfg.n_layers}")

    report_progress(1, 5, metric={"stage": "loading_sae"})

    # Load SAE
    print(f"Loading SAE: {SAE_ID}, layer {LAYER}...")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=SAE_ID,
        sae_id=f"blocks.{LAYER}.hook_resid_pre",
        device=device,
    )
    print(f"SAE loaded. d_sae: {sae.cfg.d_sae}, d_in: {sae.cfg.d_in}")

    report_progress(2, 5, metric={"stage": "loading_dataset"})

    # Load dataset
    print("Loading OpenWebText samples...")
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    ds = ds.shuffle(seed=SEED)
    texts = []
    for i, example in enumerate(ds):
        if i >= DATASET_SAMPLES:
            break
        texts.append(example["text"])
    print(f"Loaded {len(texts)} texts")

    report_progress(3, 5, metric={"stage": "ground_truth"})

    # Build ground truth
    print("Building ground truth (first-letter features)...")
    letter_top_features = build_ground_truth_letter_features(model, sae, texts[:100], device)
    print(f"Detected {len(letter_top_features)} letter features")

    absorption_pairs = build_absorption_pairs(letter_top_features)
    print(f"Ground truth absorption pairs: {len(absorption_pairs)}")
    if absorption_pairs:
        print(f"Pairs: {absorption_pairs[:10]}")

    report_progress(4, 5, metric={"stage": "uad_analysis"})

    # Run UAD
    print("Running UAD...")
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
    print("Evaluating...")
    results = evaluate_uad(filtered_pairs, absorption_pairs, letter_top_features)

    # Random baseline
    baseline = compute_random_baseline(feature_indices, absorption_pairs, len(filtered_pairs))

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
            "letter_features_detected": len(letter_top_features),
            "absorption_pairs": len(absorption_pairs),
            "absorption_pairs_list": absorption_pairs,
            "letter_top_features": {k: {"feature_idx": v[0], "mean_activation": v[1]}
                                     for k, v in letter_top_features.items()},
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
    print("UAD REPRODUCIBILITY RESULTS")
    print("=" * 60)
    print(f"Ground truth absorption pairs: {len(absorption_pairs)}")
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
