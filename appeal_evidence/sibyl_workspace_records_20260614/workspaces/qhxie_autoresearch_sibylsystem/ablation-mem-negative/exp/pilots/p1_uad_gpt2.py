#!/usr/bin/env python3
"""
Pilot P1: UAD on GPT-2 Small
Unsupervised Absorption Detection via co-occurrence clustering.
Target: F1 >= 0.5, runtime < 15 min.
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from transformer_lens import HookedTransformer
from sae_lens import SAE

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────
SEED = 42
N_TOKENS = 1000          # pilot sample size
LAYER = 8
TOP_K_FEATURES = 500     # analyze top 500 features by frequency
N_CLUSTERS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TASK_ID = "p1_uad_gpt2"

# ── Process Tracking ────────────────────────────────────────────────
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID, "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps, "loss": loss,
        "metric": metric or {}, "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    pid_file.unlink(missing_ok=True)
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress.exists():
        try:
            final_progress = json.loads(progress.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress, "timestamp": datetime.now().isoformat(),
    }))

# ── Helper: First-letter concept detection ──────────────────────────
def detect_first_letter_features(sae, model, layer, letters="abcdefghijklmnopqrstuvwxyz"):
    """Find features that maximally activate on words starting with each letter.
    Returns dict: letter -> list of (feature_idx, max_activation).
    """
    feature_max = {letter: [] for letter in letters}

    test_words = []
    for letter in letters:
        test_words.extend([
            f"{letter}pple", f"{letter}nt", f"{letter}pril", f"{letter}rt",
            f"{letter}nswer", f"{letter}rea", f"{letter}ble", f"{letter}bout"
        ])

    for word in test_words:
        tokens = model.to_tokens(word)
        _, cache = model.run_with_cache(tokens)
        acts = cache["resid_pre", layer]
        features = sae.encode(acts)
        # Max over batch and position
        max_vals = features[0].max(dim=0).values  # [d_sae]
        first_letter = word[0].lower()
        if first_letter in feature_max:
            for feat_idx in max_vals.topk(20).indices.tolist():
                feature_max[first_letter].append((feat_idx, max_vals[feat_idx].item()))

    # Aggregate: for each letter, find features that consistently top-activate
    letter_features = {}
    for letter in letters:
        feat_scores = {}
        for feat_idx, act in feature_max[letter]:
            feat_scores[feat_idx] = feat_scores.get(feat_idx, 0) + act
        if feat_scores:
            best_feat = max(feat_scores, key=feat_scores.get)
            letter_features[letter] = best_feat

    return letter_features

# ── Helper: Chanin-style collision detection ────────────────────────
def find_collisions(sae, model, layer, letters="abcdefghijklmnopqrstuvwxyz"):
    """Find features that fire for multiple first-letter concepts (absorption indicator)."""
    letter_features = detect_first_letter_features(sae, model, layer, letters)

    # For each feature, count how many letters it represents
    feat_to_letters = {}
    for letter, feat_idx in letter_features.items():
        feat_to_letters.setdefault(feat_idx, []).append(letter)

    # Collisions: features representing 2+ letters
    collisions = {feat: letters for feat, letters in feat_to_letters.items() if len(letters) >= 2}
    return collisions, letter_features

# ── Helper: Build ground-truth parent-child pairs ───────────────────
def build_ground_truth_pairs(letter_features):
    """Build (parent, child) pairs where parent is 'vowel' and children are a,e,i,o,u."""
    vowels = "aeiou"
    pairs = []
    # Find vowel parent feature
    vowel_parent = None
    for v in vowels:
        if v in letter_features:
            vowel_parent = letter_features[v]
            break

    if vowel_parent is not None:
        for v in vowels:
            if v in letter_features and letter_features[v] != vowel_parent:
                pairs.append((vowel_parent, letter_features[v]))

    # Also add collision-based pairs
    return pairs

# ── Main UAD Algorithm ──────────────────────────────────────────────
def run_uad(sae, model, layer, n_tokens=1000, top_k=500, n_clusters=50, seed=42):
    """Run Unsupervised Absorption Detection."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    report_progress(1, 5, step=1, total_steps=5, metric={"phase": "extracting activations"})

    # 1. Extract feature activations on random text
    print("[UAD] Extracting feature activations...")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence research.",
        "Scientists discovered new particles in the Large Hadron Collider.",
        "The economy grew by three percent last quarter.",
        "She walked through the garden and picked some flowers.",
        "Python is a popular programming language for data science.",
        "The movie received critical acclaim at the film festival.",
        "Climate change affects ecosystems worldwide.",
        "Mathematics provides the foundation for modern physics.",
        "The company announced record profits this year.",
    ] * 100

    all_features = []
    for i, text in enumerate(texts[:n_tokens]):
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens)
        acts = cache["resid_pre", layer]
        features = sae.encode(acts)
        # Flatten to (n_positions, d_sae)
        all_features.append(features[0].cpu())
        if i % 100 == 0:
            print(f"  Processed {i}/{n_tokens} texts")

    # Concatenate all features
    all_features = torch.cat(all_features, dim=0)  # (total_positions, d_sae)
    print(f"[UAD] Total feature activations: {all_features.shape}")

    report_progress(2, 5, step=2, total_steps=5, metric={"phase": "computing co-occurrence"})

    # 2. Compute binary activation matrix (features > 0)
    binary = (all_features > 0).float().numpy()  # (N, d_sae)

    # 3. Filter dead features
    feature_freq = binary.mean(axis=0)
    alive_mask = feature_freq > 0.001  # fire on >0.1% of tokens
    alive_indices = np.where(alive_mask)[0]
    print(f"[UAD] Alive features: {len(alive_indices)} / {binary.shape[1]}")

    binary_alive = binary[:, alive_indices]

    # 4. Select top-K features by frequency
    alive_freq = binary_alive.mean(axis=0)
    top_k_indices = np.argsort(alive_freq)[-top_k:]
    binary_topk = binary_alive[:, top_k_indices]
    topk_original_indices = alive_indices[top_k_indices]
    print(f"[UAD] Top-{top_k} features selected")

    report_progress(3, 5, step=3, total_steps=5, metric={"phase": "computing phi correlation"})

    # 5. Compute phi coefficient correlation matrix
    print("[UAD] Computing phi coefficient correlation matrix...")
    n = binary_topk.shape[0]
    # phi(i,j) = (n11*n00 - n10*n01) / sqrt(n1.*n0.*n.*1*n.*0)
    phi_matrix = np.zeros((top_k, top_k))
    for i in range(top_k):
        fi = binary_topk[:, i]
        n1_i = fi.sum()
        n0_i = n - n1_i
        for j in range(i, top_k):
            fj = binary_topk[:, j]
            n11 = (fi * fj).sum()
            n10 = (fi * (1 - fj)).sum()
            n01 = ((1 - fi) * fj).sum()
            n00 = ((1 - fi) * (1 - fj)).sum()

            denom = np.sqrt(n1_i * n0_i * fj.sum() * (n - fj.sum()))
            if denom > 0:
                phi = (n11 * n00 - n10 * n01) / denom
            else:
                phi = 0
            phi_matrix[i, j] = phi
            phi_matrix[j, i] = phi
        if i % 50 == 0:
            print(f"  Computed {i}/{top_k} rows")

    report_progress(4, 5, step=4, total_steps=5, metric={"phase": "clustering"})

    # 6. Hierarchical clustering
    print("[UAD] Running hierarchical clustering...")
    # Convert to distance matrix
    dist_matrix = 1 - np.abs(phi_matrix)
    np.fill_diagonal(dist_matrix, 0)

    # Condense for linkage
    from scipy.spatial.distance import squareform
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="ward")
    clusters = fcluster(Z, t=n_clusters, criterion="maxclust")
    print(f"[UAD] Cluster assignments: {clusters.min()} to {clusters.max()}")

    # 7. Identify parent-child pairs from clusters
    print("[UAD] Identifying parent-child pairs...")
    detected_pairs = []
    for cid in range(1, n_clusters + 1):
        cluster_features = np.where(clusters == cid)[0]
        if len(cluster_features) < 2:
            continue

        # Find highest-frequency feature as "parent" candidate
        cluster_freqs = binary_topk[:, cluster_features].mean(axis=0)
        parent_idx_local = cluster_features[np.argmax(cluster_freqs)]
        parent_idx_global = topk_original_indices[parent_idx_local]

        # Find child candidates: co-occur with parent but lower frequency
        parent_binary = binary_topk[:, parent_idx_local]
        for other_local in cluster_features:
            if other_local == parent_idx_local:
                continue
            other_binary = binary_topk[:, other_local]
            cooccur = (parent_binary * other_binary).sum() / (parent_binary.sum() + 1e-10)
            other_freq = other_binary.mean()
            parent_freq = parent_binary.mean()

            # Absorption signature: high co-occurrence, child fires when parent fires
            if cooccur > 0.5 and other_freq < parent_freq * 1.5:
                child_idx_global = topk_original_indices[other_local]
                detected_pairs.append((
                    int(parent_idx_global), int(child_idx_global),
                    float(cooccur), float(parent_freq), float(other_freq)
                ))

    print(f"[UAD] Detected {len(detected_pairs)} potential absorbed pairs")

    report_progress(5, 5, step=5, total_steps=5, metric={"phase": "evaluation"})

    return detected_pairs, phi_matrix, topk_original_indices, binary_topk

# ── Main ────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    print("=" * 60)
    print("Pilot P1: UAD on GPT-2 Small")
    print("=" * 60)

    # Load model and SAE
    print("\n[Setup] Loading model and SAE...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{LAYER}.hook_resid_pre",
        device=DEVICE
    )
    print(f"[Setup] Model: GPT-2 Small, Layer: {LAYER}")
    print(f"[Setup] SAE: d_sae={sae.cfg.d_sae}")

    # Run UAD
    detected_pairs, phi_matrix, topk_indices, binary_topk = run_uad(
        sae, model, LAYER, n_tokens=N_TOKENS, top_k=TOP_K_FEATURES,
        n_clusters=N_CLUSTERS, seed=SEED
    )

    # Build ground truth (Chanin-style first-letter collisions)
    print("\n[Eval] Building ground truth from first-letter features...")
    collisions, letter_features = find_collisions(sae, model, LAYER)
    print(f"[Eval] Found {len(collisions)} collision features")
    for feat, letters in collisions.items():
        print(f"  Feature {feat}: letters {letters}")

    # Build ground-truth pairs
    gt_pairs = set()
    for feat, letters in collisions.items():
        if len(letters) >= 2:
            # Each pair of letters sharing a feature is a collision
            for i in range(len(letters)):
                for j in range(i + 1, len(letters)):
                    li, lj = letters[i], letters[j]
                    if li in letter_features and lj in letter_features:
                        gt_pairs.add((letter_features[li], letter_features[lj]))
                        gt_pairs.add((letter_features[lj], letter_features[li]))

    print(f"[Eval] Ground-truth pairs: {len(gt_pairs)}")

    # Evaluate
    detected_set = set((p, c) for p, c, _, _, _ in detected_pairs)

    tp = len(detected_set & gt_pairs)
    fp = len(detected_set - gt_pairs)
    fn = len(gt_pairs - detected_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Detected pairs: {len(detected_pairs)}")
    print(f"Ground-truth pairs: {len(gt_pairs)}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    runtime = time.time() - start_time
    print(f"Runtime: {runtime:.1f}s")

    # Save results
    results = {
        "task_id": TASK_ID,
        "model": "gpt2-small",
        "layer": LAYER,
        "n_tokens": N_TOKENS,
        "top_k_features": TOP_K_FEATURES,
        "n_clusters": N_CLUSTERS,
        "seed": SEED,
        "detected_pairs": detected_pairs,
        "ground_truth_pairs": list(gt_pairs),
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "letter_features": {k: int(v) for k, v in letter_features.items()},
        "collisions": {int(k): v for k, v in collisions.items()},
        "runtime_seconds": runtime,
        "timestamp": datetime.now().isoformat(),
    }

    output_path = RESULTS_DIR / f"{TASK_ID}_results.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n[Save] Results saved to {output_path}")

    # Pass/fail
    passed = f1 >= 0.5 and runtime < 900
    status = "PASS" if passed else "FAIL"
    print(f"\n[{'PASS' if passed else 'FAIL'}] Pilot P1: F1={f1:.4f} (target >= 0.5), Runtime={runtime:.1f}s (target < 900s)")

    mark_done(status="success" if passed else "failure",
              summary=f"UAD pilot: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, Runtime={runtime:.1f}s")

    return results

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        mark_done(status="failure", summary=f"Error: {str(e)}")
        raise
