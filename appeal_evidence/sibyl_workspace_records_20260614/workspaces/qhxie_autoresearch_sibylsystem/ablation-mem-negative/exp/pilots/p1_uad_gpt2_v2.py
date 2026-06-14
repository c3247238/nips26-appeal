#!/usr/bin/env python3
"""
Pilot P1 v2: UAD on GPT-2 Small
Improved ground truth using direct feature activation analysis.
Target: F1 >= 0.5, runtime < 15 min.
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import precision_recall_fscore_support
from transformer_lens import HookedTransformer
from sae_lens import SAE

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────
SEED = 42
N_TEXTS = 200
LAYER = 8
TOP_K_FEATURES = 500
N_CLUSTERS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TASK_ID = "p1_uad_gpt2"

# ── Process Tracking ────────────────────────────────────────────────
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

def report_progress(epoch, total_epochs, step=0, total_steps=0, metric=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID, "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
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

# ── Improved Ground Truth: Feature-level first-letter detection ─────
def find_first_letter_features(sae, model, layer, letters="abcdefghijklmnopqrstuvwxyz", n_samples=50):
    """
    For each letter, find the feature that MOST selectively activates on words starting with that letter.
    Uses contrastive scoring: activation on target letter words minus activation on other letter words.
    """
    print("[GT] Finding first-letter features...")

    # Generate test words for each letter
    letter_words = {}
    common_words = {
        'a': ['apple', 'ant', 'arrow', 'artist', 'area', 'able', 'about', 'above'],
        'b': ['ball', 'bat', 'bird', 'book', 'blue', 'baby', 'back', 'bank'],
        'c': ['cat', 'car', 'cup', 'city', 'cold', 'call', 'care', 'case'],
        'd': ['dog', 'day', 'door', 'dark', 'deep', 'data', 'date', 'deal'],
        'e': ['egg', 'end', 'east', 'easy', 'each', 'earn', 'edge', 'edit'],
        'f': ['fish', 'fire', 'food', 'fast', 'face', 'fact', 'fail', 'fair'],
        'g': ['goat', 'game', 'good', 'gold', 'gain', 'garden', 'gas', 'gate'],
        'h': ['hat', 'hand', 'home', 'high', 'hair', 'half', 'hall', 'hard'],
        'i': ['ice', 'idea', 'iron', 'item', 'icon', 'image', 'inch', 'input'],
        'j': ['jump', 'job', 'join', 'just', 'jacket', 'jail', 'jazz', 'jet'],
        'k': ['kite', 'key', 'king', 'kind', 'keep', 'kick', 'kill', 'kind'],
        'l': ['lion', 'lake', 'long', 'last', 'lack', 'lady', 'land', 'late'],
        'm': ['moon', 'man', 'map', 'make', 'mail', 'main', 'major', 'make'],
        'n': ['nest', 'name', 'new', 'next', 'nail', 'naked', 'narrow', 'nation'],
        'o': ['owl', 'open', 'old', 'only', 'object', 'obey', 'ocean', 'offer'],
        'p': ['pig', 'pen', 'play', 'past', 'pack', 'page', 'pain', 'paint'],
        'q': ['queen', 'quick', 'quiet', 'quit', 'quality', 'quarter', 'quest', 'question'],
        'r': ['rat', 'red', 'run', 'rest', 'race', 'radio', 'rain', 'raise'],
        's': ['sun', 'sit', 'sea', 'same', 'safe', 'sad', 'sale', 'salt'],
        't': ['top', 'time', 'tree', 'test', 'table', 'tail', 'take', 'talk'],
        'u': ['up', 'use', 'unit', 'upon', 'ugly', 'uncle', 'under', 'union'],
        'v': ['van', 'very', 'view', 'vote', 'vacation', 'valley', 'value', 'various'],
        'w': ['wet', 'water', 'way', 'wait', 'wake', 'walk', 'wall', 'want'],
        'x': ['xray', 'xenon', 'xerox', 'xylophone'],
        'y': ['yes', 'year', 'young', 'your', 'yard', 'yellow', 'yield', 'you'],
        'z': ['zoo', 'zero', 'zone', 'zebra', 'zip', 'zoom', 'zinc', 'zombie'],
    }

    # Collect activations per letter
    letter_activations = defaultdict(list)
    all_activations = []

    for letter in letters:
        words = common_words.get(letter, [f"{letter}word{i}" for i in range(8)])
        for word in words[:n_samples]:
            try:
                tokens = model.to_tokens(word)
                _, cache = model.run_with_cache(tokens)
                acts = cache["resid_pre", layer]
                features = sae.encode(acts)
                # Max activation across positions for this word
                max_feat = features[0].max(dim=0).values.cpu().numpy()
                letter_activations[letter].append(max_feat)
                all_activations.append(max_feat)
            except Exception as e:
                continue

    if not all_activations:
        print("[GT] ERROR: No activations collected")
        return {}

    all_activations = np.array(all_activations)
    global_mean = all_activations.mean(axis=0)
    global_std = all_activations.std(axis=0) + 1e-8

    letter_features = {}
    for letter in letters:
        if not letter_activations[letter]:
            continue
        acts = np.array(letter_activations[letter])
        letter_mean = acts.mean(axis=0)

        # Contrastive score: how much more does this feature fire for this letter vs global mean
        contrastive = (letter_mean - global_mean) / global_std

        # Pick top feature
        best_feat = int(np.argmax(contrastive))
        best_score = float(contrastive[best_feat])

        if best_score > 0.5:  # threshold for meaningful selectivity
            letter_features[letter] = {"feature": best_feat, "score": best_score}
            print(f"  Letter '{letter}': feature {best_feat} (score={best_score:.3f})")

    return letter_features

# ── Build ground-truth absorption pairs ─────────────────────────────
def build_gt_pairs(letter_features):
    """Build ground-truth (parent, child) pairs based on shared features."""
    # Invert: feature -> letters
    feat_to_letters = defaultdict(list)
    for letter, info in letter_features.items():
        feat_to_letters[info["feature"]].append(letter)

    # Collision pairs: when two letters share the same feature
    gt_pairs = set()
    for feat, letters in feat_to_letters.items():
        if len(letters) >= 2:
            for i in range(len(letters)):
                for j in range(i + 1, len(letters)):
                    li, lj = letters[i], letters[j]
                    # Both directions
                    gt_pairs.add((letter_features[li]["feature"], letter_features[lj]["feature"]))
                    gt_pairs.add((letter_features[lj]["feature"], letter_features[li]["feature"]))

    return gt_pairs, feat_to_letters

# ── UAD Algorithm ───────────────────────────────────────────────────
def run_uad(sae, model, layer, n_texts=200, top_k=500, n_clusters=50, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    report_progress(1, 5, step=1, total_steps=5, metric={"phase": "extracting activations"})

    # Use diverse texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning transforms artificial intelligence research daily.",
        "Scientists discovered new particles in the Large Hadron Collider experiment.",
        "The economy grew by three percent last quarter despite challenges.",
        "She walked through the beautiful garden and picked some fresh flowers.",
        "Python is a popular programming language for data science and AI.",
        "The movie received critical acclaim at the international film festival.",
        "Climate change affects ecosystems and biodiversity worldwide significantly.",
        "Mathematics provides the foundation for modern physics and engineering.",
        "The company announced record profits this year exceeding all expectations.",
        "Education is the most powerful weapon which you can use to change the world.",
        "The artist painted a stunning portrait of the mountain landscape at sunset.",
        "Researchers developed a new algorithm for natural language processing tasks.",
        "The spacecraft launched successfully and reached orbit within minutes.",
        "Music has the power to heal and bring people together across cultures.",
        "The ancient ruins revealed secrets about civilization thousands of years ago.",
        "Doctors recommend regular exercise for maintaining good physical health.",
        "The chef prepared a delicious meal using locally sourced organic ingredients.",
        "Technology continues to evolve at an unprecedented pace in modern society.",
        "The athlete trained for months before winning the championship gold medal.",
    ] * 10

    all_features = []
    for i, text in enumerate(texts[:n_texts]):
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens)
        acts = cache["resid_pre", layer]
        features = sae.encode(acts)
        all_features.append(features[0].cpu())

    all_features = torch.cat(all_features, dim=0)
    print(f"[UAD] Total positions: {all_features.shape[0]}, d_sae: {all_features.shape[1]}")

    report_progress(2, 5, step=2, total_steps=5, metric={"phase": "filtering features"})

    # Binary activation
    binary = (all_features > 0).float().numpy()

    # Filter dead features
    feature_freq = binary.mean(axis=0)
    alive_mask = feature_freq > 0.001
    alive_indices = np.where(alive_mask)[0]
    print(f"[UAD] Alive features: {len(alive_indices)} / {binary.shape[1]}")

    binary_alive = binary[:, alive_indices]
    alive_freq = binary_alive.mean(axis=0)

    # Select top-K by frequency
    top_k_local = min(top_k, len(alive_indices))
    top_k_indices_local = np.argsort(alive_freq)[-top_k_local:]
    binary_topk = binary_alive[:, top_k_indices_local]
    topk_global_indices = alive_indices[top_k_indices_local]
    print(f"[UAD] Top-{top_k_local} features selected")

    report_progress(3, 5, step=3, total_steps=5, metric={"phase": "computing phi correlation"})

    # Compute phi coefficient
    print("[UAD] Computing phi coefficient matrix...")
    n = binary_topk.shape[0]
    phi_matrix = np.zeros((top_k_local, top_k_local))

    for i in range(top_k_local):
        fi = binary_topk[:, i]
        n1_i = fi.sum()
        n0_i = n - n1_i
        if n1_i == 0 or n0_i == 0:
            continue
        for j in range(i, top_k_local):
            fj = binary_topk[:, j]
            n11 = (fi * fj).sum()
            n10 = (fi * (1 - fj)).sum()
            n01 = ((1 - fi) * fj).sum()
            n00 = ((1 - fi) * (1 - fj)).sum()
            denom = np.sqrt(n1_i * n0_i * fj.sum() * (n - fj.sum()))
            if denom > 0:
                phi = (n11 * n00 - n10 * n01) / denom
                phi_matrix[i, j] = phi
                phi_matrix[j, i] = phi

    report_progress(4, 5, step=4, total_steps=5, metric={"phase": "clustering"})

    # Clustering
    print("[UAD] Clustering...")
    dist_matrix = 1 - np.abs(phi_matrix)
    np.fill_diagonal(dist_matrix, 0)
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="ward")
    clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Identify pairs
    print("[UAD] Identifying parent-child pairs...")
    detected_pairs = []
    for cid in range(1, n_clusters + 1):
        cluster_features = np.where(clusters == cid)[0]
        if len(cluster_features) < 2:
            continue

        cluster_freqs = binary_topk[:, cluster_features].mean(axis=0)
        sorted_by_freq = sorted(
            [(cf, cluster_freqs[idx]) for idx, cf in enumerate(cluster_features)],
            key=lambda x: x[1], reverse=True
        )

        # Parent = highest frequency
        parent_local = sorted_by_freq[0][0]
        parent_global = int(topk_global_indices[parent_local])
        parent_binary = binary_topk[:, parent_local]
        parent_freq = parent_binary.mean()

        # Children = others with high conditional probability
        for child_local, child_freq_val in sorted_by_freq[1:]:
            child_binary = binary_topk[:, child_local]
            # P(parent | child) = how often parent fires when child fires
            cond_prob = (parent_binary * child_binary).sum() / (child_binary.sum() + 1e-10)
            # P(child | parent)
            cond_prob_rev = (parent_binary * child_binary).sum() / (parent_binary.sum() + 1e-10)

            # Absorption signature: high mutual co-occurrence, similar frequencies
            if cond_prob > 0.3 and cond_prob_rev > 0.3:
                child_global = int(topk_global_indices[child_local])
                detected_pairs.append((
                    parent_global, child_global,
                    float(cond_prob), float(cond_prob_rev), float(parent_freq), float(child_freq_val)
                ))

    print(f"[UAD] Detected {len(detected_pairs)} pairs")
    report_progress(5, 5, step=5, total_steps=5, metric={"phase": "complete", "n_pairs": len(detected_pairs)})

    return detected_pairs, phi_matrix, topk_global_indices, binary_topk

# ── Main ────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    print("=" * 60)
    print("Pilot P1 v2: UAD on GPT-2 Small")
    print("=" * 60)

    print("\n[Setup] Loading model and SAE...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{LAYER}.hook_resid_pre",
        device=DEVICE
    )
    print(f"[Setup] Model: GPT-2 Small, Layer {LAYER}, d_sae={sae.cfg.d_sae}")

    # Run UAD
    detected_pairs, phi_matrix, topk_indices, binary_topk = run_uad(
        sae, model, LAYER, n_texts=N_TEXTS, top_k=TOP_K_FEATURES,
        n_clusters=N_CLUSTERS, seed=SEED
    )

    # Build ground truth
    print("\n[GT] Building ground truth...")
    letter_features = find_first_letter_features(sae, model, LAYER)
    gt_pairs, feat_to_letters = build_gt_pairs(letter_features)
    print(f"[GT] Found {len(letter_features)} letter features, {len(gt_pairs)} GT pairs")
    for feat, letters in feat_to_letters.items():
        if len(letters) >= 2:
            print(f"  Collision: feature {feat} -> letters {letters}")

    # Evaluate
    detected_set = set((p, c) for p, c, _, _, _, _ in detected_pairs)

    tp = len(detected_set & gt_pairs)
    fp = len(detected_set - gt_pairs)
    fn = len(gt_pairs - detected_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Also compute AUC-ROC against random baseline
    n_random_trials = 100
    random_f1s = []
    for _ in range(n_random_trials):
        random_pairs = set()
        for _ in range(len(detected_pairs)):
            p = np.random.choice(topk_indices)
            c = np.random.choice(topk_indices)
            if p != c:
                random_pairs.add((int(p), int(c)))
        rtp = len(random_pairs & gt_pairs)
        rfp = len(random_pairs - gt_pairs)
        rfn = len(gt_pairs - random_pairs)
        rp = rtp / (rtp + rfp) if (rtp + rfp) > 0 else 0
        rr = rtp / (rtp + rfn) if (rtp + rfn) > 0 else 0
        rf1 = 2 * rp * rr / (rp + rr) if (rp + rr) > 0 else 0
        random_f1s.append(rf1)

    mean_random_f1 = np.mean(random_f1s)
    auc_proxy = f1 / (f1 + mean_random_f1) if (f1 + mean_random_f1) > 0 else 0

    runtime = time.time() - start_time

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
    print(f"Random baseline F1: {mean_random_f1:.4f}")
    print(f"Runtime: {runtime:.1f}s")

    # Save results
    results = {
        "task_id": TASK_ID,
        "model": "gpt2-small",
        "layer": LAYER,
        "n_texts": N_TEXTS,
        "top_k_features": TOP_K_FEATURES,
        "n_clusters": N_CLUSTERS,
        "seed": SEED,
        "detected_pairs": detected_pairs,
        "ground_truth_pairs": list(gt_pairs),
        "letter_features": letter_features,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "random_baseline_f1": mean_random_f1,
            "auc_proxy": auc_proxy,
        },
        "runtime_seconds": runtime,
        "timestamp": datetime.now().isoformat(),
    }

    output_path = RESULTS_DIR / f"{TASK_ID}_results.json"
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[Save] Results saved to {output_path}")

    passed = f1 >= 0.5 and runtime < 900
    print(f"\n[{'PASS' if passed else 'FAIL'}] Pilot P1: F1={f1:.4f} (target >= 0.5), Runtime={runtime:.1f}s")

    mark_done(status="success" if passed else "failure",
              summary=f"UAD pilot v2: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, Runtime={runtime:.1f}s")

    return results

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        mark_done(status="failure", summary=f"Error: {str(e)}")
        raise
