#!/usr/bin/env python3
"""
Pilot P3: UAD Feasibility
Build co-occurrence matrix for features from GPT-2 SAE.
Run hierarchical clustering to identify potential absorption pairs.
Compare with supervised labels on known cases (first-letter features).
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import SAE
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# ── Configuration ──────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
RESULTS_DIR = Path(__file__).parent.parent / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TASK_ID = "p3_uad"

pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except:
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


# ── Set seeds ──────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ── Helper: build co-occurrence matrix ─────────────────────────────────────
def build_cooccurrence_matrix(model, sae, tokenizer, device, n_samples=500):
    """Build feature activation co-occurrence matrix."""
    # Generate diverse prompts
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence.",
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Earth revolves around the Sun in 365 days.",
        "Photosynthesis converts light energy into chemical energy.",
        "The Great Wall of China is one of the seven wonders.",
        "DNA contains the genetic instructions for all living organisms.",
        "The speed of light is approximately 300,000 kilometers per second.",
        "Quantum mechanics describes the behavior of matter at small scales.",
    ] * 50

    tokenized = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=64
    )
    input_ids = tokenized["input_ids"].to(device)

    hook_name = getattr(sae.cfg, 'hook_name', 'blocks.8.hook_resid_pre')
    with torch.no_grad():
        _, cache = model.run_with_cache(
            input_ids, names_filter=[hook_name]
        )
        acts = cache[hook_name]
        sae_acts = sae.encode(acts)  # [batch, pos, d_sae]

        # Flatten across batch and position
        flat_acts = sae_acts.reshape(-1, sae_acts.shape[-1])  # [N, d_sae]

        # Binary activation (feature is active or not)
        binary_acts = (flat_acts > 0).float().cpu().numpy()

    # Compute co-occurrence: how often do features i and j activate together?
    # Use a subset of features for efficiency (most active ones)
    feature_activity = binary_acts.sum(axis=0)
    top_k = min(100, len(feature_activity))
    top_indices = np.argsort(feature_activity)[-top_k:]

    subset_acts = binary_acts[:, top_indices]
    cooccur = subset_acts.T @ subset_acts  # [top_k, top_k]

    # Normalize to get co-occurrence rate
    feature_counts = subset_acts.sum(axis=0, keepdims=True).T
    cooccur_norm = cooccur / (feature_counts + 1e-8)

    return cooccur_norm, top_indices, feature_activity


# ── Helper: hierarchical clustering ────────────────────────────────────────
def cluster_features(cooccur_matrix, top_indices, n_clusters=10):
    """Apply hierarchical clustering to identify feature groups."""
    # Convert co-occurrence to distance (1 - normalized co-occurrence)
    # Use Jaccard distance for binary features
    dist_matrix = 1 - cooccur_matrix
    np.fill_diagonal(dist_matrix, 0)

    # Ensure symmetry and valid distances
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    dist_matrix = np.clip(dist_matrix, 0, 1)

    # Convert to condensed distance matrix
    condensed = squareform(dist_matrix, checks=False)

    # Hierarchical clustering
    Z = linkage(condensed, method='ward')
    clusters = fcluster(Z, n_clusters, criterion='maxclust')

    return clusters, Z


# ── Helper: supervised labels (first-letter features) ──────────────────────
def get_supervised_labels(model, sae, tokenizer, device, top_indices):
    """Get supervised labels for first-letter features."""
    letters = [chr(ord('a') + i) for i in range(26)]
    prompts = []
    labels = []
    for letter in letters:
        words = [f"{letter}{w}" for w in [
            "pple", "nimal", "lpha", "rt"
        ]]
        for word in words:
            prompts.append(f"The word '{word}'")
            labels.append(ord(letter) - ord('a'))

    tokenized = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=32
    )
    input_ids = tokenized["input_ids"].to(device)

    hook_name = getattr(sae.cfg, 'hook_name', 'blocks.8.hook_resid_pre')
    with torch.no_grad():
        _, cache = model.run_with_cache(
            input_ids, names_filter=[hook_name]
        )
        acts = cache[hook_name]
        sae_acts = sae.encode(acts)
        sample_acts = sae_acts.max(dim=1).values  # [batch, d_sae]

    labels_t = torch.tensor(labels, device=device)
    supervised_features = {}
    for letter_idx in range(26):
        mask = labels_t == letter_idx
        if mask.sum() == 0:
            continue
        letter_acts = sample_acts[mask].mean(dim=0)
        other_acts = sample_acts[~mask].mean(dim=0)
        scores = letter_acts - other_acts

        # Find best feature among top_indices
        best_score = -float('inf')
        best_feat = None
        for idx in top_indices:
            if scores[idx] > best_score:
                best_score = scores[idx].item()
                best_feat = idx

        if best_feat is not None:
            supervised_features[letter_idx] = best_feat

    return supervised_features


# ── Helper: evaluate UAD against supervised labels ─────────────────────────
def evaluate_uad(clusters, top_indices, supervised_labels):
    """Evaluate clustering against supervised labels."""
    # For each pair of letters that share a cluster, check if they also
    # share features in the supervised labels (potential absorption)

    # Build cluster membership
    cluster_map = {top_indices[i]: int(clusters[i]) for i in range(len(top_indices))}

    # Check which supervised features are in the same cluster
    same_cluster_pairs = []
    different_cluster_pairs = []

    letter_list = list(supervised_labels.keys())
    for i in range(len(letter_list)):
        for j in range(i + 1, len(letter_list)):
            li, lj = letter_list[i], letter_list[j]
            fi, fj = supervised_labels[li], supervised_labels[lj]

            ci = cluster_map.get(fi, -1)
            cj = cluster_map.get(fj, -1)

            if ci == cj and ci != -1:
                same_cluster_pairs.append((li, lj))
            else:
                different_cluster_pairs.append((li, lj))

    # Check collision in supervised labels
    supervised_collisions = []
    feat_to_letters = {}
    for letter, feat in supervised_labels.items():
        if feat not in feat_to_letters:
            feat_to_letters[feat] = []
        feat_to_letters[feat].append(letter)

    for feat, letters in feat_to_letters.items():
        if len(letters) > 1:
            for i in range(len(letters)):
                for j in range(i + 1, len(letters)):
                    supervised_collisions.append((letters[i], letters[j]))

    # Precision: of pairs in same cluster, how many are actual collisions?
    if same_cluster_pairs:
        true_positives = sum(1 for p in same_cluster_pairs if p in supervised_collisions)
        precision = true_positives / len(same_cluster_pairs)
    else:
        precision = 0.0
        true_positives = 0

    # Recall: of actual collisions, how many are in same cluster?
    if supervised_collisions:
        recall = sum(1 for p in supervised_collisions if p in same_cluster_pairs) / len(supervised_collisions)
    else:
        recall = 0.0

    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "same_cluster_pairs": len(same_cluster_pairs),
        "supervised_collisions": len(supervised_collisions),
        "cluster_map": {str(k): v for k, v in cluster_map.items()},
    }


# ── Main experiment ────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    report_progress(0, 4, step=1, total_steps=4, metric={"phase": "init"})

    results = {
        "task_id": TASK_ID,
        "device": DEVICE,
        "gpu_id": GPU_ID,
        "seed": SEED,
        "start_time": datetime.now().isoformat(),
    }

    # ── Step 1: Load GPT-2 Small and SAE ───────────────────────────────────
    print("[1/4] Loading GPT-2 Small and SAE...")
    report_progress(1, 4, step=1, total_steps=4, metric={"phase": "load_model"})

    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    try:
        sae = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id="blocks.8.hook_resid_pre",
            device=DEVICE,
        )
        print("  Loaded pre-trained GPT-2 SAE")
        results["sae_source"] = "gpt2-small-res-jb"
        results["d_sae"] = sae.cfg.d_sae
    except Exception as e:
        print(f"  Could not load pre-trained SAE: {e}")
        mark_done(status="failed", summary=f"SAE loading failed: {e}")
        return

    # ── Step 2: Build co-occurrence matrix ─────────────────────────────────
    print("[2/4] Building co-occurrence matrix...")
    report_progress(2, 4, step=2, total_steps=4, metric={"phase": "cooccurrence"})

    cooccur, top_indices, feature_activity = build_cooccurrence_matrix(
        model, sae, tokenizer, DEVICE, n_samples=500
    )
    print(f"  Co-occurrence matrix: {cooccur.shape}")
    print(f"  Top {len(top_indices)} most active features selected")
    results["top_indices"] = top_indices.tolist()
    results["feature_activity"] = feature_activity.tolist()

    # ── Step 3: Hierarchical clustering ────────────────────────────────────
    print("[3/4] Running hierarchical clustering...")
    report_progress(3, 4, step=3, total_steps=4, metric={"phase": "clustering"})

    clusters, Z = cluster_features(cooccur, top_indices, n_clusters=10)
    print(f"  Clustered into 10 groups")
    results["clusters"] = clusters.tolist()

    # ── Step 4: Evaluate against supervised labels ─────────────────────────
    print("[4/4] Evaluating against supervised labels...")
    report_progress(4, 4, step=4, total_steps=4, metric={"phase": "evaluation"})

    supervised_labels = get_supervised_labels(
        model, sae, tokenizer, DEVICE, top_indices
    )
    print(f"  Found {len(supervised_labels)} supervised first-letter features")
    results["supervised_labels"] = {str(k): v for k, v in supervised_labels.items()}

    eval_result = evaluate_uad(clusters, top_indices, supervised_labels)
    results["evaluation"] = eval_result

    print(f"  Precision: {eval_result['precision']:.3f}")
    print(f"  Recall: {eval_result['recall']:.3f}")
    print(f"  F1: {eval_result['f1']:.3f}")
    print(f"  True positives: {eval_result['true_positives']}")
    print(f"  Same-cluster pairs: {eval_result['same_cluster_pairs']}")
    print(f"  Supervised collisions: {eval_result['supervised_collisions']}")

    # ── Save results ───────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed
    results["end_time"] = datetime.now().isoformat()

    output_file = RESULTS_DIR / "p3_uad_results.json"
    output_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_file}")
    print(f"Total time: {elapsed:.1f}s")

    # Summary
    summary = (
        f"Precision={eval_result['precision']:.3f}, "
        f"Recall={eval_result['recall']:.3f}, "
        f"F1={eval_result['f1']:.3f}, "
        f"TP={eval_result['true_positives']}, "
        f"elapsed={elapsed:.1f}s"
    )

    mark_done(status="success", summary=summary)
    print("\nPilot P3 complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_msg}")
        mark_done(status="failed", summary=error_msg[:500])
        sys.exit(1)
