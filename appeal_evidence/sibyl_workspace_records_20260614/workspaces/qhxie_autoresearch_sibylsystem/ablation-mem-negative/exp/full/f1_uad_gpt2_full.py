#!/usr/bin/env python3
"""
Full E1: UAD on GPT-2 Small (Full Scale)
Scale UAD to 10k tokens, 500 top features, 50 clusters on GPT-2 Small layer 8.
Target: F1 >= 0.6.
"""

import os
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import SAE
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TASK_ID = "f1_uad_gpt2_full"
LAYER = 8
N_SAMPLES = 1000  # 1000 texts x ~10 tokens each = ~10k tokens
TOP_K = 500
N_CLUSTERS = 50

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
        except:
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress, "timestamp": datetime.now().isoformat(),
    }))

# ── Set seeds ───────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Helper: build co-occurrence matrix ──────────────────────────────
def build_cooccurrence_matrix(model, sae, tokenizer, device, n_samples=1000):
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence research daily.",
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
    ] * 50

    tokenized = tokenizer(
        prompts[:n_samples], return_tensors="pt", padding=True,
        truncation=True, max_length=64
    )
    input_ids = tokenized["input_ids"].to(device)

    hook_name = getattr(sae.cfg, 'hook_name', f'blocks.{LAYER}.hook_resid_pre')
    with torch.no_grad():
        _, cache = model.run_with_cache(input_ids, names_filter=[hook_name])
        acts = cache[hook_name]
        sae_acts = sae.encode(acts)
        flat = sae_acts.reshape(-1, sae_acts.shape[-1])
        binary = (flat > 0).float().cpu().numpy()

    feature_activity = binary.sum(axis=0)
    top_k = min(TOP_K, len(feature_activity))
    top_indices = np.argsort(feature_activity)[-top_k:]

    subset = binary[:, top_indices]
    cooccur = subset.T @ subset
    counts = subset.sum(axis=0, keepdims=True).T
    cooccur_norm = cooccur / (counts + 1e-8)

    return cooccur_norm, top_indices, feature_activity, binary

# ── Helper: clustering ──────────────────────────────────────────────
def cluster_features(cooccur_matrix, top_indices, n_clusters=50):
    dist_matrix = 1 - cooccur_matrix
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    dist_matrix = np.clip(dist_matrix, 0, 1)
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method='ward')
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    return clusters, Z

# ── Helper: supervised labels (first-letter features) ───────────────
def get_supervised_labels(model, sae, tokenizer, device, top_indices):
    letters = [chr(ord('a') + i) for i in range(26)]
    prompts = []
    labels = []
    for letter in letters:
        words = [f"{letter}{w}" for w in ["pple", "nimal", "lpha", "rt"]]
        for word in words:
            prompts.append(f"The word '{word}'")
            labels.append(ord(letter) - ord('a'))

    tokenized = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=32
    )
    input_ids = tokenized["input_ids"].to(device)

    hook_name = getattr(sae.cfg, 'hook_name', f'blocks.{LAYER}.hook_resid_pre')
    with torch.no_grad():
        _, cache = model.run_with_cache(input_ids, names_filter=[hook_name])
        acts = cache[hook_name]
        sae_acts = sae.encode(acts)
        sample_acts = sae_acts.max(dim=1).values

    labels_t = torch.tensor(labels, device=device)
    supervised_features = {}
    for letter_idx in range(26):
        mask = labels_t == letter_idx
        if mask.sum() == 0:
            continue
        letter_acts = sample_acts[mask].mean(dim=0)
        other_acts = sample_acts[~mask].mean(dim=0)
        scores = letter_acts - other_acts

        best_score = -float('inf')
        best_feat = None
        for idx in top_indices:
            if scores[idx] > best_score:
                best_score = scores[idx].item()
                best_feat = idx

        if best_feat is not None:
            supervised_features[letter_idx] = best_feat

    return supervised_features

# ── Helper: evaluate UAD ────────────────────────────────────────────
def evaluate_uad(clusters, top_indices, supervised_labels):
    cluster_map = {int(top_indices[i]): int(clusters[i]) for i in range(len(top_indices))}

    same_cluster_pairs = []
    letter_list = list(supervised_labels.keys())
    for i in range(len(letter_list)):
        for j in range(i + 1, len(letter_list)):
            li, lj = letter_list[i], letter_list[j]
            fi, fj = supervised_labels[li], supervised_labels[lj]
            ci = cluster_map.get(fi, -1)
            cj = cluster_map.get(fj, -1)
            if ci == cj and ci != -1:
                same_cluster_pairs.append((li, lj))

    supervised_collisions = []
    feat_to_letters = {}
    for letter, feat in supervised_labels.items():
        feat_to_letters.setdefault(feat, []).append(letter)
    for feat, letters in feat_to_letters.items():
        if len(letters) > 1:
            for i in range(len(letters)):
                for j in range(i + 1, len(letters)):
                    supervised_collisions.append((letters[i], letters[j]))

    if same_cluster_pairs:
        true_positives = sum(1 for p in same_cluster_pairs if p in supervised_collisions)
        precision = true_positives / len(same_cluster_pairs)
    else:
        precision = 0.0
        true_positives = 0

    if supervised_collisions:
        recall = sum(1 for p in supervised_collisions if p in same_cluster_pairs) / len(supervised_collisions)
    else:
        recall = 0.0

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
    }

# ── Main ────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    print("=" * 60)
    print("Full E1: UAD on GPT-2 Small (Full Scale)")
    print("=" * 60)

    report_progress(0, 4, step=1, total_steps=4, metric={"phase": "init"})

    # Load model and SAE
    print("[1/4] Loading GPT-2 Small and SAE...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{LAYER}.hook_resid_pre",
        device=DEVICE,
    )
    print(f"  Loaded SAE: d_sae={sae.cfg.d_sae}")

    # Build co-occurrence
    print(f"[2/4] Building co-occurrence matrix ({N_SAMPLES} samples)...")
    report_progress(1, 4, step=2, total_steps=4, metric={"phase": "cooccurrence"})
    cooccur, top_indices, feature_activity, binary = build_cooccurrence_matrix(
        model, sae, tokenizer, DEVICE, n_samples=N_SAMPLES
    )
    print(f"  Co-occurrence: {cooccur.shape}")
    print(f"  Total positions: {binary.shape[0]}")
    print(f"  Alive features: {(feature_activity > 0).sum()} / {len(feature_activity)}")

    # Clustering
    print("[3/4] Clustering...")
    report_progress(2, 4, step=3, total_steps=4, metric={"phase": "clustering"})
    clusters, Z = cluster_features(cooccur, top_indices, n_clusters=N_CLUSTERS)

    # Supervised labels
    print("[4/4] Evaluating...")
    report_progress(3, 4, step=4, total_steps=4, metric={"phase": "evaluation"})
    supervised_labels = get_supervised_labels(model, sae, tokenizer, DEVICE, top_indices)
    print(f"  Supervised labels: {len(supervised_labels)} letters")

    eval_result = evaluate_uad(clusters, top_indices, supervised_labels)

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Precision: {eval_result['precision']:.4f}")
    print(f"Recall:    {eval_result['recall']:.4f}")
    print(f"F1 Score:  {eval_result['f1']:.4f}")
    print(f"TP: {eval_result['true_positives']}")
    print(f"Same-cluster pairs: {eval_result['same_cluster_pairs']}")
    print(f"Supervised collisions: {eval_result['supervised_collisions']}")
    print(f"Runtime: {elapsed:.1f}s")

    results = {
        "task_id": TASK_ID,
        "model": "gpt2-small",
        "layer": LAYER,
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "d_sae": sae.cfg.d_sae,
        "top_indices_count": len(top_indices),
        "n_clusters": N_CLUSTERS,
        "supervised_labels": {str(k): int(v) for k, v in supervised_labels.items()},
        "evaluation": eval_result,
        "feature_activity_stats": {
            "mean": float(feature_activity.mean()),
            "std": float(feature_activity.std()),
            "max": float(feature_activity.max()),
            "min": float(feature_activity.min()),
        },
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }

    output_path = RESULTS_DIR / f"{TASK_ID}_results.json"
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[Save] Results saved to {output_path}")

    passed = eval_result['f1'] >= 0.6 and elapsed < 1200
    print(f"\n[{'PASS' if passed else 'FAIL'}] F1={eval_result['f1']:.4f} (target >= 0.6), Runtime={elapsed:.1f}s")

    mark_done(status="success" if passed else "failure",
              summary=f"UAD full: F1={eval_result['f1']:.4f}, P={eval_result['precision']:.4f}, R={eval_result['recall']:.4f}, T={elapsed:.1f}s")

    return results

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_msg}")
        mark_done(status="failure", summary=error_msg[:500])
        raise
