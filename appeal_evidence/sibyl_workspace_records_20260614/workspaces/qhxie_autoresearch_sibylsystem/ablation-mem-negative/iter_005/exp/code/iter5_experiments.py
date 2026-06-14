#!/usr/bin/env python3
"""
Iteration 5 Experiments:
E1: Semantic hierarchy test (animal, emotion, color)
E2: Decoder weight similarity pilot
E3: Ground truth expansion (numbers 1-12)

GPU: 4
Estimated time: 60 min
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
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TASK_ID = "iter5_experiments"

# ── Progress Tracking ───────────────────────────────────────────────
def report_progress(step, total, desc="", metrics=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID, "step": step, "total": total,
        "description": desc, "metrics": metrics or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary="", results=None):
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    data = {
        "task_id": TASK_ID, "status": status, "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }
    if results:
        data["results"] = results
    marker.write_text(json.dumps(data))

# ── Hierarchy Definitions ───────────────────────────────────────────
SEMANTIC_HIERARCHIES = {
    "animal": {
        "parent": ["animal", "creature", "being"],
        "children": {
            "dog": ["dog", "puppy", "canine"],
            "cat": ["cat", "kitten", "feline"],
            "bird": ["bird", "avian"],
            "fish": ["fish", "aquatic"],
        }
    },
    "emotion": {
        "parent": ["emotion", "feeling"],
        "children": {
            "joy": ["joy", "happy", "happiness"],
            "sadness": ["sadness", "sad", "sorrow"],
            "anger": ["anger", "angry", "rage"],
            "fear": ["fear", "afraid", "scared"],
        }
    },
    "color": {
        "parent": ["color", "colour"],
        "children": {
            "red": ["red", "crimson"],
            "blue": ["blue", "azure"],
            "green": ["green", "emerald"],
            "yellow": ["yellow", "gold"],
        }
    },
}

NUMBER_HIERARCHY = {
    "numbers_1_12": {
        "children": {str(n): [str(n)] for n in range(1, 13)}
    }
}

# ── Helper: Get top-K features for a concept ────────────────────────
def get_concept_features(model, sae, layer, words, k=10, max_samples=100):
    """Find top-K SAE features that activate on given words."""
    feature_activations = {}

    for word in words[:max_samples]:
        try:
            tokens = model.to_tokens(word)
            _, cache = model.run_with_cache(tokens)
            acts = cache["resid_pre", layer]
            features = sae.encode(acts)
            # Mean over positions
            mean_feats = features[0].mean(dim=0).cpu()  # [d_sae]
            for feat_idx in mean_feats.topk(k * 2).indices.tolist():
                feat_val = mean_feats[feat_idx].item()
                if feat_idx not in feature_activations:
                    feature_activations[feat_idx] = []
                feature_activations[feat_idx].append(feat_val)
        except Exception as e:
            print(f"  Warning: error processing '{word}': {e}")
            continue

    # Aggregate: mean activation per feature
    aggregated = {feat: np.mean(vals) for feat, vals in feature_activations.items()}
    # Top K
    topk = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:k]
    return [feat for feat, _ in topk]

# ── E1: Semantic Hierarchy Test ─────────────────────────────────────
def run_semantic_hierarchy_test(model, sae, layer, k=10):
    """Test UAD on semantic hierarchies where children CAN co-occur."""
    results = {}

    for hierarchy_name, hierarchy in SEMANTIC_HIERARCHIES.items():
        print(f"\n[E1] Testing hierarchy: {hierarchy_name}")

        # Get top-K features for each child concept
        child_features = {}
        for child_name, words in hierarchy["children"].items():
            print(f"  Finding features for '{child_name}'...")
            feats = get_concept_features(model, sae, layer, words, k=k)
            child_features[child_name] = feats
            print(f"    Top features: {feats[:5]}...")

        # Compute collision rate (Jaccard) between all child pairs
        pairs = list(hierarchy["children"].keys())
        n_pairs = len(pairs) * (len(pairs) - 1) // 2
        collision_rates = []

        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                p1, p2 = pairs[i], pairs[j]
                set1 = set(child_features[p1])
                set2 = set(child_features[p2])
                if set1 or set2:
                    jaccard = len(set1 & set2) / len(set1 | set2)
                else:
                    jaccard = 0.0
                collision_rates.append({
                    "pair": (p1, p2),
                    "jaccard": jaccard,
                    "shared": list(set1 & set2),
                })

        # Summary stats
        jaccards = [r["jaccard"] for r in collision_rates]
        avg_collision = np.mean(jaccards) if jaccards else 0

        print(f"  Average collision rate: {avg_collision:.3f}")
        print(f"  Max collision: {max(jaccards):.3f}, Min: {min(jaccards):.3f}")

        results[hierarchy_name] = {
            "child_features": child_features,
            "collision_rates": collision_rates,
            "avg_collision": avg_collision,
            "max_collision": max(jaccards) if jaccards else 0,
            "min_collision": min(jaccards) if jaccards else 0,
        }

    return results

# ── E2: Decoder Weight Similarity ───────────────────────────────────
def run_decoder_similarity(sae, known_pairs=None, n_random=70):
    """Compute decoder weight cosine similarity for absorption vs random pairs."""
    print("\n[E2] Computing decoder weight similarities...")

    W_dec = sae.W_dec.detach().cpu()  # [d_sae, d_model]
    d_sae = W_dec.shape[0]

    # Normalize for cosine similarity
    W_dec_norm = W_dec / (W_dec.norm(dim=1, keepdim=True) + 1e-10)

    pairs_to_test = []

    # Known absorption pairs (from token-disjoint hierarchies)
    if known_pairs:
        for p in known_pairs[:10]:
            pairs_to_test.append(("known_absorption", p[0], p[1]))

    # Random pairs
    rng = np.random.RandomState(42)
    for _ in range(n_random):
        i, j = rng.choice(d_sae, 2, replace=False)
        pairs_to_test.append(("random", int(i), int(j)))

    results = []
    for label, i, j in pairs_to_test:
        sim = (W_dec_norm[i] @ W_dec_norm[j]).item()
        results.append({"label": label, "i": i, "j": j, "similarity": sim})

    # Summary stats
    known_sims = [r["similarity"] for r in results if r["label"] == "known_absorption"]
    random_sims = [r["similarity"] for r in results if r["label"] == "random"]

    print(f"  Known absorption pairs: n={len(known_sims)}, mean_sim={np.mean(known_sims):.4f}")
    print(f"  Random pairs: n={len(random_sims)}, mean_sim={np.mean(random_sims):.4f}")

    if known_sims and random_sims:
        print(f"  Difference: {np.mean(known_sims) - np.mean(random_sims):.4f}")

    return results

# ── E3: Ground Truth Expansion ──────────────────────────────────────
def expand_ground_truth(model, sae, layer, k=10):
    """Expand number hierarchy to 1-12."""
    print("\n[E3] Expanding ground truth (numbers 1-12)...")

    numbers = list(range(1, 13))
    number_words = {str(n): [str(n)] for n in numbers}

    # Get features for each number
    num_features = {}
    for num_str in number_words:
        feats = get_concept_features(model, sae, layer, number_words[num_str], k=k)
        num_features[num_str] = feats

    # Compute all pair collision rates
    all_pairs = []
    keys = list(num_features.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            p1, p2 = keys[i], keys[j]
            set1 = set(num_features[p1])
            set2 = set(num_features[p2])
            if set1 or set2:
                jaccard = len(set1 & set2) / len(set1 | set2)
            else:
                jaccard = 0.0
            all_pairs.append({"pair": (p1, p2), "jaccard": jaccard})

    jaccards = [p["jaccard"] for p in all_pairs]

    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  Avg collision: {np.mean(jaccards):.3f}")
    print(f"  Pairs with collision > 0: {sum(1 for j in jaccards if j > 0)}/{len(jaccards)}")

    return {
        "number_features": num_features,
        "collision_rates": all_pairs,
        "avg_collision": np.mean(jaccards),
        "n_pairs": len(all_pairs),
    }

# ── Main ────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    print("=" * 60)
    print("Iteration 5: Semantic Hierarchies + Decoder Similarity")
    print("=" * 60)

    report_progress(1, 4, "Loading model and SAE")

    print(f"\n[Setup] Loading GPT-2 Small... (GPU: {DEVICE})")
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    layer = 8
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{layer}.hook_resid_pre",
        device=DEVICE
    )
    print(f"[Setup] SAE: d_sae={sae.cfg.d_sae}, d_model={sae.cfg.d_in}")

    all_results = {"task_id": TASK_ID, "layer": layer, "timestamp": datetime.now().isoformat()}

    # ── E1: Semantic Hierarchies ──────────────────────────────────
    report_progress(2, 4, "E1: Semantic hierarchy test")
    semantic_results = run_semantic_hierarchy_test(model, sae, layer, k=10)
    all_results["semantic_hierarchies"] = semantic_results

    # ── E2: Decoder Similarity ────────────────────────────────────
    report_progress(3, 4, "E2: Decoder weight similarity")
    # Use number features from previous iteration as known pairs
    known_pairs = [
        (11513, 24189),  # from previous results: number absorption features
    ]
    decoder_results = run_decoder_similarity(sae, known_pairs=known_pairs, n_random=100)
    all_results["decoder_similarity"] = decoder_results

    # ── E3: Ground Truth Expansion ────────────────────────────────
    report_progress(4, 4, "E3: Ground truth expansion")
    expansion_results = expand_ground_truth(model, sae, layer, k=10)
    all_results["ground_truth_expansion"] = expansion_results

    # ── Save ──────────────────────────────────────────────────────
    runtime = time.time() - start_time
    all_results["runtime_seconds"] = runtime

    output_path = RESULTS_DIR / f"{TASK_ID}_results.json"
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n[Save] Results saved to {output_path}")

    print(f"\n{'=' * 60}")
    print(f"Runtime: {runtime:.1f}s")
    print("=" * 60)

    mark_done("success", f"Iter 5 complete: {len(semantic_results)} hierarchies tested, {len(decoder_results)} decoder pairs, {expansion_results['n_pairs']} number pairs", all_results)

    return all_results

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        mark_done("failure", f"Error: {str(e)}\n{traceback.format_exc()}")
        raise
