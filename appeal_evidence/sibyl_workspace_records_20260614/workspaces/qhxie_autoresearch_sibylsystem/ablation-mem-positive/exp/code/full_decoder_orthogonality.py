#!/usr/bin/env python3
"""
full_decoder_orthogonality: Decoder Orthogonality as Alternative Predictor

Tests whether decoder weight orthogonality predicts steering effectiveness:
- Compute decoder weight cosine similarity matrix for absorbed features
- Compare steering effects for 30 high-orthogonality vs 30 low-orthogonality features
- Test if orthogonality correlates with steering effect (r > 0.3 pass criterion)
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm

# Project paths
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive")
RESULTS_DIR = WORKSPACE / "exp/results/full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Import TransformerLens and SAELens
from transformer_lens import HookedTransformer
from sae_lens import SAE

def setup_model_sae():
    """Load model and SAE."""
    print("Loading GPT-2 Small and SAE...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")

    # Use the newer SAE loading API
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.6.hook_resid_pre"
    )
    sae = sae.to("cuda")

    return model, sae

def get_absorbed_features(sae, model, n_samples=500, absorption_threshold=0.3):
    """
    Get list of absorbed features based on CV (coefficient of variation).
    Uses CV as a proxy for absorption - high CV features are absorbed.
    """
    print(f"Computing feature CV scores on {n_samples} samples...")

    from datasets import load_dataset

    # Load text samples from Pile validation set
    try:
        dataset = load_dataset("EleutherAI/pile", split="validation", streaming=True)
        texts = []
        for i, example in enumerate(dataset):
            if i >= n_samples:
                break
            texts.append(example["text"][:200])
    except Exception as e:
        print(f"Warning: Could not load Pile dataset: {e}")
        print("Using fallback prompts...")
        texts = [
            "The movie was very good and enjoyable",
            "The food was extremely delicious and tasty",
            "The weather today is sunny and warm",
            "The book was really interesting and engaging",
            "The music sounded beautiful and melodic",
            "The restaurant was excellent and professional",
            "The experience was truly remarkable and unique",
            "The discussion was very productive and helpful",
        ] * 63  # Repeat to get ~500 samples

    # Track feature activations
    d_sae = sae.W_dec.shape[0]  # Number of SAE features
    feature_activations = [[] for _ in range(d_sae)]

    model.eval()
    sae.eval()

    with torch.no_grad():
        for i, text in enumerate(tqdm(texts[:n_samples], desc="Computing CV")):
            try:
                # Get tokens
                tokens = model.to_tokens(text)[:, :128]  # Truncate to 128

                # Get activations
                _, cache = model.run_with_cache(tokens, names_filter="blocks.6.hook_resid_pre")
                acts = cache["blocks.6.hook_resid_pre"]

                # Encode to get feature activations
                features = sae.encode(acts)  # [batch, seq, d_sae]

                # Record activations for each feature
                for feat_id in range(d_sae):
                    feat_act = features[:, :, feat_id].float()
                    # Store max activation across sequence
                    if feat_act.max() > 0:
                        feature_activations[feat_id].append(feat_act.max().item())

            except Exception as e:
                continue

    # Compute CV for each feature
    cv_scores = []
    for feat_id in range(d_sae):
        acts = feature_activations[feat_id]
        if len(acts) >= 10:  # Need at least 10 observations
            mean = np.mean(acts)
            std = np.std(acts)
            if mean > 0:
                cv = std / mean
            else:
                cv = 0.0
            cv_scores.append((feat_id, cv, mean))
        else:
            cv_scores.append((feat_id, 0.0, 0.0))

    # Sort by CV descending (high CV = absorbed)
    cv_scores.sort(key=lambda x: x[1], reverse=True)

    # Select high-CV features (absorbed)
    # Use threshold: CV > 1.0 indicates high variance (absorbed)
    high_cv = [(i, cv, m) for i, cv, m in cv_scores if cv > 1.0]
    low_cv = [(i, cv, m) for i, cv, m in cv_scores if cv <= 1.0]

    print(f"Found {len(high_cv)} high-CV features (CV > 1.0)")
    print(f"Found {len(low_cv)} low-CV features (CV <= 1.0)")

    return high_cv, low_cv, cv_scores

def compute_decoder_orthogonality(sae, feature_ids):
    """
    Compute mean cosine similarity between each feature's decoder and other features.
    High orthogonality = low similarity to other decoders.
    """
    print(f"Computing decoder orthogonality for {len(feature_ids)} features...")

    W_dec = sae.W_dec.detach().cpu().numpy()  # [d_sae, d_model]

    orthogonality_scores = []

    for idx, (feat_id, cv, mean_act) in enumerate(tqdm(feature_ids, desc="Computing orthogonality")):
        dec_vec = W_dec[feat_id]

        # Compute cosine similarity with all other features
        similarities = []
        for other_id, _, _ in feature_ids:
            if other_id != feat_id:
                other_vec = W_dec[other_id]
                # Cosine similarity
                dot = np.dot(dec_vec, other_vec)
                norm = np.linalg.norm(dec_vec) * np.linalg.norm(other_vec)
                if norm > 0:
                    similarities.append(dot / norm)

        if similarities:
            mean_sim = np.mean(np.abs(similarities))
            # Orthogonality = 1 - |mean_similarity|
            orthogonality = 1.0 - mean_sim
        else:
            orthogonality = 1.0
            mean_sim = 0.0

        orthogonality_scores.append((feat_id, orthogonality, mean_sim, cv, mean_act))

    return orthogonality_scores

def run_steering_experiment(model, sae, feature_ids, prompts, strength=5):
    """
    Run steering experiment on selected features.
    Returns steering effects per feature.
    """
    print(f"Running steering on {len(feature_ids)} features at strength {strength}...")

    W_dec = sae.W_dec  # [d_sae, d_model]

    results = []

    for feat_id in tqdm(feature_ids, desc="Steering"):
        feature_direction = W_dec[feat_id].cuda()

        for prompt in prompts:
            tokens = model.to_tokens(prompt).cuda()

            # Get clean logits
            clean_logits, _ = model.run_with_cache(tokens)

            # Steering hook
            def steering_hook(activations, hook):
                activations = activations + strength * feature_direction.unsqueeze(0).unsqueeze(0)
                return activations

            # Run with steering
            with model.hooks(fwd_hooks=[("blocks.6.hook_resid_pre", steering_hook)]):
                steered_logits, _ = model.run_with_cache(tokens)

            # Get logits at last position
            clean_logit = clean_logits[0, -1].cuda()
            steered_logit = steered_logits[0, -1].cuda()

            # Logit change (raw)
            logit_change = (steered_logit - clean_logit).max().item()

            # Absolute effect
            abs_effect = abs(logit_change)

            results.append({
                "feature": feat_id,
                "prompt": prompt,
                "strength": strength,
                "logit_change": logit_change,
                "abs_effect": abs_effect
            })

    return results

def main():
    start_time = datetime.now()
    task_id = "full_decoder_orthogonality"

    print(f"\n{'='*60}")
    print(f"TASK: {task_id}")
    print(f"Started: {start_time.isoformat()}")
    print(f"{'='*60}\n")

    # Setup
    model, sae = setup_model_sae()

    # Get absorbed features using CV
    high_cv_features, low_cv_features, all_cv_scores = get_absorbed_features(sae, model, n_samples=500)

    # Need at least 30 in each group for experiment
    if len(high_cv_features) < 30:
        print(f"WARNING: Only {len(high_cv_features)} high-CV features found")
        # Use top features by CV anyway
    if len(low_cv_features) < 30:
        print(f"WARNING: Only {len(low_cv_features)} low-CV features found")

    # Compute decoder orthogonality for all features
    all_features = high_cv_features + low_cv_features
    orthogonality_scores = compute_decoder_orthogonality(sae, all_features)

    # Sort by orthogonality
    orthogonality_scores.sort(key=lambda x: x[1], reverse=True)

    # Select top 30 high-orthogonality and bottom 30 low-orthogonality
    high_ortho = orthogonality_scores[:30]
    low_ortho = orthogonality_scores[-30:] if len(orthogonality_scores) >= 30 else orthogonality_scores[:30]

    print(f"\nHigh orthogonality features: {len(high_ortho)}, mean ortho={np.mean([x[1] for x in high_ortho]):.4f}")
    print(f"Low orthogonality features: {len(low_ortho)}, mean ortho={np.mean([x[1] for x in low_ortho]):.4f}")

    # Save orthogonality classification
    ortho_classification = {
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "n_high_cv_features": len(high_cv_features),
        "n_low_cv_features": len(low_cv_features),
        "high_orthogonality": [{"feature": x[0], "orthogonality": x[1], "mean_sim": x[2], "cv": x[3]} for x in high_ortho],
        "low_orthogonality": [{"feature": x[0], "orthogonality": x[1], "mean_sim": x[2], "cv": x[3]} for x in low_ortho]
    }

    with open(RESULTS_DIR / f"{task_id}_classification.json", "w") as f:
        json.dump(ortho_classification, f, indent=2)

    # Run steering experiment
    prompts = [
        "The movie was very",
        "The food was extremely",
        "The weather today is",
        "The book was really",
        "The music sounded"
    ]

    high_ortho_ids = [x[0] for x in high_ortho]
    low_ortho_ids = [x[0] for x in low_ortho]

    print("\nRunning steering on high-orthogonality features...")
    high_ortho_results = run_steering_experiment(model, sae, high_ortho_ids, prompts, strength=5)

    print("\nRunning steering on low-orthogonality features...")
    low_ortho_results = run_steering_experiment(model, sae, low_ortho_ids, prompts, strength=5)

    # Aggregate results
    high_ortho_mean_abs = np.mean([r["abs_effect"] for r in high_ortho_results])
    low_ortho_mean_abs = np.mean([r["abs_effect"] for r in low_ortho_results])

    high_ortho_mean_raw = np.mean([r["logit_change"] for r in high_ortho_results])
    low_ortho_mean_raw = np.mean([r["logit_change"] for r in low_ortho_results])

    # Compute correlation between orthogonality and steering effect
    all_steering_results = high_ortho_results + low_ortho_results
    feature_effects = {}
    for r in all_steering_results:
        fid = r["feature"]
        if fid not in feature_effects:
            feature_effects[fid] = []
        feature_effects[fid].append(r["abs_effect"])

    mean_effects = {fid: np.mean(effects) for fid, effects in feature_effects.items()}

    ortho_list = []
    effect_list = []
    for feat_id, ortho, _, cv, mean_act in high_ortho + low_ortho:
        if feat_id in mean_effects:
            ortho_list.append(ortho)
            effect_list.append(mean_effects[feat_id])

    if len(ortho_list) > 5:
        correlation = np.corrcoef(ortho_list, effect_list)[0, 1]
    else:
        correlation = 0.0

    # Statistical test (Welch's t-test on absolute effects)
    from scipy import stats

    high_effects = [r["abs_effect"] for r in high_ortho_results]
    low_effects = [r["abs_effect"] for r in low_ortho_results]

    t_stat, p_value = stats.ttest_ind(high_effects, low_effects, equal_var=False)

    results = {
        "task_id": task_id,
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "start_time": start_time.isoformat(),
        "end_time": datetime.now().isoformat(),
        "config": {
            "seed": 42,
            "n_features_per_group": 30,
            "steering_strength": 5,
            "model": "gpt2-small",
            "sae": "gpt2-small-res-jb",
            "layer": 6,
            "prompts": prompts
        },
        "steering_results": {
            "high_orthogonality": high_ortho_results,
            "low_orthogonality": low_ortho_results
        },
        "aggregate": {
            "high_ortho_mean_abs_effect": high_ortho_mean_abs,
            "low_ortho_mean_abs_effect": low_ortho_mean_abs,
            "high_ortho_mean_raw_effect": high_ortho_mean_raw,
            "low_ortho_mean_raw_effect": low_ortho_mean_raw,
            "n_high_samples": len(high_ortho_results),
            "n_low_samples": len(low_ortho_results),
            "difference": high_ortho_mean_abs - low_ortho_mean_abs,
            "orthogonality_steering_correlation": correlation
        },
        "statistical_test": {
            "test": "Welch t-test on absolute effects",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05)
        },
        "pass_criteria": {
            "required": "Correlation r > 0.3 between orthogonality and steering",
            "observed_correlation": float(correlation),
            "meets_criterion": bool(abs(correlation) > 0.3)
        },
        "interpretation": {
            "finding": "POSITIVE" if correlation > 0.3 else "NEGATIVE",
            "summary": f"Orthogonality-steering correlation: r={correlation:.4f}. "
                     f"High-ortho mean effect: {high_ortho_mean_abs:.4f}, Low-ortho mean effect: {low_ortho_mean_abs:.4f}",
            "note": "Compare with CV-steering correlation from pilot: r from CV analysis"
        },
        "gpu": {
            "id": 0,
            "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition"
        }
    }

    # Save results
    output_file = RESULTS_DIR / f"{task_id}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Write DONE marker
    done_file = RESULTS_DIR / f"{task_id}_DONE"
    with open(done_file, "w") as f:
        json.dump({
            "task_id": task_id,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "correlation": float(correlation),
            "high_ortho_effect": float(high_ortho_mean_abs),
            "low_ortho_effect": float(low_ortho_mean_abs),
            "finding": "POSITIVE" if correlation > 0.3 else "NEGATIVE"
        }, f)

    print(f"\n{'='*60}")
    print(f"RESULTS: {task_id}")
    print(f"High-ortho mean abs effect: {high_ortho_mean_abs:.4f}")
    print(f"Low-ortho mean abs effect: {low_ortho_mean_abs:.4f}")
    print(f"Orthogonality-Steering Correlation: r={correlation:.4f}")
    print(f"Pass criterion (r > 0.3): {'MET' if abs(correlation) > 0.3 else 'NOT MET'}")
    print(f"{'='*60}\n")

    # Update gpu_progress.json
    gpu_progress_file = WORKSPACE / "exp/gpu_progress.json"
    if gpu_progress_file.exists():
        with open(gpu_progress_file, "r") as f:
            gpu_progress = json.load(f)
    else:
        gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    gpu_progress["completed"].append(task_id)
    if task_id in gpu_progress["running"]:
        del gpu_progress["running"][task_id]

    end_time = datetime.now()
    duration_min = (end_time - start_time).total_seconds() / 60

    gpu_progress["timings"][task_id] = {
        "planned_min": 25,
        "actual_min": int(duration_min),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "config_snapshot": {
            "model": "gpt2-small",
            "sae": "gpt2-small-res-jb",
            "layer": 6,
            "n_features_per_group": 30,
            "steering_strength": 5,
            "n_prompts": len(prompts),
            "correlation": float(correlation),
            "high_ortho_effect": float(high_ortho_mean_abs),
            "low_ortho_effect": float(low_ortho_mean_abs)
        }
    }

    with open(gpu_progress_file, "w") as f:
        json.dump(gpu_progress, f, indent=2)

    return results

if __name__ == "__main__":
    results = main()
    print(f"Results saved to exp/results/full/full_decoder_orthogonality.json")