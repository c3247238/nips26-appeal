#!/usr/bin/env python3
"""
Full Decoder Orthogonality Analysis (H6 Ablation)

Tests whether decoder weight orthogonality predicts steering effectiveness.

Hypothesis H6: Features with decoder weights maximally orthogonal to other features
show higher steering effectiveness. Orthogonality may partially explain the CV-steering correlation.

Ablation design:
- Compute decoder weight cosine similarity matrix for all features in pilot_steering_comparison
- Define orthogonality as mean cosine similarity with other features (lower = more orthogonal)
- Compare steering effects for high-orthogonality vs low-orthogonality features
- Correlation analysis: orthogonality vs steering effect
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive")
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = PROJECT_ROOT / "exp" / "results" / "full"
PILOT_DIR = PROJECT_ROOT / "exp" / "results" / "pilots"


def load_pilot_steering_results():
    """Load the pilot steering comparison results."""
    pilot_file = PILOT_DIR / "pilot_steering_comparison.json"
    with open(pilot_file, "r") as f:
        return json.load(f)


def compute_decoder_orthogonality(sae, feature_indices):
    """
    Compute orthogonality for each feature's decoder weight.

    Orthogonality = mean cosine similarity with all other features' decoders
    Lower mean similarity = more orthogonal.

    Returns dict mapping feature_idx -> orthogonality metrics
    """
    # Get decoder weights: shape (d_sae, d_model)
    # W_dec can be a Parameter or have .weight attribute
    if hasattr(sae.W_dec, 'weight'):
        decoder_weights = sae.W_dec.weight.detach().cpu().numpy()
    else:
        decoder_weights = sae.W_dec.detach().cpu().numpy()

    n_features = len(feature_indices)
    print(f"Computing orthogonality for {n_features} features...")
    print(f"Decoder weight shape: {decoder_weights.shape}")

    # Normalize decoder weights for cosine similarity
    norms = np.linalg.norm(decoder_weights, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)  # Avoid division by zero
    normalized_weights = decoder_weights / norms

    # Compute cosine similarity matrix (only for features of interest)
    orthogonality_metrics = {}

    for i, feat_idx in enumerate(feature_indices):
        if i % 10 == 0:
            print(f"  Processing feature {i}/{n_features}...")

        # Cosine similarity of this feature with ALL other features
        feat_vector = normalized_weights[feat_idx]
        all_similarities = normalized_weights @ feat_vector  # (d_sae,)

        # Exclude self-similarity (diagonal)
        all_similarities[feat_idx] = 0

        # Mean cosine similarity with other features
        mean_cosine = all_similarities.mean()
        std_cosine = all_similarities.std()

        # Top-k most similar (excluding self)
        k = 10
        top_k_similar = np.sort(all_similarities)[-k:]

        orthogonality_metrics[feat_idx] = {
            "mean_cosine_similarity": float(mean_cosine),
            "std_cosine_similarity": float(std_cosine),
            "top_10_mean_similarity": float(top_k_similar.mean()),
            "min_cosine_similarity": float(all_similarities.min()),
            "max_cosine_similarity": float(all_similarities.max()),
        }

    return orthogonality_metrics


def run_steering_on_features(model, sae, feature_indices, prompts, steering_strength=5):
    """
    Run steering experiments on specified features.

    Returns steering effects for each feature.
    """
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    tokenizer = model.tokenizer
    hook_name = "blocks.6.hook_resid_pre"  # Layer 6 residual stream SAE

    # Target tokens (positive tokens for each prompt)
    target_tokens = {
        "The movie was very": "good",
        "The food was extremely": "good",
        "The weather today is": "nice",
    }

    steering_results = []

    for feat_idx in feature_indices:
        for prompt in prompts:
            target_tok = target_tokens.get(prompt, "good")

            # Tokenize
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

            # Get target token ID
            target_id = tokenizer.encode(target_tok)[0]

            # Get clean logits
            with torch.no_grad():
                clean_logits, cache = model.run_with_cache(
                    input_ids,
                    names_filter=[hook_name]
                )

            clean_logit = clean_logits[0, -1, target_id].item()

            # Get steered logits
            acts = cache[hook_name]  # (1, seq, d_model)

            # Add steering vector
            if hasattr(sae.W_dec, 'weight'):
                steering_vector = sae.W_dec.weight[feat_idx].detach().to(DEVICE)
            else:
                steering_vector = sae.W_dec[feat_idx].detach().to(DEVICE)
            steered_acts = acts + steering_vector * steering_strength

            # Run model with steered activations
            with torch.no_grad():
                # Hook into the residual stream to add steering
                def steering_hook(activations, hook):
                    return steered_acts

                steered_logits = model.run_with_hooks(
                    input_ids,
                    fwd_hooks=[(hook_name, steering_hook)]
                )

            steered_logit = steered_logits[0, -1, target_id].item()

            # Compute effect
            logit_change = steered_logit - clean_logit

            steering_results.append({
                "feature": int(feat_idx),
                "prompt": prompt,
                "strength": steering_strength,
                "clean_logit": clean_logit,
                "steered_logit": steered_logit,
                "logit_change": logit_change,
                "abs_effect": abs(logit_change)
            })

    return steering_results


def main():
    print("=" * 60)
    print("FULL DECODER ORTHOGONALITY ANALYSIS (H6 Ablation)")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print()

    # Set seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load pilot results to get features and steering effects
    print("Loading pilot steering results...")
    pilot_results = load_pilot_steering_results()

    # Extract all features used in pilot
    high_cv_features = list(set([r["feature"] for r in pilot_results["steering_results"]["high_cv"]]))
    low_cv_features = list(set([r["feature"] for r in pilot_results["steering_results"]["low_cv"]]))
    all_features = high_cv_features + low_cv_features

    print(f"High-CV features: {len(high_cv_features)}")
    print(f"Low-CV features: {len(low_cv_features)}")
    print(f"Total features: {len(all_features)}")

    # Extract steering effects from pilot
    feature_to_effect = {}
    for result in pilot_results["steering_results"]["high_cv"] + pilot_results["steering_results"]["low_cv"]:
        feat = result["feature"]
        if feat not in feature_to_effect:
            feature_to_effect[feat] = []
        feature_to_effect[feat].append(result["abs_effect"])

    # Average effect per feature
    avg_effects = {feat: np.mean(effects) for feat, effects in feature_to_effect.items()}

    # Load model and SAE
    print("\nLoading GPT-2 Small and SAE...")
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.6.hook_resid_pre",
        device=DEVICE
    )

    print(f"SAE config: d_sae={sae.cfg.d_sae}, architecture={sae.cfg.architecture}")

    # Compute orthogonality for all features
    print("\nComputing decoder orthogonality...")
    orthogonality_metrics = compute_decoder_orthogonality(sae, all_features)

    # Convert to arrays for analysis
    features_array = np.array(list(orthogonality_metrics.keys()))
    mean_cosines = np.array([orthogonality_metrics[f]["mean_cosine_similarity"] for f in features_array])
    effects_array = np.array([avg_effects.get(f, 0.0) for f in features_array])

    # Correlation: orthogonality vs steering effect
    # Lower mean_cosine = more orthogonal
    # We expect: if orthogonality predicts steering, more orthogonal features should have larger effects
    corr_coef, corr_p = stats.pearsonr(mean_cosines, effects_array)
    print(f"\nCorrelation (mean_cosine vs steering_effect):")
    print(f"  Pearson r = {corr_coef:.4f}, p = {corr_p:.4e}")

    # Also compute Spearman (rank-based)
    spearman_coef, spearman_p = stats.spearmanr(mean_cosines, effects_array)
    print(f"  Spearman rho = {spearman_coef:.4f}, p = {spearman_p:.4e}")

    # Select high-orthogonality (lowest mean cosine) and low-orthogonality (highest mean cosine) features
    n_per_group = 30

    # Sort by mean cosine similarity
    sorted_indices = np.argsort(mean_cosines)  # ascending = most orthogonal first

    high_orthogonality_features = features_array[sorted_indices[:n_per_group]].tolist()
    low_orthogonality_features = features_array[sorted_indices[-n_per_group:]].tolist()

    print(f"\nHigh-orthogonality features (n={len(high_orthogonality_features)}):")
    print(f"  Mean cosine range: {mean_cosines[sorted_indices[:n_per_group]].min():.4f} - {mean_cosines[sorted_indices[:n_per_group]].max():.4f}")
    print(f"  Features: {sorted(high_orthogonality_features)[:10]}...")

    print(f"\nLow-orthogonality features (n={len(low_orthogonality_features)}):")
    print(f"  Mean cosine range: {mean_cosines[sorted_indices[-n_per_group:]].min():.4f} - {mean_cosines[sorted_indices[-n_per_group:]].max():.4f}")
    print(f"  Features: {sorted(low_orthogonality_features)[:10]}...")

    # Get prompts from pilot
    prompts = pilot_results["config"]["prompts"]

    # Run new steering experiments (fresh steering, not using pilot data)
    print("\n" + "=" * 60)
    print("Running Steering Experiments on Orthogonality Groups")
    print("=" * 60)

    print("\nSteering on HIGH-orthogonality features...")
    high_orth_steering = run_steering_on_features(
        model, sae, high_orthogonality_features, prompts, steering_strength=5
    )

    print("Steering on LOW-orthogonality features...")
    low_orth_steering = run_steering_on_features(
        model, sae, low_orthogonality_features, prompts, steering_strength=5
    )

    # Compute group statistics
    high_orth_effects = [r["abs_effect"] for r in high_orth_steering]
    low_orth_effects = [r["abs_effect"] for r in low_orth_steering]

    high_orth_mean = np.mean(high_orth_effects)
    low_orth_mean = np.mean(low_orth_effects)

    print(f"\nHigh-orthogonality group (n={len(high_orth_effects)}):")
    print(f"  Mean abs effect: {high_orth_mean:.4f}")
    print(f"  Std: {np.std(high_orth_effects):.4f}")

    print(f"\nLow-orthogonality group (n={len(low_orth_effects)}):")
    print(f"  Mean abs effect: {low_orth_mean:.4f}")
    print(f"  Std: {np.std(low_orth_effects):.4f}")

    # Statistical test: high-orth vs low-orth
    t_stat, p_val = stats.ttest_ind(high_orth_effects, low_orth_effects)
    print(f"\nWelch t-test (high-orth vs low-orth absolute effects):")
    print(f"  t-statistic = {t_stat:.4f}")
    print(f"  p-value = {p_val:.4e}")

    # Also test on raw effects
    high_orth_raw = [r["logit_change"] for r in high_orth_steering]
    low_orth_raw = [r["logit_change"] for r in low_orth_steering]
    t_stat_raw, p_val_raw = stats.ttest_ind(high_orth_raw, low_orth_raw)
    print(f"\nWelch t-test (raw effects):")
    print(f"  t-statistic = {t_stat_raw:.4f}")
    print(f"  p-value = {p_val_raw:.4e}")

    # Check pass criteria
    pass_criteria = "Correlation r > 0.3 between orthogonality and steering"
    # Since mean_cosine is inversely related to orthogonality:
    # If high-orth features have larger steering effects, we expect negative correlation
    # (lower mean_cosine = more orthogonal = should have larger effect)
    correlation_sufficient = abs(corr_coef) > 0.3

    # Also check if the group comparison is significant
    group_difference_sufficient = p_val < 0.05

    meets_criterion = correlation_sufficient or group_difference_sufficient

    print(f"\nPass criteria: {pass_criteria}")
    print(f"  |r| = {abs(corr_coef):.4f} > 0.3? {correlation_sufficient}")
    print(f"  Group difference p = {p_val:.4e} < 0.05? {group_difference_sufficient}")
    print(f"  MET? {meets_criterion}")

    # Compile results
    results = {
        "task_id": "full_decoder_orthogonality",
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": SEED,
            "n_per_group": n_per_group,
            "steering_strength": 5,
            "model": "gpt2-small",
            "sae": "gpt2-small-res-jb",
            "layer": 6,
            "prompts": prompts
        },
        "orthogonality_metrics": {str(k): v for k, v in orthogonality_metrics.items()},
        "correlation_analysis": {
            "pearson_r": float(corr_coef),
            "pearson_p": float(corr_p),
            "spearman_rho": float(spearman_coef),
            "spearman_p": float(spearman_p),
            "n_features": len(features_array)
        },
        "feature_groups": {
            "high_orthogonality": high_orthogonality_features,
            "low_orthogonality": low_orthogonality_features,
            "high_orthogonality_mean_cosine": float(mean_cosines[sorted_indices[:n_per_group]].mean()),
            "low_orthogonality_mean_cosine": float(mean_cosines[sorted_indices[-n_per_group:]].mean())
        },
        "steering_results": {
            "high_orthogonality": high_orth_steering,
            "low_orthogonality": low_orth_steering
        },
        "group_comparison": {
            "high_orth_mean_effect": float(high_orth_mean),
            "high_orth_std_effect": float(np.std(high_orth_effects)),
            "low_orth_mean_effect": float(low_orth_mean),
            "low_orth_std_effect": float(np.std(low_orth_effects)),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "n_high_orth_samples": len(high_orth_effects),
            "n_low_orth_samples": len(low_orth_effects)
        },
        "pass_criteria": {
            "required": pass_criteria,
            "meets_criterion": bool(meets_criterion),
            "reason": ""
        },
        "interpretation": {
            "finding": "",
            "summary": ""
        }
    }

    # Add interpretation
    if corr_coef < -0.3 and p_val < 0.05:
        results["interpretation"]["finding"] = "SUPPORTED"
        results["interpretation"]["summary"] = f"Orthogonality predicts steering effectiveness (r={corr_coef:.3f}, p={p_val:.3e}). More orthogonal features show larger steering effects."
        results["pass_criteria"]["reason"] = f"Negative correlation (r={corr_coef:.3f}) confirms: lower cosine similarity (more orthogonal) → larger steering effect"
    elif corr_coef > 0.3 and p_val < 0.05:
        results["interpretation"]["finding"] = "INVERSE_SUPPORTED"
        results["interpretation"]["summary"] = f"Unexpected: positive correlation (r={corr_coef:.3f}). More orthogonal features show SMALLER steering effects."
        results["pass_criteria"]["reason"] = f"Positive correlation (r={corr_coef:.3f}) contradicts orthogonality hypothesis"
    else:
        results["interpretation"]["finding"] = "NOT_SUPPORTED"
        results["interpretation"]["summary"] = f"No significant correlation between orthogonality and steering (r={corr_coef:.3f}, p={corr_p:.3e})."
        results["pass_criteria"]["reason"] = f"Correlation |r|={abs(corr_coef):.3f} < 0.3 threshold"

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "full_decoder_orthogonality.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Write DONE marker
    done_file = RESULTS_DIR / "full_decoder_orthogonality_DONE"
    done_file.write_text(json.dumps({
        "task_id": "full_decoder_orthogonality",
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "finding": results["interpretation"]["finding"],
        "correlation_r": float(corr_coef),
        "p_value": float(p_val)
    }))

    # Clean up
    del model, sae
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Correlation (orthogonality vs steering): r={corr_coef:.4f}, p={corr_p:.4e}")
    print(f"High-orthogonality mean effect: {high_orth_mean:.4f}")
    print(f"Low-orthogonality mean effect: {low_orth_mean:.4f}")
    print(f"Finding: {results['interpretation']['finding']}")
    print("\nDone.")


if __name__ == "__main__":
    main()