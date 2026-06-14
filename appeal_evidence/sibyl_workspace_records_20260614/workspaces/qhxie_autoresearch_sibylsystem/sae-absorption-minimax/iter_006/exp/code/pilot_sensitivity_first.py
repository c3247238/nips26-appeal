#!/usr/bin/env python3
"""
Pilot H-A: Sensitivity-First Steering Validation

Test H-A: Low-sensitivity features (Q3) show steering not different from random baseline.
Uses Q3 features from iter_008 pilot_classify_features.json. Steering at beta=5.

Pass criteria: Q3 steering ≈ random baseline (p > 0.05)
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from scipy import stats

# Constants
WORKSPACE = "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current"
RESULTS_DIR = f"{WORKSPACE}/exp/results/pilots"
SEED = 42
BETA = 5.0
N_TOKENS = 500  # tokens for steering test

# Q3 features from iter_008 (20 features with UAS=1.0 and low sensitivity from pilot_classify_features.json)
Q3_FEATURES = [
    10454, 589, 18654, 8304, 22005, 17373, 9990, 565, 3356, 17542,
    22191, 747, 17659, 11842, 19683, 7772, 7815, 8397, 19155, 23664
]

print(f"[{datetime.now().isoformat()}] Starting pilot_sensitivity_first")
print(f"Task: H-A Sensitivity-First Steering Validation")
print(f"Q3 features: {len(Q3_FEATURES)}")
print(f"Beta: {BETA}")
print(f"Test tokens: {N_TOKENS}")

# Set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_model_and_sae():
    """Load GPT-2 Small + SAE layer 8 from SAELens"""
    print("\nLoading model and SAE...")

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    # Load GPT-2 Small
    model = HookedTransformer.from_pretrained("gpt2-small")
    print(f"Model loaded: gpt2-small, layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")

    # Load SAE from SAELens
    sae = SAE.from_pretrained("gpt2-small-res-jb", "blocks.8.hook_resid_pre")
    hook_name = sae.cfg.metadata.hook_name
    print(f"SAE loaded: gpt2-small-res-jb, hook={hook_name}, d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}")

    return model, sae, hook_name

def get_steering_tokens(model, n_tokens=500):
    """Get diverse tokens for steering test"""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "It is a beautiful day for a walk in the park.",
        "Scientists have discovered a new species of plant.",
        "The company announced record profits this quarter.",
        "The weather forecast shows sunshine tomorrow.",
        "Students are studying for their final exams.",
        "The restaurant serves delicious food every day.",
        "Music plays softly in the background.",
        "The book contains many interesting stories.",
        "Children are playing in the playground.",
    ]

    all_tokens = []
    for text in texts * 10:  # Repeat to get enough tokens
        tokens = model.to_tokens(text)
        all_tokens.extend(tokens[0].tolist())

    return all_tokens[:n_tokens]

def compute_steering_effect(model, sae, hook_name, feature_idx, tokens, beta):
    """
    Compute steering effect for a single feature.

    Steering effect = change in cosine similarity between residual and feature direction
    after adding beta * W_dec[feature] to the residual stream.
    """
    device = model.cfg.device

    # Get decoder direction for this feature
    W_dec = sae.W_dec.detach().to(device)  # (d_sae, d_in)
    feature_direction = W_dec[feature_idx]  # (d_in,)
    feature_direction = feature_direction / feature_direction.norm()

    # Get original residuals at the hook
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens.to(device), names_filter=[hook_name])
        resid = cache[hook_name]  # (batch, seq, d_in)

    # Compute cosine similarity before steering for each position
    cos_before_list = []
    for pos in range(resid.shape[1]):
        resid_vec = resid[0, pos]
        cos_before = torch.nn.functional.cosine_similarity(
            resid_vec.unsqueeze(0), feature_direction.unsqueeze(0)
        ).item()
        cos_before_list.append(cos_before)

    # Apply steering and compute cosine similarity after
    steered_resid = resid + beta * feature_direction.unsqueeze(0).unsqueeze(0)

    cos_after_list = []
    for pos in range(steered_resid.shape[1]):
        resid_vec = steered_resid[0, pos]
        cos_after = torch.nn.functional.cosine_similarity(
            resid_vec.unsqueeze(0), feature_direction.unsqueeze(0)
        ).item()
        cos_after_list.append(cos_after)

    # Mean effect across positions
    effect = np.mean([a - b for a, b in zip(cos_after_list, cos_before_list)])
    return effect

def compute_random_effect(model, sae, hook_name, tokens, beta, n_directions=100):
    """
    Compute baseline steering effects for random directions.
    This establishes what effect looks like for non-feature directions.
    """
    device = model.cfg.device
    d_in = sae.cfg.d_in

    # Get a sample residual for computing effect scale
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens.to(device), names_filter=[hook_name])
        resid = cache[hook_name]  # (batch, seq, d_in)

    random_effects = []

    for _ in range(n_directions):
        # Random direction with same norm as feature directions
        random_dir = torch.randn(d_in).to(device)
        random_dir = random_dir / random_dir.norm()

        # Cosine similarity before
        cos_before_list = []
        for pos in range(resid.shape[1]):
            cos_before = torch.nn.functional.cosine_similarity(
                resid[0, pos].unsqueeze(0), random_dir.unsqueeze(0)
            ).item()
            cos_before_list.append(cos_before)

        # After steering
        steered_resid = resid + beta * random_dir.unsqueeze(0).unsqueeze(0)

        cos_after_list = []
        for pos in range(steered_resid.shape[1]):
            cos_after = torch.nn.functional.cosine_similarity(
                steered_resid[0, pos].unsqueeze(0), random_dir.unsqueeze(0)
            ).item()
            cos_after_list.append(cos_after)

        effect = np.mean([a - b for a, b in zip(cos_after_list, cos_before_list)])
        random_effects.append(effect)

    return random_effects

def main():
    # Load model and SAE
    model, sae, hook_name = load_model_and_sae()

    # Get steering tokens
    print("\nGenerating steering test tokens...")
    tokens_list = get_steering_tokens(model, n_tokens=N_TOKENS)
    tokens = torch.tensor(tokens_list).unsqueeze(0)
    print(f"Using {len(tokens_list)} tokens for steering tests")

    # Compute steering effects for Q3 features
    print("\nComputing steering effects for Q3 features...")
    q3_effects = []

    for i, feat_id in enumerate(Q3_FEATURES):
        effect = compute_steering_effect(model, sae, hook_name, feat_id, tokens, BETA)
        q3_effects.append(effect)

        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(Q3_FEATURES)} features, mean_effect={np.mean(q3_effects):.4f}")

    print(f"\nQ3 steering effects: mean={np.mean(q3_effects):.4f}, std={np.std(q3_effects):.4f}")

    # Compute random baseline
    print("\nComputing random baseline (100 random directions)...")
    random_effects = compute_random_effect(model, sae, hook_name, tokens, BETA, n_directions=100)

    print(f"Random baseline: mean={np.mean(random_effects):.4f}, std={np.std(random_effects):.4f}")

    # Statistical test: Mann-Whitney U
    print("\nStatistical test: Mann-Whitney U test (Q3 vs random)")
    stat, p_value = stats.mannwhitneyu(q3_effects, random_effects, alternative='two-sided')

    print(f"U statistic: {stat:.2f}")
    print(f"p-value: {p_value:.6f}")

    # Effect size (rank-biserial correlation)
    n1, n2 = len(q3_effects), len(random_effects)
    rank_biserial = 1 - (2 * stat) / (n1 * n2)
    print(f"Rank-biserial correlation (effect size): {rank_biserial:.4f}")

    # Determine pass/fail
    alpha = 0.05
    if p_value > alpha:
        test_result = "PASS"
        conclusion = "Q3 steering is NOT significantly different from random baseline (p > 0.05)"
    else:
        test_result = "FAIL"
        conclusion = "Q3 steering IS significantly different from random baseline (p < 0.05)"

    print(f"\nTest result: {test_result}")
    print(f"Conclusion: {conclusion}")

    # Compile results
    results = {
        "task_id": "pilot_sensitivity_first",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": SEED,
            "n_q3_features": len(Q3_FEATURES),
            "n_tokens": N_TOKENS,
            "beta": BETA,
            "random_baseline_n": 100
        },
        "q3_steering_effects": {
            "mean": float(np.mean(q3_effects)),
            "std": float(np.std(q3_effects)),
            "min": float(np.min(q3_effects)),
            "max": float(np.max(q3_effects)),
            "n": len(q3_effects),
            "values": [float(e) for e in q3_effects]
        },
        "random_baseline": {
            "mean": float(np.mean(random_effects)),
            "std": float(np.std(random_effects)),
            "min": float(np.min(random_effects)),
            "max": float(np.max(random_effects)),
            "n": len(random_effects)
        },
        "statistical_test": {
            "test": "Mann-Whitney U",
            "U_statistic": float(stat),
            "p_value": float(p_value),
            "rank_biserial_correlation": float(rank_biserial),
            "alpha": alpha
        },
        "test_result": test_result,
        "conclusion": conclusion,
        "pass_criteria": "Q3 steering ≈ random baseline (p > 0.05)",
        "hypothesis": "H-A"
    }

    # Save results
    results_path = f"{RESULTS_DIR}/sensitivity_first_pilot.json"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Summary
    print("\n" + "="*60)
    print("PILOT SENSITIVITY-FIRST RESULTS")
    print("="*60)
    print(f"Q3 features tested: {len(Q3_FEATURES)}")
    print(f"Q3 mean effect: {np.mean(q3_effects):.4f} (std={np.std(q3_effects):.4f})")
    print(f"Random baseline: {np.mean(random_effects):.4f} (std={np.std(random_effects):.4f})")
    print(f"Mann-Whitney U p-value: {p_value:.6f}")
    print(f"Test result: {test_result}")
    print(f"Conclusion: {conclusion}")
    print("="*60)

    return results

if __name__ == "__main__":
    main()