#!/usr/bin/env python3
"""
Full H3 Gemma: Replicate H3 Steering Signature on Gemma-2B

Replicates the H3 REVERSED finding on Gemma-2B layer 12.
High-absorption features should show HIGHER steering sensitivity.

Pass criteria: Spearman r > 0.2 between UAS and steering sensitivity
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformer_lens import HookedTransformer
from sae_lens import SAE

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
DEVICE = "cuda:0"
SEED = 42
N_HIGH_ABSORPTION = 50
N_LOW_ABSORPTION = 50
ALPHA_VALUES = [1, 3, 5, 10, 20]
N_TEST_PROMPTS = 10
UAS_HIGH_THRESHOLD = 1.0
UAS_LOW_THRESHOLD = 0.3

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def load_model_and_sae(device: str):
    """Load Gemma-2B and SAE for layer 12."""
    print("Loading Gemma-2B model...")
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b-it",
        device=device,
        dtype=torch.bfloat16
    )

    print("Loading SAE for layer 12...")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gemma-2b-res",
        sae_id="blocks.12.hook_resid_pre",
        device=device
    )

    return model, sae


def get_test_prompts() -> List[str]:
    """Diverse test prompts for steering."""
    return [
        "The capital of France is",
        "The CEO decided to",
        "In the morning, she drinks",
        "The experiment proved that",
        "After working hard, he",
        "The weather today is",
        "Scientists discovered that",
        "The book contains many",
        "She studied late into the",
        "The solution is",
    ]


def compute_uas_scores(sae, model, prompts: List[str], n_tokens: int = 500) -> Dict[int, float]:
    """Compute UAS scores using feature geometry."""
    print("Computing UAS scores...")

    all_features = []
    for prompt in tqdm(prompts, desc="Collecting activations"):
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)
        activations = cache["resid_pre", 12]
        features = sae.encode(activations)
        all_features.append(features.cpu())

    all_features = torch.cat(all_features, dim=1)[0]
    d_sae = all_features.shape[1]

    # Compute UAS: cos_sim_variance + freq_skewness
    uas_scores = {}
    W_dec = sae.W_dec.detach().cpu()

    for feat_idx in tqdm(range(min(d_sae, 10000)), desc="Computing UAS"):
        # Feature direction
        feat_dir = W_dec[feat_idx]

        # Cosine similarity with other features
        cos_sims = torch.cosine_similarity(
            feat_dir.unsqueeze(0), W_dec, dim=1
        )
        cos_variance = cos_sims.var().item()

        # Activation frequency skewness
        act_freq = (all_features[:, feat_idx] > 0).float().mean().item()

        # UAS formula
        uas = cos_variance * 1.0 + act_freq * 0.5
        uas_scores[feat_idx] = uas

    return uas_scores


def select_features_by_uas(uas_scores: Dict[int, float], n_high: int, n_low: int) -> Tuple[List[int], List[int]]:
    """Select high and low absorption features by UAS."""
    sorted_features = sorted(uas_scores.items(), key=lambda x: x[1], reverse=True)

    high_features = [f for f, score in sorted_features[:n_high]]
    low_features = [f for f, score in sorted_features[-n_low:]]

    return high_features, low_features


def steering_experiment(model, sae, feature_idx: int, prompt: str, alpha: float) -> float:
    """Perform steering experiment and return effect magnitude."""
    tokens = model.to_tokens(prompt)

    # Get original logits
    with torch.no_grad():
        original_logits = model(tokens, return_type="logits")

    # Get feature direction
    feature_direction = sae.W_dec[feature_idx].to(model.cfg.device)

    # Apply steering
    def steering_hook(activation, hook):
        activation = activation + alpha * feature_direction
        return activation

    with model.hooks(fwd_hooks=[("blocks.12.hook_resid_pre", steering_hook)]):
        with torch.no_grad():
            steered_logits = model(tokens, return_type="logits")

    # Measure effect
    effect = torch.abs(steered_logits - original_logits).max().item()

    return effect


def run_steering_experiments(model, sae, features: List[int], prompts: List[str], alphas: List[float]) -> Dict[int, List[float]]:
    """Run steering experiments for all features."""
    results = {}

    for feat_idx in tqdm(features, desc="Steering experiments"):
        effects = []
        for alpha in alphas:
            for prompt in prompts:
                try:
                    effect = steering_experiment(model, sae, feat_idx, prompt, alpha)
                    effects.append(effect)
                except Exception as e:
                    print(f"Error with feature {feat_idx}, alpha {alpha}: {e}")
                    effects.append(0.0)

        results[feat_idx] = effects

    return results


def main():
    print("=" * 60)
    print("Full H3 Gemma: Steering Signature Replication")
    print("=" * 60)
    print(f"Time: {datetime.now()}")
    print(f"Device: {DEVICE}")

    # Load model and SAE
    model, sae = load_model_and_sae(DEVICE)

    # Test prompts
    prompts = get_test_prompts()

    # Compute UAS scores
    uas_scores = compute_uas_scores(sae, model, prompts)

    # Select features
    high_features, low_features = select_features_by_uas(
        uas_scores, N_HIGH_ABSORPTION, N_LOW_ABSORPTION
    )

    print(f"\nSelected {len(high_features)} high-absorption and {len(low_features)} low-absorption features")

    # Run steering experiments
    high_results = run_steering_experiments(model, sae, high_features, prompts, ALPHA_VALUES)
    low_results = run_steering_experiments(model, sae, low_features, prompts, ALPHA_VALUES)

    # Compute statistics
    high_mean = np.mean([np.mean(v) for v in high_results.values()])
    low_mean = np.mean([np.mean(v) for v in low_results.values()])

    # Correlation between UAS and steering sensitivity
    all_features = high_features + low_features
    all_uas = [uas_scores[f] for f in all_features]
    all_sensitivity = [np.mean(high_results.get(f, low_results.get(f, [0]))) for f in all_features]

    spearman_r, p_value = spearmanr(all_uas, all_sensitivity)

    # Results
    results = {
        "experiment": "full_h3_gemma",
        "model": "gemma-2-2b-it",
        "layer": 12,
        "n_high_absorption": len(high_features),
        "n_low_absorption": len(low_features),
        "high_absorption_mean_effect": high_mean,
        "low_absorption_mean_effect": low_mean,
        "ratio": low_mean / high_mean if high_mean > 0 else 0,
        "spearman_r": spearman_r,
        "p_value": p_value,
        "alpha_values": ALPHA_VALUES,
        "n_test_prompts": len(prompts),
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    results_file = RESULTS_DIR / "full_h3_gemma.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"High-absorption mean effect: {high_mean:.4f}")
    print(f"Low-absorption mean effect: {low_mean:.4f}")
    print(f"Ratio (low/high): {results['ratio']:.4f}")
    print(f"Spearman r: {spearman_r:.4f} (p={p_value:.4e})")
    print(f"\nResults saved to: {results_file}")

    # Pass criteria check
    pass_criteria = spearman_r > 0.2
    print(f"\nPass criteria (r > 0.2): {'PASS' if pass_criteria else 'FAIL'}")

    if pass_criteria:
        with open(RESULTS_DIR / "full_h3_gemma_DONE", "w") as f:
            f.write(f"Spearman r={spearman_r:.4f}, p={p_value:.4e}\n")

    print("\nDone!")


if __name__ == "__main__":
    main()
