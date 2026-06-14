#!/usr/bin/env python3
"""
Full H3 Null: Null Controls with Full Alpha Range [1,3,5,10,20]

CRITICAL FIX: Original pilot_h3_null used alpha=[3,10] only.
This experiment uses alpha=[1,3,5,10,20] to MATCH the main H3 protocol.

Purpose:
- Null baseline: random unit vectors
- Control: high-absorption vs low-absorption features
- All alpha values tested to check if effect is alpha-dependent

Pass criteria:
- Random directions show significantly lower effect than feature directions
- High-absorption features show higher effect than low-absorption features
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr, ttest_ind
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformer_lens import HookedTransformer
from sae_lens import SAE

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
DEVICE = "cuda:0"
SEED = 42
N_FEATURES = 50
# CRITICAL: Use SAME alpha values as main H3 experiment
ALPHA_VALUES = [1, 3, 5, 10, 20]
N_TEST_PROMPTS = 10

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def load_model_and_sae(device: str):
    """Load GPT-2 Small and SAE for layer 8."""
    print("Loading GPT-2 Small model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    print("Loading SAE for layer 8...")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        device=device
    )

    return model, sae


def get_test_prompts() -> List[str]:
    """Diverse test prompts."""
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


def compute_uas_scores(sae, model, prompts: List[str]) -> Dict[int, float]:
    """Compute UAS scores for features."""
    print("Computing UAS scores...")

    all_features = []
    for prompt in tqdm(prompts, desc="Collecting activations"):
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)
        activations = cache["resid_pre", 8]
        features = sae.encode(activations)
        all_features.append(features.cpu())

    all_features = torch.cat(all_features, dim=1)[0]
    d_sae = all_features.shape[1]

    W_dec = sae.W_dec.detach().cpu()
    uas_scores = {}

    for feat_idx in tqdm(range(min(d_sae, 10000)), desc="Computing UAS"):
        feat_dir = W_dec[feat_idx]
        cos_sims = torch.cosine_similarity(feat_dir.unsqueeze(0), W_dec, dim=1)
        cos_variance = cos_sims.var().item()
        act_freq = (all_features[:, feat_idx] > 0).float().mean().item()
        uas = cos_variance * 1.0 + act_freq * 0.5
        uas_scores[feat_idx] = uas

    return uas_scores


def steering_with_direction(model, direction, prompt: str, alpha: float) -> float:
    """Perform steering with given direction."""
    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        original_logits = model(tokens, return_type="logits")

    direction_device = direction.to(model.cfg.device)

    def steering_hook(activation, hook):
        return activation + alpha * direction_device

    with model.hooks(fwd_hooks=[("blocks.8.hook_resid_pre", steering_hook)]):
        with torch.no_grad():
            steered_logits = model(tokens, return_type="logits")

    effect = torch.abs(steered_logits - original_logits).max().item()
    return effect


def main():
    print("=" * 60)
    print("Full H3 Null: Null Controls with Full Alpha Range")
    print("=" * 60)
    print(f"Time: {datetime.now()}")
    print(f"Device: {DEVICE}")
    print(f"Alpha values: {ALPHA_VALUES}")  # FIXED: Same as main H3

    # Load model and SAE
    model, sae = load_model_and_sae(DEVICE)
    prompts = get_test_prompts()

    # Compute UAS scores
    uas_scores = compute_uas_scores(sae, model, prompts)

    # Select features
    sorted_features = sorted(uas_scores.items(), key=lambda x: x[1], reverse=True)
    high_features = [f for f, _ in sorted_features[:N_FEATURES]]
    low_features = [f for f, _ in sorted_features[-N_FEATURES:]]

    print(f"\nSelected {N_FEATURES} high and {N_FEATURES} low absorption features")

    # Get feature directions
    W_dec = sae.W_dec.detach().cpu()
    W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)  # Normalize

    # Create null controls - use random unit vectors
    n_null = N_FEATURES
    d_model = W_dec.shape[1]
    random_directions = []
    for _ in range(n_null):
        random_dir = torch.randn(d_model)
        random_dir = random_dir / random_dir.norm()
        random_directions.append(random_dir)

    # Run experiments - collect per-alpha data
    results = {
        "high_absorption": {alpha: [] for alpha in ALPHA_VALUES},
        "low_absorption": {alpha: [] for alpha in ALPHA_VALUES},
        "random": {alpha: [] for alpha in ALPHA_VALUES},
    }

    print("\nRunning steering experiments...")

    # High absorption features
    for feat_idx in tqdm(high_features, desc="High absorption"):
        direction = W_dec[feat_idx].to(model.cfg.device)
        for alpha in ALPHA_VALUES:
            for prompt in prompts:
                effect = steering_with_direction(model, direction, prompt, alpha)
                results["high_absorption"][alpha].append(effect)

    # Low absorption features
    for feat_idx in tqdm(low_features, desc="Low absorption"):
        direction = W_dec[feat_idx].to(model.cfg.device)
        for alpha in ALPHA_VALUES:
            for prompt in prompts:
                effect = steering_with_direction(model, direction, prompt, alpha)
                results["low_absorption"][alpha].append(effect)

    # Random directions (null baseline)
    for i, random_dir in enumerate(tqdm(random_directions, desc="Random")):
        direction = random_dir.to(model.cfg.device)
        for alpha in ALPHA_VALUES:
            for prompt in prompts:
                effect = steering_with_direction(model, direction, prompt, alpha)
                results["random"][alpha].append(effect)

    # Compute statistics per alpha
    per_alpha_stats = {}
    for alpha in ALPHA_VALUES:
        high_mean = np.mean(results["high_absorption"][alpha])
        high_std = np.std(results["high_absorption"][alpha])
        low_mean = np.mean(results["low_absorption"][alpha])
        low_std = np.std(results["low_absorption"][alpha])
        rand_mean = np.mean(results["random"][alpha])
        rand_std = np.std(results["random"][alpha])

        # T-tests
        high_vs_random_t, high_vs_random_p = ttest_ind(
            results["high_absorption"][alpha], results["random"][alpha]
        )
        high_vs_low_t, high_vs_low_p = ttest_ind(
            results["high_absorption"][alpha], results["low_absorption"][alpha]
        )

        per_alpha_stats[alpha] = {
            "high_mean": float(high_mean),
            "high_std": float(high_std),
            "low_mean": float(low_mean),
            "low_std": float(low_std),
            "random_mean": float(rand_mean),
            "random_std": float(rand_std),
            "high_vs_random_t": float(high_vs_random_t),
            "high_vs_random_p": float(high_vs_random_p),
            "high_vs_low_t": float(high_vs_low_t),
            "high_vs_low_p": float(high_vs_low_p),
        }

    # Aggregate statistics (all alphas combined)
    all_high = [e for alpha in ALPHA_VALUES for e in results["high_absorption"][alpha]]
    all_low = [e for alpha in ALPHA_VALUES for e in results["low_absorption"][alpha]]
    all_random = [e for alpha in ALPHA_VALUES for e in results["random"][alpha]]

    stats = {
        "high_absorption_mean": float(np.mean(all_high)),
        "high_absorption_std": float(np.std(all_high)),
        "low_absorption_mean": float(np.mean(all_low)),
        "low_absorption_std": float(np.std(all_low)),
        "random_mean": float(np.mean(all_random)),
        "random_std": float(np.std(all_random)),
        "per_alpha": per_alpha_stats,
    }

    # T-tests on aggregated
    high_vs_random_t, high_vs_random_p = ttest_ind(all_high, all_random)
    high_vs_low_t, high_vs_low_p = ttest_ind(all_high, all_low)

    stats["high_vs_random_p"] = float(high_vs_random_p)
    stats["high_vs_low_p"] = float(high_vs_low_p)

    # Effect ratio
    effect_ratio = stats["high_absorption_mean"] / stats["low_absorption_mean"]
    stats["effect_ratio_high_low"] = float(effect_ratio)

    print("\n" + "=" * 60)
    print("RESULTS (Aggregated)")
    print("=" * 60)
    print(f"High absorption mean effect: {stats['high_absorption_mean']:.4f} ± {stats['high_absorption_std']:.4f}")
    print(f"Low absorption mean effect: {stats['low_absorption_mean']:.4f} ± {stats['low_absorption_std']:.4f}")
    print(f"Random mean effect: {stats['random_mean']:.4f} ± {stats['random_std']:.4f}")
    print(f"Effect ratio (high/low): {effect_ratio:.4f}")
    print(f"High vs Random p-value: {stats['high_vs_random_p']:.6f}")
    print(f"High vs Low p-value: {stats['high_vs_low_p']:.6f}")

    print("\n" + "=" * 60)
    print("RESULTS (Per Alpha)")
    print("=" * 60)
    for alpha in ALPHA_VALUES:
        s = per_alpha_stats[alpha]
        print(f"\nAlpha={alpha}:")
        print(f"  High: {s['high_mean']:.4f} ± {s['high_std']:.4f}")
        print(f"  Low:  {s['low_mean']:.4f} ± {s['low_std']:.4f}")
        print(f"  Random: {s['random_mean']:.4f} ± {s['random_std']:.4f}")
        print(f"  High vs Random p={s['high_vs_random_p']:.6f}")
        print(f"  High vs Low p={s['high_vs_low_p']:.6f}")

    # Save results
    output = {
        "experiment": "full_h3_null_fixed",
        "description": "Null controls with full alpha range [1,3,5,10,20] matching main H3",
        "n_features": N_FEATURES,
        "alpha_values": ALPHA_VALUES,
        "n_test_prompts": N_TEST_PROMPTS,
        "stats": stats,
        "timestamp": datetime.now().isoformat(),
    }

    results_file = RESULTS_DIR / "full_h3_null_fixed.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    with open(RESULTS_DIR / "full_h3_null_fixed_DONE", "w") as f:
        f.write(f"Aggregated: High={stats['high_absorption_mean']:.4f}, Low={stats['low_absorption_mean']:.4f}, Random={stats['random_mean']:.4f}\n")
        f.write(f"Effect ratio (high/low): {effect_ratio:.4f}\n")

    print("\nDone!")


if __name__ == "__main__":
    main()
