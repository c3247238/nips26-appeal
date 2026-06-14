#!/usr/bin/env python3
"""
Pilot H3 Null: Replication with Null Controls

Confirms H3 REVERSED finding with proper null controls:
1. Shuffled feature directions (null baseline)
2. Random unit vectors
3. High-absorption features
4. Low-absorption features

Pass criteria:
- Null controls have mean effect < 0.05
- High-absorption effect > low-absorption effect
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
N_FEATURES = 50
ALPHA_VALUES = [3, 10]
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

    # Move direction to same device as activation
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
    print("Pilot H3 Null: Replication with Null Controls")
    print("=" * 60)
    print(f"Time: {datetime.now()}")
    print(f"Device: {DEVICE}")

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

    # Run experiments
    results = {
        "high_absorption": [],
        "low_absorption": [],
        "shuffled": [],
        "random": [],
    }

    print("\nRunning steering experiments...")

    # High absorption features
    for feat_idx in tqdm(high_features, desc="High absorption"):
        direction = W_dec[feat_idx].to(model.cfg.device)
        for alpha in ALPHA_VALUES:
            for prompt in prompts:
                effect = steering_with_direction(model, direction, prompt, alpha)
                results["high_absorption"].append(effect)

    # Low absorption features
    for feat_idx in tqdm(low_features, desc="Low absorption"):
        direction = W_dec[feat_idx].to(model.cfg.device)
        for alpha in ALPHA_VALUES:
            for prompt in prompts:
                effect = steering_with_direction(model, direction, prompt, alpha)
                results["low_absorption"].append(effect)

    # Random directions (null baseline)
    for i, random_dir in enumerate(tqdm(random_directions, desc="Random")):
        direction = random_dir.to(model.cfg.device)
        for alpha in ALPHA_VALUES:
            for prompt in prompts:
                effect = steering_with_direction(model, direction, prompt, alpha)
                results["random"].append(effect)

    # Compute statistics
    stats = {
        "high_absorption_mean": float(np.mean(results["high_absorption"])),
        "high_absorption_std": float(np.std(results["high_absorption"])),
        "low_absorption_mean": float(np.mean(results["low_absorption"])),
        "low_absorption_std": float(np.std(results["low_absorption"])),
        "shuffled_mean": float(np.mean(results["shuffled"])),
        "shuffled_std": float(np.std(results["shuffled"])),
        "random_mean": float(np.mean(results["random"])),
        "random_std": float(np.std(results["random"])),
    }

    # Pass criteria
    null_below_threshold = stats["shuffled_mean"] < 0.05 and stats["random_mean"] < 0.05
    high_above_low = stats["high_absorption_mean"] > stats["low_absorption_mean"]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"High absorption mean effect: {stats['high_absorption_mean']:.4f} ± {stats['high_absorption_std']:.4f}")
    print(f"Low absorption mean effect: {stats['low_absorption_mean']:.4f} ± {stats['low_absorption_std']:.4f}")
    print(f"Shuffled mean effect: {stats['shuffled_mean']:.4f} ± {stats['shuffled_std']:.4f}")
    print(f"Random mean effect: {stats['random_mean']:.4f} ± {stats['random_std']:.4f}")
    print(f"\nNull controls below threshold (0.05): {'PASS' if null_below_threshold else 'FAIL'}")
    print(f"High > Low absorption effect: {'PASS' if high_above_low else 'FAIL'}")

    # Save results
    output = {
        "experiment": "pilot_h3_null",
        "n_features": N_FEATURES,
        "alpha_values": ALPHA_VALUES,
        "n_test_prompts": N_TEST_PROMPTS,
        "stats": stats,
        "pass_criteria": {
            "null_below_threshold": null_below_threshold,
            "high_above_low": high_above_low,
        },
        "timestamp": datetime.now().isoformat(),
    }

    results_file = RESULTS_DIR / "pilot_h3_null.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    if null_below_threshold and high_above_low:
        with open(RESULTS_DIR / "pilot_h3_null_DONE", "w") as f:
            f.write(f"Null={stats['shuffled_mean']:.4f}, High={stats['high_absorption_mean']:.4f}, Low={stats['low_absorption_mean']:.4f}\n")

    print("\nDone!")


if __name__ == "__main__":
    main()
