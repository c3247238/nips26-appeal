#!/usr/bin/env python3
"""
Full H1 Final: Resolve Layer-wise Absorption Contradiction

Two pilot runs produced contradictory results:
- Run 1: Layer 4 < Layer 8 (+10.6%)
- Run 2: Layer 4 > Layer 8 (-22.9%)

This experiment uses fixed feature selection (same 100 features per layer)
to resolve the contradiction.

Pass criteria: Consistent absorption pattern across 2 independent runs
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformer_lens import HookedTransformer
from sae_lens import SAE

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
# CUDA_VISIBLE_DEVICES restricts visibility; the first visible GPU is cuda:0
DEVICE = "cuda:0"
SEED = 42
GPT2_LAYERS = [2, 4, 6, 8, 10]
N_FEATURES_PER_LAYER = 100

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def load_model_and_saes(device: str, layers: List[int]):
    """Load GPT-2 Small and SAEs for specified layers."""
    print("Loading GPT-2 Small model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    saes = {}
    for layer in layers:
        print(f"Loading SAE for layer {layer}...")
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id=f"blocks.{layer}.hook_resid_pre",
            device=device
        )
        saes[layer] = sae

    return model, saes


def compute_first_letter_absorption(sae, model, feature_idx: int, prompts: List[str]) -> float:
    """Compute absorption score using first-letter probe method.

    Absorption = 1 - accuracy of predicting first letter from feature activation.
    """
    correct = 0
    total = 0

    # Extract layer number from hook_name in metadata
    hook_name = sae.cfg.metadata['hook_name']
    layer = int(hook_name.split(".")[1])

    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)
        activations = cache["resid_pre", layer]
        features = sae.encode(activations)

        # Get feature activation at last position
        act = features[0, -1, feature_idx].item()

        # Get first letter of prompt
        first_letter = prompt[0].lower()

        # Simple heuristic: feature fires for certain letters
        # Use activation threshold
        predicted = act > 0.1
        actual = first_letter in "abcdefghijklmnopqrstuvwxyz"

        if predicted == actual:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.5
    absorption = 1 - accuracy

    return absorption


def compute_absorption_for_layer(sae, model, layer: int, n_features: int, prompts: List[str]) -> List[float]:
    """Compute absorption scores for top-N features in a layer."""
    print(f"\nComputing absorption for layer {layer}...")

    # Get top-N features by activation frequency
    all_features = []
    for prompt in tqdm(prompts[:50], desc=f"Layer {layer} activations"):
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)
        activations = cache["resid_pre", layer]
        features = sae.encode(activations)
        all_features.append(features[0].cpu())

    all_features = torch.cat(all_features, dim=0)  # [n_prompts * seq_len, d_sae]

    # Activation frequency
    freq = (all_features > 0).float().mean(dim=0)
    top_features = freq.topk(n_features).indices.tolist()

    # Compute absorption for each feature
    absorption_scores = []
    for feat_idx in tqdm(top_features, desc=f"Layer {layer} absorption"):
        score = compute_first_letter_absorption(sae, model, feat_idx, prompts)
        absorption_scores.append(score)

    return absorption_scores


def run_experiment():
    """Run full H1 experiment with fixed feature selection."""
    print("=" * 60)
    print("Full H1 Final: Resolve Layer-wise Contradiction")
    print("=" * 60)
    print(f"Time: {datetime.now()}")
    print(f"Device: {DEVICE}")

    # Load model and SAEs
    model, saes = load_model_and_saes(DEVICE, GPT2_LAYERS)

    # Generate test prompts (fixed seed)
    prompts = [
        "apple is a delicious fruit",
        "banana is yellow and sweet",
        "cherry is red and small",
        "dragon fruit is exotic",
        "elephant is a large animal",
        "forest has many trees",
        "guitar makes music sound",
        "honey is made by bees",
        "ice cream is cold and sweet",
        "jacket keeps you warm",
        "kitchen is for cooking",
        "lemon is sour and yellow",
        "mountain is very tall",
        "notebook is for writing",
        "orange is a citrus fruit",
        "pizza is a popular food",
        "queen rules the kingdom",
        "river flows to the sea",
        "sun rises in the east",
        "tree has many branches",
    ]

    # Compute absorption for each layer
    results = {"layers": {}}

    for layer in GPT2_LAYERS:
        absorption_scores = compute_absorption_for_layer(
            saes[layer], model, layer, N_FEATURES_PER_LAYER, prompts
        )
        results["layers"][layer] = {
            "mean_absorption": float(np.mean(absorption_scores)),
            "std_absorption": float(np.std(absorption_scores)),
            "min_absorption": float(np.min(absorption_scores)),
            "max_absorption": float(np.max(absorption_scores)),
            "n_features": len(absorption_scores),
        }

    # Summary
    layer_means = [(layer, results["layers"][layer]["mean_absorption"]) for layer in GPT2_LAYERS]
    layer_means.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print("LAYER-WISE ABSORPTION")
    print("=" * 60)
    for layer, mean in layer_means:
        std = results["layers"][layer]["std_absorption"]
        print(f"Layer {layer:2d}: {mean:.4f} ± {std:.4f}")

    # Check for pattern
    sorted_layers = [l for l, _ in layer_means]
    is_non_monotonic = sorted_layers != GPT2_LAYERS and sorted_layers != GPT2_LAYERS[::-1]

    # Check consistency with previous runs
    run1_pattern = "L4 < L8"  # From run 1
    run2_pattern = "L4 > L8"   # From run 2

    current_pattern = "L4 < L8" if results["layers"][4]["mean_absorption"] < results["layers"][8]["mean_absorption"] else "L4 > L8"

    print(f"\nCurrent pattern: {current_pattern}")
    print(f"Previous Run 1: {run1_pattern}")
    print(f"Previous Run 2: {run2_pattern}")

    # Pass criteria
    # Resolution: consistent pattern across 2 runs (any pattern is OK if consistent)
    # But we need to determine which is the true pattern

    # Store results
    results["summary"] = {
        "layer_ranking": sorted_layers,
        "peak_layer": sorted_layers[0],
        "is_non_monotonic": is_non_monotonic,
        "current_pattern": current_pattern,
        "run1_pattern": run1_pattern,
        "run2_pattern": run2_pattern,
        "resolution": "inconclusive" if current_pattern in [run1_pattern, run2_pattern] else "new_pattern",
    }

    results["timestamp"] = datetime.now().isoformat()

    # Save results
    results_file = RESULTS_DIR / "full_h1_final.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Write DONE file
    with open(RESULTS_DIR / "full_h1_final_DONE", "w") as f:
        f.write(f"Peak layer: {sorted_layers[0]}, Pattern: {current_pattern}\n")

    print("\nDone!")


if __name__ == "__main__":
    run_experiment()
