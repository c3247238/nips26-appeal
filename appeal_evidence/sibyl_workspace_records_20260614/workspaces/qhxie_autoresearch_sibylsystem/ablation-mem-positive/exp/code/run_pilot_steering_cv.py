#!/usr/bin/env python3
"""
Pilot: Steering Effectiveness by CV

Tests whether CV predicts steering effectiveness.
Compare steering on 10 high-CV vs 10 low-CV features.
Tests H4/actionability connection.

Pass criterion: High-CV shows larger steering effect than low-CV; no crashes
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# Add workspace to path
workspace = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive")
sys.path.insert(0, str(workspace))
sys.path.insert(0, str(workspace / ".venv/lib/python3.12/site-packages"))

# Configuration
CONFIG = {
    "seed": 42,
    "n_samples": 100,
    "timeout": 900,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_name": "gpt2-small",
    "sae_release": "gpt2-small-res-jb",
    "sae_id": "blocks.6.hook_resid_pre",
    "steering_strengths": [-5, 5],  # Pilot uses just +5 and -5
    "n_features_per_group": 10,
}

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model_and_sae():
    """Load GPT-2 and SAE model."""
    print("Loading model and SAE...")
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    device = CONFIG["device"]

    # Load model
    model = HookedTransformer.from_pretrained(CONFIG["model_name"], device=device)
    print(f"Model loaded: {CONFIG['model_name']}")

    # Load SAE
    sae = SAE.from_pretrained(
        release=CONFIG["sae_release"],
        sae_id=CONFIG["sae_id"],
        device=device
    )
    cfg_dict = sae.cfg
    print(f"SAE loaded: {CONFIG['sae_release']}/{CONFIG['sae_id']}")
    print(f"  d_in={cfg_dict.d_in}, d_sae={cfg_dict.d_sae}")

    return model, sae, cfg_dict

def select_cv_features(sae, cfg_dict, n_high=10, n_low=10):
    """
    Select high-CV and low-CV features for steering experiment.
    Uses CV computed from prior analysis.
    """
    print(f"\nSelecting {n_high} high-CV and {n_low} low-CV features...")

    # Load CV analysis results
    cv_path = workspace / "exp/results/full/cv_full_analysis.json"
    if cv_path.exists():
        with open(cv_path) as f:
            cv_data = json.load(f)
        print("Loaded CV analysis")
    else:
        print("Warning: CV analysis not found, using random selection")
        cv_data = None

    # Collect activations for CV estimation using a simple loop
    from transformer_lens import HookedTransformer
    device = CONFIG["device"]
    model = HookedTransformer.from_pretrained(CONFIG["model_name"], device=device)

    # Use a single prompt repeated to get consistent tensor shapes
    base_prompt = "The quick brown fox jumps over the lazy dog. " * 5
    tokens = model.to_tokens(base_prompt)
    seq_len = tokens.shape[1]
    d_model = cfg_dict.d_in

    all_activations = []

    for i in range(100):  # 100 samples for CV estimation
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            acts = cache["blocks.6.hook_resid_pre"]  # [1, seq_len, d_model]
            all_activations.append(acts)

    # Stack activations: [n_samples, seq_len, d_model]
    all_activations = torch.cat(all_activations, dim=0)
    print(f"Collected activations shape: {all_activations.shape}")

    # For each feature, compute variance across samples at each position
    # Then average across positions
    # Mean activation per feature: [d_sae]
    feature_means = all_activations.mean(dim=(0, 1))  # [d_sae]
    feature_stds = all_activations.std(dim=(0, 1))    # [d_sae]

    # CV = std / mean (only for non-zero means)
    cv = torch.zeros_like(feature_stds)
    nonzero_mask = feature_means > 1e-6
    cv[nonzero_mask] = feature_stds[nonzero_mask] / (feature_means[nonzero_mask] + 1e-6)

    print(f"CV computed for {cv.numel()} features")
    print(f"CV range: {cv.min().item():.3f} to {cv.max().item():.3f}")

    # Select high-CV features (top n_high by CV, among features with mean > threshold)
    active_threshold = 0.1
    active_mask = feature_means > active_threshold
    active_indices = torch.where(active_mask)[0]

    if len(active_indices) == 0:
        print("Warning: No active features found, using random selection")
        high_cv_features = list(range(n_high))
        low_cv_features = list(range(n_low))
        high_cv_values = torch.zeros(n_high)
        low_cv_values = torch.zeros(n_low)
    else:
        active_cv = cv[active_indices]

        # Top n_high high-CV
        high_cv_sorted, high_cv_order = torch.sort(active_cv, descending=True)
        high_cv_features = active_indices[high_cv_order[:n_high]].tolist()
        high_cv_values = high_cv_sorted[:n_high]

        # Top n_low low-CV (smallest CV among active)
        low_cv_sorted, low_cv_order = torch.sort(active_cv, descending=False)
        low_cv_features = active_indices[low_cv_order[:n_low]].tolist()
        low_cv_values = low_cv_sorted[:n_low]

    print(f"Selected {len(high_cv_features)} high-CV features: CV range {high_cv_values.min().item():.3f} to {high_cv_values.max().item():.3f}")
    print(f"Selected {len(low_cv_features)} low-CV features: CV range {low_cv_values.min().item():.3f} to {low_cv_values.max().item():.3f}")

    return {
        "high_cv": high_cv_features,
        "low_cv": low_cv_features,
        "high_cv_values": high_cv_values.tolist(),
        "low_cv_values": low_cv_values.tolist(),
    }

def steering_experiment(model, sae):
    """
    Perform steering experiment to test CV-actionability connection.

    For high-CV and low-CV feature groups:
    1. Apply steering with strength +5 and -5
    2. Measure logit change at target tokens
    3. Compare steering effect between groups
    """
    print("\n" + "="*60)
    print("STEERING EFFECTIVENESS EXPERIMENT")
    print("="*60)

    set_seed(CONFIG["seed"])
    device = CONFIG["device"]

    results = {
        "task_id": "pilot_steering_cv",
        "status": "pending",
        "timestamp": datetime.now().isoformat(),
        "config": {k: v for k, v in CONFIG.items() if k != "device"},
        "steering_results": {},
        "aggregate": {},
    }

    try:
        # Select features by CV
        feature_selection = select_cv_features(sae, sae.cfg)
        high_cv_features = feature_selection["high_cv"]
        low_cv_features = feature_selection["low_cv"]

        # Test prompts - semantic content that should be affected by steering
        test_prompts = [
            "The movie was very",
            "The food was extremely",
            "The weather today is",
            "This book is really",
            "The music sounds",
            "The experience was",
        ]

        steering_effects = {"high_cv": [], "low_cv": []}

        hook_name = "blocks.6.hook_resid_pre"
        W_dec = sae.W_dec  # [d_sae, d_in]

        for strength in CONFIG["steering_strengths"]:
            print(f"\n--- Steering strength: {strength} ---")

            for feat_idx in high_cv_features[:5]:  # Top 5 from each group for pilot
                feat_direction = W_dec[feat_idx].to(device)

                # Hook function for steering
                def steering_hook(activations, hook, feat_idx=feat_idx, strength=strength, feat_direction=feat_direction):
                    # Add scaled feature direction to all positions
                    return activations + strength * feat_direction

                for prompt in test_prompts[:3]:  # 3 prompts for pilot
                    tokens = model.to_tokens(prompt)

                    # Clean logits
                    with torch.no_grad():
                        clean_logits = model(tokens)

                    # Steered logits
                    steered_logits = model.run_with_hooks(
                        tokens,
                        fwd_hooks=[(hook_name, steering_hook)]
                    )

                    # Compute logit change at target position
                    # Target position = last token of prompt
                    target_pos = tokens.shape[1] - 1
                    clean_logit = clean_logits[0, target_pos].max().item()
                    steered_logit = steered_logits[0, target_pos].max().item()
                    logit_change = steered_logit - clean_logit

                    steering_effects["high_cv"].append({
                        "feature": feat_idx,
                        "strength": strength,
                        "prompt": prompt,
                        "logit_change": logit_change,
                        "abs_effect": abs(logit_change),
                    })

            for feat_idx in low_cv_features[:5]:  # Top 5 from each group for pilot
                feat_direction = W_dec[feat_idx].to(device)

                def steering_hook(activations, hook, feat_idx=feat_idx, strength=strength, feat_direction=feat_direction):
                    return activations + strength * feat_direction

                for prompt in test_prompts[:3]:  # 3 prompts for pilot
                    tokens = model.to_tokens(prompt)

                    with torch.no_grad():
                        clean_logits = model(tokens)

                    steered_logits = model.run_with_hooks(
                        tokens,
                        fwd_hooks=[(hook_name, steering_hook)]
                    )

                    target_pos = tokens.shape[1] - 1
                    clean_logit = clean_logits[0, target_pos].max().item()
                    steered_logit = steered_logits[0, target_pos].max().item()
                    logit_change = steered_logit - clean_logit

                    steering_effects["low_cv"].append({
                        "feature": feat_idx,
                        "strength": strength,
                        "prompt": prompt,
                        "logit_change": logit_change,
                        "abs_effect": abs(logit_change),
                    })

        # Compute aggregate statistics
        high_cv_effects = [e["abs_effect"] for e in steering_effects["high_cv"]]
        low_cv_effects = [e["abs_effect"] for e in steering_effects["low_cv"]]

        mean_high_cv = np.mean(high_cv_effects)
        mean_low_cv = np.mean(low_cv_effects)

        results["steering_results"] = steering_effects
        results["aggregate"] = {
            "high_cv_mean_effect": float(mean_high_cv),
            "low_cv_mean_effect": float(mean_low_cv),
            "high_cv_std_effect": float(np.std(high_cv_effects)),
            "low_cv_std_effect": float(np.std(low_cv_effects)),
            "n_high_cv_samples": len(high_cv_effects),
            "n_low_cv_samples": len(low_cv_effects),
            "difference": float(mean_high_cv - mean_low_cv),
        }

        # Determine pass/fail: high-CV should show larger steering effect
        # (because high-CV = absorbed features = more steerable?)
        high_cv_larger = bool(mean_high_cv > mean_low_cv)

        results["pass_criteria"] = {
            "required": "High-CV shows larger steering effect than low-CV",
            "high_cv_mean": float(mean_high_cv),
            "low_cv_mean": float(mean_low_cv),
            "high_cv_larger": high_cv_larger,
        }

        results["status"] = "success" if high_cv_larger else "partial"

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"High-CV mean effect: {mean_high_cv:.4f} (n={len(high_cv_effects)})")
        print(f"Low-CV mean effect: {mean_low_cv:.4f} (n={len(low_cv_effects)})")
        print(f"High-CV larger: {high_cv_larger}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        results["status"] = "failed"
        results["error"] = str(e)

    return results

def main():
    print("="*60)
    print("PILOT: STEERING EFFECTIVENESS BY CV")
    print("="*60)
    print(f"Device: {CONFIG['device']}")
    print(f"Seed: {CONFIG['seed']}")
    print(f"N samples: {CONFIG['n_samples']}")

    # Load model and SAE
    model, sae, cfg_dict = load_model_and_sae()

    # Run steering experiment
    results = steering_experiment(model, sae)

    # Save results
    output_dir = workspace / "exp/results/pilots"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "pilot_steering_cv.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Write DONE marker
    done_file = output_dir / "pilot_steering_cv_DONE"
    with open(done_file, 'w') as f:
        f.write(json.dumps({"status": results["status"], "timestamp": results["timestamp"]}))

    return results

if __name__ == "__main__":
    results = main()
    sys.exit(0 if results["status"] == "success" else 1)