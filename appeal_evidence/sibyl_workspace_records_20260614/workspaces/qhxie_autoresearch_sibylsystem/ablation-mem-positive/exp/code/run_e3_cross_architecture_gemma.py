#!/usr/bin/env python3
"""
E3: Cross-Architecture Validation on Gemma-2-2B (PILOT)

Tests whether CV-steering correlation generalizes across architectures.
Uses Gemma-2-2B JumpReLU SAE (GemmaScope) with architecture-specific median CV split.

NOTE: This experiment requires access to google/gemma-2-2b which is a gated model.
If the model cannot be loaded, this script will report the issue and exit gracefully.
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add workspace to path
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive")
REMOTE_BASE = WORKSPACE
RESULTS_DIR = REMOTE_BASE / "exp" / "results"

# Configuration
TASK_ID = "e3_cross_architecture_gemma"
CONFIG = {
    "seed": 42,
    "n_samples": 100,
    "timeout": 900,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_name": "google/gemma-2-2b",  # Gated model - may not load
    "sae_release": "gemma-scope-2b-pt-res",
    "sae_id": "layer_6/width_16k/average_l0_70",  # JumpReLU SAE
    "steering_strength": 5,
    "n_features_per_group": 30,  # Pilot: 30 per group
    "prompts": [
        "The movie was very",
        "The food was extremely",
        "The weather today is",
        "The book was quite",
        "The experience was"
    ],
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_sae():
    """Load Gemma-2-2B and GemmaScope JumpReLU SAE."""
    print("Loading model and SAE...")

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    device = CONFIG["device"]

    # Load model
    try:
        model = HookedTransformer.from_pretrained(CONFIG["model_name"], device=device)
        print(f"Model loaded: {CONFIG['model_name']}")
    except Exception as e:
        print(f"Error loading model {CONFIG['model_name']}: {e}")
        print("Gemma-2-2B is a gated model and requires HuggingFace authentication.")
        print("Please set HF_TOKEN or use a different access method.")
        raise

    # Load SAE from GemmaScope
    try:
        sae = SAE.from_pretrained(
            release=CONFIG["sae_release"],
            sae_id=CONFIG["sae_id"],
            device=device
        )
        print(f"SAE loaded: {CONFIG['sae_release']}/{CONFIG['sae_id']}")
        print(f"  SAE type: {type(sae).__name__}")
        print(f"  d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
    except Exception as e:
        print(f"Error loading SAE: {e}")
        raise

    return model, sae


def compute_feature_cv(sae, model, n_samples=500):
    """Compute coefficient of variation for each feature."""
    print(f"\nComputing CV for {n_samples} samples...")

    from datasets import load_dataset

    dataset = load_dataset("EleutherAI/pile-val-backup", split="validation", streaming=True)

    feature_means = np.zeros(sae.cfg.d_sae)
    feature_vars = np.zeros(sae.cfg.d_sae)
    sample_count = 0

    texts = [example["text"] for _, example in zip(range(n_samples), dataset)]

    with torch.no_grad():
        for i, text in enumerate(texts[:n_samples]):
            try:
                tokens = model.to_tokens(text)
                if tokens.shape[-1] > 512:
                    tokens = tokens[:, :512]

                _, cache = model.run_with_cache(tokens)
                acts = cache["blocks.6.hook_resid_pre"]  # Layer 6

                # Encode to get SAE features
                sae_acts = sae.encode(acts)  # [batch, seq, d_sae]

                # Compute per-feature statistics across positions
                sae_acts_np = sae_acts[0].cpu().numpy()  # [seq, d_sae]

                # Update running mean and variance
                for pos in range(sae_acts_np.shape[0]):
                    pos_acts = sae_acts_np[pos]
                    sample_count += 1
                    delta = pos_acts - feature_means
                    feature_means += delta / sample_count
                    delta2 = pos_acts - feature_means
                    feature_vars += delta * delta2

            except Exception as e:
                continue

    # Compute CV = std / mean
    feature_stds = np.sqrt(feature_vars / max(sample_count - 1, 1))
    cv_values = np.where(feature_means > 1e-6, feature_stds / (np.abs(feature_means) + 1e-6), 0)

    print(f"CV computed for {len(cv_values)} features")
    print(f"CV range: {cv_values.min():.3f} to {cv_values.max():.3f}")
    print(f"CV median: {np.median(cv_values):.3f}")

    return cv_values, feature_means


def select_features_by_cv(cv_values, feature_means, n_per_group=30):
    """Select high-CV and low-CV features for steering experiment."""
    print(f"\nSelecting {n_per_group} high-CV and {n_per_group} low-CV features...")

    # Use architecture-specific median CV as threshold (prospective)
    median_cv = np.median(cv_values)
    print(f"Architecture-specific median CV: {median_cv:.3f}")

    # Select features with CV above/below median
    high_cv_mask = cv_values > median_cv
    low_cv_mask = cv_values <= median_cv

    # Further filter by activation magnitude (avoid dead features)
    active_threshold = 0.01
    active_mask = feature_means > active_threshold

    high_cv_indices = np.where(high_cv_mask & active_mask)[0]
    low_cv_indices = np.where(low_cv_mask & active_mask)[0]

    # Sort by CV and select top N
    high_cv_sorted = high_cv_indices[np.argsort(cv_values[high_cv_indices])[::-1]]
    low_cv_sorted = low_cv_indices[np.argsort(cv_values[low_cv_indices])]

    n_high = min(n_per_group, len(high_cv_sorted))
    n_low = min(n_per_group, len(low_cv_sorted))

    high_cv_features = high_cv_sorted[:n_high].tolist()
    low_cv_features = low_cv_sorted[:n_low].tolist()

    print(f"Selected {len(high_cv_features)} high-CV features: CV range {cv_values[high_cv_features].min():.3f} to {cv_values[high_cv_features].max():.3f}")
    print(f"Selected {len(low_cv_features)} low-CV features: CV range {cv_values[low_cv_features].min():.3f} to {cv_values[low_cv_features].max():.3f}")

    return {
        "high_cv": high_cv_features,
        "low_cv": low_cv_features,
        "high_cv_cv": cv_values[high_cv_features].tolist(),
        "low_cv_cv": cv_values[low_cv_features].tolist(),
        "median_cv": float(median_cv),
    }


def run_steering_experiment(model, sae, features_by_group, strength=5):
    """Run steering experiment for high-CV vs low-CV features."""
    print(f"\nRunning steering experiment at strength +{strength}...")

    results = []
    hook_name = "blocks.6.hook_resid_pre"

    for group_name, feature_indices in features_by_group.items():
        for feat_idx in feature_indices:
            # Get feature direction
            feature_direction = sae.W_dec[feat_idx].float().to(CONFIG["device"])

            def steering_hook(activations, hook, feat_direction=feature_direction):
                return activations + strength * feat_direction.to(activations.device)

            for prompt in CONFIG["prompts"]:
                try:
                    tokens = model.to_tokens(prompt)
                    if tokens.shape[-1] > 128:
                        tokens = tokens[:, :128]

                    # Clean logits
                    with torch.no_grad():
                        clean_logits = model(tokens)
                        baseline_logit = clean_logits[0, -1, :].cpu()

                    # Steered logits
                    with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
                        steered_logits = model(tokens)
                        steered_logit = steered_logits[0, -1, :].cpu()

                    logit_change = (steered_logit - baseline_logit).max().item()

                    results.append({
                        "feature": int(feat_idx),
                        "group": group_name,
                        "prompt": prompt,
                        "strength": strength,
                        "logit_change": logit_change,
                        "abs_effect": abs(logit_change)
                    })

                except Exception as e:
                    print(f"Error with feature {feat_idx}, prompt '{prompt}': {e}")
                    continue

    return results


def compute_statistics(all_results):
    """Compute statistics for high-CV vs low-CV steering effects."""
    high_cv_results = [r for r in all_results if r["group"] == "high_cv"]
    low_cv_results = [r for r in all_results if r["group"] == "low_cv"]

    high_cv_abs = [r["abs_effect"] for r in high_cv_results]
    low_cv_abs = [r["abs_effect"] for r in low_cv_results]

    high_cv_raw = [r["logit_change"] for r in high_cv_results]
    low_cv_raw = [r["logit_change"] for r in low_cv_results]

    from scipy import stats as scipy_stats

    if len(high_cv_abs) > 0 and len(low_cv_abs) > 0:
        t_stat, p_val = scipy_stats.ttest_ind(high_cv_abs, low_cv_abs, equal_var=False)
        p_one_sided = p_val / 2 if t_stat > 0 else 1 - p_val / 2
    else:
        t_stat, p_one_sided = 0, 1

    mean_high = np.mean(high_cv_abs) if high_cv_abs else 0
    mean_low = np.mean(low_cv_abs) if low_cv_abs else 0

    return {
        "high_cv_mean_abs_effect": float(mean_high),
        "low_cv_mean_abs_effect": float(mean_low),
        "high_cv_std": float(np.std(high_cv_abs)) if high_cv_abs else 0,
        "low_cv_std": float(np.std(low_cv_abs)) if low_cv_abs else 0,
        "n_high_cv": len(high_cv_abs),
        "n_low_cv": len(low_cv_abs),
        "t_statistic": float(t_stat),
        "p_value_one_sided": float(p_one_sided),
        "significant_05": bool(p_one_sided < 0.05),
        "significant_01": bool(p_one_sided < 0.01),
        "effect_ratio": float(mean_high / mean_low) if mean_low > 0 else 0,
    }


def main():
    print("=" * 60)
    print("E3: CROSS-ARCHITECTURE VALIDATION ON GEMMA-2-2B (PILOT)")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Seed: {CONFIG['seed']}")
    print(f"Model: {CONFIG['model_name']}")
    print(f"SAE: {CONFIG['sae_release']}/{CONFIG['sae_id']}")

    set_seed(CONFIG["seed"])

    # Try to load model and SAE
    try:
        model, sae = load_model_and_sae()
    except Exception as e:
        # Write failure result
        error_result = {
            "task_id": TASK_ID,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "config": {k: v for k, v in CONFIG.items() if k != "device"},
        }

        output_path = RESULTS_DIR / f"{TASK_ID}.json"
        with open(output_path, "w") as f:
            json.dump(error_result, f, indent=2)

        print(f"\nError result saved to: {output_path}")
        print(f"ERROR: {e}")
        return error_result

    # Compute CV for all features
    cv_values, feature_means = compute_feature_cv(sae, model, n_samples=500)

    # Select high-CV and low-CV features
    features_by_group = select_features_by_cv(cv_values, feature_means, n_per_group=CONFIG["n_features_per_group"])

    # Run steering experiment
    all_results = run_steering_experiment(model, sae, features_by_group, strength=CONFIG["steering_strength"])
    print(f"\nCollected {len(all_results)} steering measurements")

    # Compute statistics
    stats = compute_statistics(all_results)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"High-CV mean abs effect: {stats['high_cv_mean_abs_effect']:.4f} (n={stats['n_high_cv']})")
    print(f"Low-CV mean abs effect:  {stats['low_cv_mean_abs_effect']:.4f} (n={stats['n_low_cv']})")
    print(f"Effect ratio: {stats['effect_ratio']:.2f}x")
    print(f"t-statistic: {stats['t_statistic']:.4f}, p (one-sided): {stats['p_value_one_sided']:.6f}")
    print(f"Significant at 0.05: {stats['significant_05']}, at 0.01: {stats['significant_01']}")

    # Determine finding
    if stats['significant_05'] and stats['high_cv_mean_abs_effect'] > stats['low_cv_mean_abs_effect']:
        finding = "CONFIRMED"
        interpretation = "CV-steering correlation replicates on Gemma-2-2B JumpReLU SAE"
    elif stats['high_cv_mean_abs_effect'] > stats['low_cv_mean_abs_effect']:
        finding = "INCONCLUSIVE"
        interpretation = "Trend observed but not significant"
    else:
        finding = "NOT_CONFIRMED"
        interpretation = "Low-CV features show larger steering effect (opposite of GPT-2)"

    # Save results
    output = {
        "task_id": TASK_ID,
        "status": "success",
        "finding": finding,
        "interpretation": interpretation,
        "timestamp": datetime.now().isoformat(),
        "config": {k: v for k, v in CONFIG.items() if k != "device"},
        "feature_selection": features_by_group,
        "statistics": stats,
        "all_results": all_results[:100],  # Save first 100 for inspection
        "gpu": {
            "id": 0,
            "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        }
    }

    output_path = RESULTS_DIR / f"{TASK_ID}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"\nFINAL FINDING: {finding}")
    print(f"INTERPRETATION: {interpretation}")

    # Write DONE marker
    DONE_path = RESULTS_DIR / f"{TASK_ID}_DONE"
    DONE_path.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": "success" if finding == "CONFIRMED" else "partial",
        "timestamp": datetime.now().isoformat(),
        "finding": finding,
        "interpretation": interpretation,
    }))

    # Clean up PID file if exists
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()

    return output


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result.get("status") == "success" else 1)