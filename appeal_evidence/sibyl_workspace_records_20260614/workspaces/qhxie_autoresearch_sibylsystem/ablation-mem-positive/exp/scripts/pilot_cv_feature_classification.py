#!/usr/bin/env python3
"""
Pilot: CV Feature Classification
Classify features into high-CV (CV > 1.0) and low-CV (CV <= 1.0) groups on GPT-2 layer 6 SAE.
Target: 30+ features in each group.

Output: exp/results/pilot_cv_classification.json
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

# Setup
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive")
RESULTS_DIR = WORKSPACE / "exp" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Ensure we use the correct Python environment
print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import sae_lens and transformer_lens
from transformer_lens import HookedTransformer
from sae_lens import SAE

def compute_cv_per_feature(activations, eps=1e-10):
    """Compute coefficient of variation (CV = std/mean) per feature across samples."""
    # activations: shape [n_samples, n_features]
    mean = activations.mean(dim=0)  # [n_features]
    std = activations.std(dim=0)    # [n_features]
    cv = std / (mean + eps)
    return cv.cpu().numpy()

def main():
    print("\n" + "="*60)
    print("PILOT: CV Feature Classification")
    print("="*60)

    # Load GPT-2 model
    print("\n[1/5] Loading GPT-2 Small model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    print(f"  Model: GPT-2 Small, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Load SAE (GPT-2 layer 6 residual stream)
    print("\n[2/5] Loading GPT-2 Layer 6 SAE...")
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.6.hook_resid_pre",
        device=str(device)
    )
    print(f"  SAE: gpt2-small-res-jb, layer 6, d_sae={sae.cfg.d_sae}")

    # Get dataset (openwebtext - 1000 samples)
    print("\n[3/5] Loading dataset (1000 samples)...")
    from datasets import load_dataset

    try:
        dataset = load_dataset("openwebtext", split="train", streaming=True)
        text_samples = []
        for i, ex in enumerate(dataset):
            if i >= 1000:
                break
            text_samples.append(ex["text"][:500])  # Truncate to 500 chars for speed
    except Exception as e:
        print(f"  Warning: Could not load openwebtext, using synthetic prompts: {e}")
        text_samples = ["The capital of France is", "I walked down the street to",
                       "The theory of relativity was", "In the beginning,"] * 250
        text_samples = text_samples[:1000]

    print(f"  Loaded {len(text_samples)} text samples")

    # Tokenize and compute activations
    print("\n[4/5] Computing activations for CV analysis...")
    all_activations = []

    # Process in batches
    batch_size = 32
    n_batches = (len(text_samples) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(text_samples))
        batch_texts = text_samples[start_idx:end_idx]

        # Tokenize
        tokens = model.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        tokens = tokens.input_ids.to(device)

        # Run model and get residuals
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=["blocks.6.hook_resid_pre"]
            )
            resid_pre = cache["blocks.6.hook_resid_pre"]  # [batch, seq, d_model]

            # Get SAE activations
            sae_acts = sae.encode(resid_pre)  # [batch, seq, d_sae]
            # Take max over sequence dimension for feature detection
            sae_acts_max = sae_acts.amax(dim=1)  # [batch, d_sae]
            all_activations.append(sae_acts_max.cpu())

    # Concatenate all activations
    all_activations = torch.cat(all_activations, dim=0)  # [n_samples, d_sae]
    print(f"  Activation matrix shape: {all_activations.shape}")

    # Compute CV for all features
    print("\n[5/5] Classifying features by CV...")
    cv_scores = compute_cv_per_feature(all_activations)

    # Use absorption score based on proportion of non-zero activations
    # absorption_score > 0.0 means feature fires in at least one sample
    sae_max = all_activations  # already max over seq
    absorption_scores = (sae_max > 0.0).float().mean(dim=0).cpu().numpy()

    # Filter to absorbed features (feature is "active" in >10% of samples)
    # Lower threshold to get enough features for comparison
    absorbed_mask = absorption_scores > 0.1
    n_absorbed = absorbed_mask.sum()
    print(f"  Absorbed features (active in >10% of samples): {n_absorbed}")

    # Create masked arrays for absorbed features
    absorbed_indices = np.where(absorbed_mask)[0]
    absorbed_cv = cv_scores[absorbed_mask]

    # Classify into high-CV (CV > 1.0) and low-CV (CV <= 1.0)
    high_cv_bool = absorbed_cv > 1.0
    low_cv_bool = absorbed_cv <= 1.0

    high_cv_indices = absorbed_indices[high_cv_bool]
    low_cv_indices = absorbed_indices[low_cv_bool]

    n_high_cv = len(high_cv_indices)
    n_low_cv = len(low_cv_indices)

    print(f"  High-CV (CV > 1.0): {n_high_cv} features")
    print(f"  Low-CV (CV <= 1.0): {n_low_cv} features")

    # Get top 30 by CV for each group (or all if fewer than 30)
    # Sort high CV by descending CV value
    if n_high_cv > 0:
        high_cv_sorted = high_cv_indices[np.argsort(-absorbed_cv[high_cv_bool])]
        top_high_cv_indices = high_cv_sorted[:30]
        high_cv_values = absorbed_cv[high_cv_bool][np.argsort(-absorbed_cv[high_cv_bool])[:30]]
    else:
        top_high_cv_indices = np.array([], dtype=np.int64)
        high_cv_values = np.array([])

    # Sort low CV by ascending CV value (lowest first)
    if n_low_cv > 0:
        low_cv_sorted = low_cv_indices[np.argsort(absorbed_cv[low_cv_bool])]
        top_low_cv_indices = low_cv_sorted[:30]
        low_cv_values = absorbed_cv[low_cv_bool][np.argsort(absorbed_cv[low_cv_bool])[:30]]
    else:
        top_low_cv_indices = np.array([], dtype=np.int64)
        low_cv_values = np.array([])

    # Compute group statistics
    high_cv_mean = float(absorbed_cv[high_cv_bool].mean()) if n_high_cv > 0 else 0.0
    high_cv_std = float(absorbed_cv[high_cv_bool].std()) if n_high_cv > 0 else 0.0
    low_cv_mean = float(absorbed_cv[low_cv_bool].mean()) if n_low_cv > 0 else 0.0
    low_cv_std = float(absorbed_cv[low_cv_bool].std()) if n_low_cv > 0 else 0.0

    # Prepare results
    results = {
        "task_id": "pilot_cv_feature_classification",
        "timestamp": datetime.now().isoformat(),
        "pass": n_high_cv >= 30 and n_low_cv >= 30,
        "pass_criteria": {
            "required": "At least 30 features in each CV group (high/low)",
            "n_high_cv": int(n_high_cv),
            "n_low_cv": int(n_low_cv),
            "met": n_high_cv >= 30 and n_low_cv >= 30
        },
        "statistics": {
            "total_features": int(sae.cfg.d_sae),
            "absorbed_features": int(n_absorbed),
            "high_cv_count": int(n_high_cv),
            "low_cv_count": int(n_low_cv),
            "high_cv_ratio": float(n_high_cv / n_absorbed) if n_absorbed > 0 else 0,
            "low_cv_ratio": float(n_low_cv / n_absorbed) if n_absorbed > 0 else 0,
            "high_cv_mean_cv": high_cv_mean,
            "high_cv_std_cv": high_cv_std,
            "low_cv_mean_cv": low_cv_mean,
            "low_cv_std_cv": low_cv_std,
            "overall_cv_mean": float(absorbed_cv.mean()),
            "overall_cv_std": float(absorbed_cv.std())
        },
        "groups": {
            "high_cv": {
                "threshold": "> 1.0",
                "count": int(n_high_cv),
                "mean_cv": high_cv_mean,
                "std_cv": high_cv_std,
                "indices": [int(x) for x in top_high_cv_indices],
                "cv_values": [float(x) for x in high_cv_values]
            },
            "low_cv": {
                "threshold": "<= 1.0",
                "count": int(n_low_cv),
                "mean_cv": low_cv_mean,
                "std_cv": low_cv_std,
                "indices": [int(x) for x in top_low_cv_indices],
                "cv_values": [float(x) for x in low_cv_values]
            }
        },
        "absorption_threshold": 0.1,
        "n_samples": len(text_samples),
        "model": "gpt2-small",
        "sae": "gpt2-small-res-jb",
        "layer": 6
    }

    # Save results
    output_path = RESULTS_DIR / "pilot_cv_classification.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Write DONE marker
    done_path = RESULTS_DIR / "pilot_cv_classification_DONE"
    done_path.write_text(json.dumps({
        "task_id": "pilot_cv_feature_classification",
        "status": "success" if results["pass"] else "needs_review",
        "timestamp": datetime.now().isoformat(),
    }))

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Pass criteria met: {results['pass']}")
    print(f"High-CV features (CV > 1.0): {n_high_cv} (need >= 30)")
    print(f"Low-CV features (CV <= 1.0): {n_low_cv} (need >= 30)")
    print(f"\nHigh-CV mean CV: {high_cv_mean:.3f}")
    print(f"Low-CV mean CV: {low_cv_mean:.3f}")

    return results

if __name__ == "__main__":
    results = main()
    sys.exit(0 if results['pass'] else 1)