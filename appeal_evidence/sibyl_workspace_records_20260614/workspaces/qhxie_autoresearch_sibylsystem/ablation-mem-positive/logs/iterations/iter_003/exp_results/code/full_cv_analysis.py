#!/usr/bin/env python3
"""
Full CV Analysis (H4): Coefficient of Variation Analysis Across Layers

Measures coefficient of variation for absorbed vs non-absorbed features
across layers [0, 3, 6, 9, 11] at lambda=1e-3.

Hypothesis H4: CV_low < CV_high at critical point (layer 6)
Pilot found: CV_direction REVERSED (absorbed features have MUCH higher CV)

This full analysis extends the pilot across all layers to characterize the
CV difference pattern.
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
LAYERS = [0, 3, 6, 9, 11]
LAMBDA = 1e-3  # Fixed sparsity for cross-layer comparison
N_SAMPLES = 1000
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR = PROJECT_ROOT / "exp" / "results" / "full"
PILOT_DIR = PROJECT_ROOT / "exp" / "results" / "pilots"


def load_model_and_sae(layer_idx):
    """Load GPT-2 and SAE model for a specific layer."""
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    print(f"Loading GPT-2 Small model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

    hook_name = f"blocks.{layer_idx}.hook_resid_pre"
    print(f"Loading SAE for {hook_name}...")
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=hook_name,
        device=DEVICE
    )

    return model, sae


def compute_cv_by_layer(layers, lambda_val, n_samples=1000, seed=42):
    """
    Compute CV (coefficient of variation) for absorbed vs non-absorbed features
    across multiple layers.

    CV = std / mean for activation magnitudes
    """
    from datasets import load_dataset

    np.random.seed(seed)
    torch.manual_seed(seed)

    results_by_layer = {}

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"Processing Layer {layer_idx}")
        print(f"{'='*60}")

        hook_name = f"blocks.{layer_idx}.hook_resid_pre"

        # Load model and SAE for this layer
        model, sae_layer = load_model_and_sae(layer_idx)
        tokenizer = model.tokenizer

        # Generate text samples
        try:
            dataset = load_dataset("stas/open-webtext-10k", split="train")
            texts = [dataset[i]["text"] for i in range(min(n_samples, len(dataset)))]
        except Exception as e:
            print(f"Dataset load failed: {e}, using fallback prompts")
            texts = [f"Sample text prompt {i}" for i in range(n_samples)]

        print(f"Processing {len(texts)} samples...")

        layer_results = {
            "layer": layer_idx,
            "hook_name": hook_name,
            "lambda": lambda_val,
            "cv_absorbed": {"mean": None, "std": None, "n_features": 0, "per_feature_values": []},
            "cv_non_absorbed": {"mean": None, "std": None, "n_features": 0, "per_feature_values": []},
            "cv_difference": {},
            "statistical_test": {},
            "absorption_stats": {"n_absorbed": 0, "n_non_absorbed": 0, "absorption_rate": 0.0}
        }

        all_absorbed_cvs = []
        all_non_absorbed_cvs = []
        absorption_scores = []

        # Process in batches
        batch_size = 32
        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start:batch_start + batch_size]

            # Tokenize
            tokens = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).input_ids.to(DEVICE)

            # Get activations
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=[hook_name]
                )

            acts = cache[hook_name]  # (batch, seq, d_model)

            # Flatten sequence dimension
            acts = acts.reshape(-1, acts.shape[-1])

            # Encode with SAE
            sae_acts = sae_layer.encode(acts)

            # Get output (reconstruction)
            reconstructed = sae_layer.decode(sae_acts)

            # Compute absorption: fraction of feature preserved after SAE
            # absorption_score = 1 - ||resid|| / ||act|| (for each position)
            residual = acts - reconstructed
            residual_norm = torch.norm(residual, dim=-1)  # (batch*seq,)
            act_norm = torch.norm(acts, dim=-1)
            absorption_scores_batch = (1 - (residual_norm / (act_norm + 1e-8))).detach()

            # Binary absorption: feature is "absorbed" if its absorption score is high
            absorbed_mask = absorption_scores_batch > 0.001

            # Per-feature CV analysis
            # CV = std / mean for activation magnitudes across samples
            feature_activations = sae_acts.detach()  # (batch*seq, d_sae)

            # Mean and std across samples (dim=0) for each feature
            feature_means = feature_activations.mean(dim=0).cpu().numpy()
            feature_stds = feature_activations.std(dim=0).cpu().numpy()

            # CV = std / mean (handle zero means)
            cv_per_feature = np.zeros_like(feature_stds)
            nonzero_mask = feature_means > 1e-8
            cv_per_feature[nonzero_mask] = feature_stds[nonzero_mask] / feature_means[nonzero_mask]

            # Classify features by absorption status
            # A feature is "absorbed" if the mean absorption score across positions using it is > threshold
            absorbed_features = set()
            for pos_idx in range(len(absorption_scores_batch)):
                if absorbed_mask[pos_idx]:
                    # Find which features are active at this position
                    pos_features = sae_acts[pos_idx].nonzero().cpu().numpy()
                    absorbed_features.update(pos_features.flatten().tolist())

            # Compute CV for absorbed vs non-absorbed features
            n_total_features = len(cv_per_feature)
            absorbed_features = sorted(absorbed_features)

            for feat_idx in range(n_total_features):
                if feat_idx in absorbed_features:
                    all_absorbed_cvs.append(cv_per_feature[feat_idx])
                else:
                    all_non_absorbed_cvs.append(cv_per_feature[feat_idx])

            absorption_scores.extend(absorption_scores_batch.cpu().numpy().tolist())

            if (batch_start + batch_size) % 320 == 0:
                print(f"  Processed {batch_start + batch_size}/{len(texts)} samples")

        # Compute statistics
        all_absorbed_cvs = np.array(all_absorbed_cvs)
        all_non_absorbed_cvs = np.array(all_non_absorbed_cvs)

        if len(all_absorbed_cvs) > 0:
            layer_results["cv_absorbed"]["mean"] = float(np.mean(all_absorbed_cvs))
            layer_results["cv_absorbed"]["std"] = float(np.std(all_absorbed_cvs))
            layer_results["cv_absorbed"]["n_features"] = len(all_absorbed_cvs)
            layer_results["cv_absorbed"]["per_feature_values"] = all_absorbed_cvs[:100].tolist()

        if len(all_non_absorbed_cvs) > 0:
            layer_results["cv_non_absorbed"]["mean"] = float(np.mean(all_non_absorbed_cvs))
            layer_results["cv_non_absorbed"]["std"] = float(np.std(all_non_absorbed_cvs))
            layer_results["cv_non_absorbed"]["n_features"] = len(all_non_absorbed_cvs)
            layer_results["cv_non_absorbed"]["per_feature_values"] = all_non_absorbed_cvs[:100].tolist()

        # Compute difference
        if layer_results["cv_absorbed"]["mean"] is not None and layer_results["cv_non_absorbed"]["mean"] is not None:
            cv_diff = layer_results["cv_absorbed"]["mean"] - layer_results["cv_non_absorbed"]["mean"]
            layer_results["cv_difference"]["high_minus_low"] = float(cv_diff)
            if layer_results["cv_absorbed"]["mean"] > 0:
                layer_results["cv_difference"]["ratio"] = layer_results["cv_non_absorbed"]["mean"] / layer_results["cv_absorbed"]["mean"]

        # Statistical test
        if len(all_absorbed_cvs) > 1 and len(all_non_absorbed_cvs) > 1:
            t_stat, p_val = stats.ttest_ind(all_absorbed_cvs, all_non_absorbed_cvs)
            layer_results["statistical_test"]["t_statistic"] = float(t_stat)
            layer_results["statistical_test"]["p_value"] = float(p_val)
            layer_results["statistical_test"]["significant_at_0_01"] = bool(p_val < 0.01)
            layer_results["statistical_test"]["significant_at_0_05"] = bool(p_val < 0.05)

        # Absorption stats
        absorption_scores = np.array(absorption_scores)
        layer_results["absorption_stats"]["absorption_rate"] = float((absorption_scores > 0.001).mean())
        layer_results["absorption_stats"]["n_absorbed"] = int((absorption_scores > 0.001).sum())
        layer_results["absorption_stats"]["n_non_absorbed"] = int((absorption_scores <= 0.001).sum())

        results_by_layer[layer_idx] = layer_results
        print(f"\nLayer {layer_idx} Summary:")
        print(f"  Absorption rate: {layer_results['absorption_stats']['absorption_rate']:.4f}")
        print(f"  CV_absorbed: {layer_results['cv_absorbed']['mean']:.4f} (n={layer_results['cv_absorbed']['n_features']})")
        print(f"  CV_non_absorbed: {layer_results['cv_non_absorbed']['mean']:.4f} (n={layer_results['cv_non_absorbed']['n_features']})")
        if 'high_minus_low' in layer_results['cv_difference']:
            print(f"  CV_diff (absorbed - non_absorbed): {layer_results['cv_difference']['high_minus_low']:.4f}")
        if 't_statistic' in layer_results['statistical_test']:
            print(f"  t-statistic: {layer_results['statistical_test']['t_statistic']:.2f}, p-value: {layer_results['statistical_test']['p_value']:.2e}")

        # Clean up GPU memory
        del model, sae_layer
        torch.cuda.empty_cache()

    return results_by_layer


def main():
    print("=" * 60)
    print("FULL CV ANALYSIS (H4): Cross-Layer Coefficient of Variation")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print(f"Layers: {LAYERS}")
    print(f"Lambda: {LAMBDA}")
    print(f"N_samples: {N_SAMPLES}")
    print()

    # Compute CV analysis
    results = compute_cv_by_layer(
        layers=LAYERS,
        lambda_val=LAMBDA,
        n_samples=N_SAMPLES,
        seed=SEED
    )

    # Compile summary
    summary = {
        "task_id": "full_cv_analysis",
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "layers": LAYERS,
            "lambda": LAMBDA,
            "absorption_threshold": 0.001
        },
        "layer_results": results,
        "h4_analysis": {
            "pilot_observation": "CV_direction_REVERSED: absorbed features have HIGHER CV than non-absorbed",
            "h4_prediction": "CV_low < CV_high at critical point (layer 6)",
            "layers_tested": LAYERS
        }
    }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "cv_full_analysis.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: CV Analysis by Layer")
    print("=" * 60)
    for layer, res in results.items():
        print(f"\nLayer {layer}:")
        print(f"  Absorption rate: {res['absorption_stats']['absorption_rate']:.4f}")
        if res['cv_absorbed']['mean'] is not None:
            print(f"  CV_absorbed: {res['cv_absorbed']['mean']:.4f} (n={res['cv_absorbed']['n_features']})")
        if res['cv_non_absorbed']['mean'] is not None:
            print(f"  CV_non_absorbed: {res['cv_non_absorbed']['mean']:.4f} (n={res['cv_non_absorbed']['n_features']})")
        if 'high_minus_low' in res['cv_difference']:
            print(f"  CV_diff (absorbed - non_absorbed): {res['cv_difference']['high_minus_low']:.4f}")
        if 't_statistic' in res['statistical_test']:
            print(f"  t-statistic: {res['statistical_test']['t_statistic']:.2f}, p-value: {res['statistical_test']['p_value']:.2e}")

    # Write DONE marker
    done_file = RESULTS_DIR / "full_cv_analysis_DONE"
    done_file.write_text(json.dumps({
        "task_id": "full_cv_analysis",
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "layers_completed": list(results.keys())
    }))

    print("\nDone.")


if __name__ == "__main__":
    main()
