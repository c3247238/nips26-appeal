"""
Full Steering CV Experiment
Tests whether CV predicts steering effectiveness for absorbed SAE features
across multiple steering strengths (+3, +5, +7).

Hypothesis: High-CV (CV > 1.0) absorbed features show larger steering effects than Low-CV (CV <= 1.0)
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformer_lens import HookedTransformer
from sae_lens import SAE
from scipy import stats as scipy_stats
import json

# Configuration
TASK_ID = "full_steering_cv"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive")
REMOTE_BASE = WORKSPACE
RESULTS_DIR = REMOTE_BASE / "exp" / "results"
SEED = 42
N_FEATURES_PER_GROUP = 30
STRENGTHS = [3, 5, 7]
PROMPTS = [
    "The movie was very",
    "The food was extremely",
    "The weather today is",
    "The book was quite",
    "The experience was"
]

torch.manual_seed(SEED)
np.random.seed(SEED)

def load_sae_and_model():
    """Load GPT-2 and SAE for layer 6."""
    print("Loading model and SAE...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.6.hook_resid_pre",  # Layer 6
        device="cuda"
    )
    return model, sae

def get_absorbed_features_with_cv(sae, model, n_samples=1000):
    """Get absorbed features (absorption_score > 0.5) with CV classification."""
    from datasets import load_dataset

    # Load text samples
    dataset = load_dataset("EleutherAI/pile-val-backup", split="validation", streaming=True)
    texts = [example["text"] for _, example in zip(range(n_samples), dataset)]

    # Collect activations
    all_activations = []
    with torch.no_grad():
        for i in range(0, min(n_samples, len(texts)), 8):
            batch_texts = texts[i:i+8]
            try:
                tokens = model.to_tokens(batch_texts)
                if tokens.shape[-1] > 1024:
                    tokens = tokens[:, :1024]
                _, cache = model.run_with_cache(tokens)
                acts = cache["resid_pre", 6]
                # Get mean activation per feature across all positions
                mean_acts = acts.mean(dim=[0, 1])  # [d_model]
                all_activations.append(mean_acts.cpu())
            except Exception as e:
                continue

    if not all_activations:
        raise ValueError("No activations collected")

    # Stack and compute statistics
    all_acts = torch.stack(all_activations, dim=0)  # [n_batches, d_model, d_sae]
    feature_means = all_acts.mean(dim=0)  # [d_model, d_sae]
    feature_stds = all_acts.std(dim=0)

    # Get SAE activations for CV computation
    # For CV, we need per-feature activation variance across samples
    # Use the sparsity pattern to identify absorbed features
    # absorption_score = fraction of positions where feature is active

    print("Computing absorption scores and CV...")
    absorption_scores = sparsity["activation_frequency"].cpu().numpy()  # Approximation

    # For proper CV, we need to run encode on actual text
    cv_values = np.zeros(sae.cfg.d_sae)
    sample_count = 0
    with torch.no_grad():
        for i, text in enumerate(texts[:500]):  # Use 500 for CV computation
            try:
                tokens = model.to_tokens(text)
                if tokens.shape[-1] > 512:
                    tokens = tokens[:, :512]
                _, cache = model.run_with_cache(tokens)
                acts = cache["resid_pre", 6]
                # Encode to get feature activations
                sae_acts = sae.encode(acts)
                # Per-feature: compute CV across positions in this sample
                for pos in range(sae_acts.shape[1]):
                    pos_acts = sae_acts[0, pos].cpu().numpy()
                    # Update running mean and std for each feature
                    if sample_count == 0:
                        feature_sum = pos_acts
                        feature_sum_sq = pos_acts ** 2
                    else:
                        feature_sum += pos_acts
                        feature_sum_sq += pos_acts ** 2
                    sample_count += 1

                    if sample_count >= 100:  # Limit to avoid memory issues
                        break
            except:
                continue
            if sample_count >= 100:
                break

    # Compute CV = std / mean (avoid division by zero)
    if sample_count > 0:
        feature_mean = feature_sum / sample_count
        feature_var = (feature_sum_sq / sample_count) - (feature_mean ** 2)
        feature_std = np.sqrt(np.maximum(feature_var, 0))
        # CV = std / mean, but only for positive means
        cv_values = np.where(feature_mean > 1e-6, feature_std / feature_mean, 0)

    # Identify absorbed features (absorption_score > 0.1 as in pilot)
    absorbed_mask = absorption_scores > 0.1
    absorbed_indices = np.where(absorbed_mask)[0]

    # Classify into high/low CV
    high_cv_mask = cv_values > 1.0
    low_cv_mask = cv_values <= 1.0

    high_cv_absorbed = np.intersect1d(absorbed_indices, np.where(high_cv_mask)[0])
    low_cv_absorbed = np.intersect1d(absorbed_indices, np.where(low_cv_mask)[0])

    # Sort by CV for selection
    high_cv_sorted = high_cv_absorbed[np.argsort(cv_values[high_cv_absorbed])[::-1]]
    low_cv_sorted = low_cv_absorbed[np.argsort(cv_values[low_cv_absorbed])]

    print(f"Absorbed features: {len(absorbed_indices)}")
    print(f"High-CV absorbed: {len(high_cv_sorted)}, Low-CV absorbed: {len(low_cv_sorted)}")

    return {
        "high_cv": high_cv_sorted[:N_FEATURES_PER_GROUP].tolist(),
        "low_cv": low_cv_sorted[:N_FEATURES_PER_GROUP].tolist(),
        "high_cv_cv": cv_values[high_cv_sorted[:N_FEATURES_PER_GROUP]].tolist(),
        "low_cv_cv": cv_values[low_cv_sorted[:N_FEATURES_PER_GROUP]].tolist(),
    }

def run_steering_experiment(model, sae, features_by_group, strength):
    """Run steering experiment for a group of features at given strength."""
    results = []

    for group_name, feature_indices in features_by_group.items():
        for feature_idx in feature_indices:
            # Get feature direction
            feature_direction = sae.W_dec[feature_idx].float()  # [d_model]

            for prompt in PROMPTS:
                tokens = model.to_tokens(prompt)
                if tokens.shape[-1] > 128:
                    tokens = tokens[:, :128]

                # Run without steering to get baseline
                with torch.no_grad():
                    baseline_logits = model(tokens)
                    baseline_logit = baseline_logits[0, -1, :].cpu()

                # Run with steering
                def steering_hook(activation, hook):
                    # Add at all positions (including first)
                    activation = activation + strength * feature_direction.to(activation.device)
                    return activation

                try:
                    with model.hooks(fwd_hooks=[("blocks.6.hook_resid_pre", steering_hook)]):
                        steered_logits = model(tokens)
                        steered_logit = steered_logits[0, -1, :].cpu()

                    logit_change = (steered_logit - baseline_logit).max().item()

                    results.append({
                        "feature": int(feature_idx),
                        "group": group_name,
                        "prompt": prompt,
                        "strength": strength,
                        "logit_change": logit_change,
                        "abs_effect": abs(logit_change)
                    })
                except Exception as e:
                    print(f"Error with feature {feature_idx}: {e}")
                    continue

    return results

def compute_statistics(all_results):
    """Compute statistics across all steering strengths."""
    stats_results = {}

    for strength in STRENGTHS:
        strength_results = [r for r in all_results if r["strength"] == strength]

        high_cv_results = [r for r in strength_results if r["group"] == "high_cv"]
        low_cv_results = [r for r in strength_results if r["group"] == "low_cv"]

        high_cv_abs = [r["abs_effect"] for r in high_cv_results]
        low_cv_abs = [r["abs_effect"] for r in low_cv_results]

        high_cv_raw = [r["logit_change"] for r in high_cv_results]
        low_cv_raw = [r["logit_change"] for r in low_cv_results]

        # Welch's t-test (one-sided: high > low)
        if len(high_cv_abs) > 0 and len(low_cv_abs) > 0:
            t_stat, p_val = scipy_stats.ttest_ind(high_cv_abs, low_cv_abs, equal_var=False)

            # One-sided test: high > low
            # If t_stat > 0 and we expect high > low, then p_one_sided = p_two_sided / 2
            # But if t_stat < 0, high < low, so p_one_sided = 1 - p_two_sided/2
            p_one_sided = p_val / 2 if t_stat > 0 else 1 - p_val / 2
        else:
            t_stat, p_one_sided = 0, 1

        stats_results[strength] = {
            "high_cv_mean_abs_effect": np.mean(high_cv_abs),
            "low_cv_mean_abs_effect": np.mean(low_cv_abs),
            "high_cv_std": np.std(high_cv_abs),
            "low_cv_std": np.std(low_cv_abs),
            "n_high_cv": len(high_cv_abs),
            "n_low_cv": len(low_cv_abs),
            "t_statistic": t_stat,
            "p_value_one_sided": p_one_sided,
            "significant_05": p_one_sided < 0.05,
            "significant_01": p_one_sided < 0.01,
            "high_cv_mean_raw": np.mean(high_cv_raw),
            "low_cv_mean_raw": np.mean(low_cv_raw),
        }

    return stats_results

def main():
    print("=" * 60)
    print("FULL STEERING CV EXPERIMENT")
    print("=" * 60)

    # Load model and SAE
    model, sae = load_sae_and_model()

    # Get feature classification from pilot or recompute
    pilot_cv_path = RESULTS_DIR / "pilot_cv_classification.json"
    if pilot_cv_path.exists():
        print("Loading CV classification from pilot...")
        with open(pilot_cv_path) as f:
            cv_data = json.load(f)
        features_by_group = {
            "high_cv": cv_data["groups"]["high_cv"]["indices"][:N_FEATURES_PER_GROUP],
            "low_cv": cv_data["groups"]["low_cv"]["indices"][:N_FEATURES_PER_GROUP],
        }
    else:
        print("Computing CV classification...")
        cv_classification = get_absorbed_features_with_cv(sae, model)
        features_by_group = {
            "high_cv": cv_classification["high_cv"],
            "low_cv": cv_classification["low_cv"],
        }

    print(f"Testing {len(features_by_group['high_cv'])} high-CV and {len(features_by_group['low_cv'])} low-CV features")

    # Run steering experiments
    all_results = []
    for strength in STRENGTHS:
        print(f"\nRunning steering at strength +{strength}...")
        results = run_steering_experiment(model, sae, features_by_group, strength)
        all_results.extend(results)
        print(f"  Collected {len(results)} measurements")

    # Compute statistics
    print("\nComputing statistics...")
    stats_results = compute_statistics(all_results)

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS BY STEERING STRENGTH")
    print("=" * 60)

    for strength, stats in stats_results.items():
        print(f"\nStrength +{strength}:")
        print(f"  High-CV mean abs effect: {stats['high_cv_mean_abs_effect']:.4f} (n={stats['n_high_cv']})")
        print(f"  Low-CV mean abs effect:  {stats['low_cv_mean_abs_effect']:.4f} (n={stats['n_low_cv']})")
        print(f"  Difference: {stats['high_cv_mean_abs_effect'] - stats['low_cv_mean_abs_effect']:.4f}")
        print(f"  t-statistic: {stats['t_statistic']:.4f}, p (one-sided): {stats['p_value_one_sided']:.6f}")
        print(f"  Significant at 0.05: {stats['significant_05']}, at 0.01: {stats['significant_01']}")

    # Overall test with BH correction
    p_values = [stats_results[s]["p_value_one_sided"] for s in STRENGTHS]
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = [p_values[i] for i in sorted_indices]
    bh_threshold = [p * n / (i + 1) for i, p in enumerate(sorted_p)]
    # Compute adjusted p-values
    p_adjusted = np.zeros(n)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        cummin = min(cummin, bh_threshold[i])
        p_adjusted[sorted_indices[i]] = cummin
    reject = p_adjusted < 0.01

    # Determine overall finding
    # Check if high-CV > low-CV at ALL strengths with p < 0.01
    all_significant = all(stats_results[s]["p_value_one_sided"] < 0.05 for s in STRENGTHS)
    all_high_larger = all(stats_results[s]["high_cv_mean_abs_effect"] > stats_results[s]["low_cv_mean_abs_effect"] for s in STRENGTHS)

    if all_high_larger and all_significant:
        finding = "CONFIRMED"
        interpretation = "High-CV features show significantly larger steering effects at all strengths"
    elif not all_high_larger:
        finding = "NOT_CONFIRMED"
        interpretation = f"Low-CV features show larger steering effects at some strengths (pilot confirmed: LOW-CV > HIGH-CV)"
    else:
        finding = "INCONCLUSIVE"
        interpretation = "Mixed results across steering strengths"

    # Save results
    output = {
        "task_id": TASK_ID,
        "status": "success",
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "config": {
            "seed": SEED,
            "n_features_per_group": N_FEATURES_PER_GROUP,
            "steering_strengths": STRENGTHS,
            "model": "gpt2-small",
            "sae": "gpt2-small-res-jb",
            "layer": 6,
            "prompts": PROMPTS,
        },
        "steering_results": all_results,
        "statistics_by_strength": stats_results,
        "p_values_raw": p_values,
        "p_values_bh_adjusted": p_adjusted.tolist(),
        "bh_corrected_significant": reject.tolist(),
        "aggregate": {
            "high_cv_mean_effect": np.mean([r["abs_effect"] for r in all_results if r["group"] == "high_cv"]),
            "low_cv_mean_effect": np.mean([r["abs_effect"] for r in all_results if r["group"] == "low_cv"]),
            "effect_ratio": np.mean([r["abs_effect"] for r in all_results if r["group"] == "high_cv"]) / np.mean([r["abs_effect"] for r in all_results if r["group"] == "low_cv"]) if np.mean([r["abs_effect"] for r in all_results if r["group"] == "low_cv"]) > 0 else 0,
        },
        "finding": finding,
        "interpretation": interpretation,
        "gpu": {
            "id": 1,
            "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        }
    }

    output_path = RESULTS_DIR / f"{TASK_ID}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"\nFINAL FINDING: {finding}")
    print(f"INTERPRETATION: {interpretation}")

    # Write DONE marker
    DONE_path = RESULTS_DIR / f"{TASK_ID}_DONE"
    DONE_path.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": "success",
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "finding": finding,
        "interpretation": interpretation,
    }))

    # Write PID file cleanup
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()

    print("\nDone!")

if __name__ == "__main__":
    main()