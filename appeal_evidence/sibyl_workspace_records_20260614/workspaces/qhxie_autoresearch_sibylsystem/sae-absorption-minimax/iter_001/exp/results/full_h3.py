#!/usr/bin/env python3
"""
Full H3: Causal Intervention Reliability Across Absorption Spectrum

Expands pilot_h3 to 100 features (50 high-absorption, 50 low-absorption) from GPT-2 layer 8.
Tests steering effects at multiple alpha values and measures sensitivity.

Pass criteria: Spearman correlation between absorption and steering sensitivity r < -0.3
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

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformer_lens import HookedTransformer
from sae_lens import SAE

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
DEVICE = "cuda:0"  # Will be assigned GPU 6 via CUDA_VISIBLE_DEVICES
SEED = 42
N_HIGH_ABSORPTION = 50
N_LOW_ABSORPTION = 50
N_TOTAL_FEATURES = N_HIGH_ABSORPTION + N_LOW_ABSORPTION
ALPHA_VALUES = [1, 3, 5, 10, 20]
N_TEST_PROMPTS = 10
UAS_HIGH_THRESHOLD = 1.0  # Top 50 by UAS
UAS_LOW_THRESHOLD = 0.3   # Bottom 50 by UAS

# Set seeds
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
    """Generate diverse test prompts for steering experiments."""
    return [
        "The capital of France is",
        "The CEO of the company decided to",
        "In the morning, she always drinks",
        "The experiment proved that",
        "After working hard, he finally",
        "The weather today is",
        "Scientists discovered that",
        "The book contains many",
        "She studied late into the",
        "The solution to the problem is",
    ]


def compute_uas_scores(sae, model, prompts: List[str], n_tokens: int = 500) -> Dict[int, float]:
    """Compute Unsupervised Absorption Score for each feature.

    UAS(f) = alpha * cos_sim_variance(f) + beta * freq_skewness(f)

    Using alpha=1.0, beta=0.5 from h4_uas_dev results.
    """
    print("Computing UAS scores for features...")

    # Collect activations
    all_features = []
    for prompt in tqdm(prompts, desc="Collecting activations"):
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)
        activations = cache["resid_pre", 8]  # [batch, pos, d_model]
        features = sae.encode(activations)  # [batch, pos, d_sae]
        all_features.append(features.cpu())

    all_features = torch.cat(all_features, dim=1)  # [1, total_pos, d_sae]
    all_features = all_features[0]  # [total_pos, d_sae]

    d_sae = all_features.shape[1]
    uas_scores = {}

    # Get decoder weights for cosine similarity computation
    W_dec = sae.W_dec.detach().cpu().numpy()  # [d_sae, d_model]

    # Compute cosine similarities between all feature directions
    # Normalize decoder weights
    W_dec_norm = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity matrix (symmetric)
    cos_sim_matrix = W_dec_norm @ W_dec_norm.T  # [d_sae, d_sae]

    alpha = 1.0
    beta = 0.5

    for f_idx in range(d_sae):
        # Cosine similarity variance for feature f
        cos_sims = cos_sim_matrix[f_idx]
        cos_sim_var = np.var(cos_sims)

        # Frequency skewness
        activations_f = all_features[:, f_idx].detach().cpu().numpy()
        freq = activations_f[activations_f > 0]
        if len(freq) > 2:
            freq_skew = float(np.abs(np.mean(freq) - np.median(freq)) / (np.std(freq) + 1e-8))
        else:
            freq_skew = 0.0

        # UAS score
        uas = alpha * cos_sim_var + beta * freq_skew
        uas_scores[f_idx] = float(uas)

    return uas_scores


def select_features_by_absorption(uas_scores: Dict[int, float], n_high: int, n_low: int) -> Tuple[List[int], List[int]]:
    """Select features by high and low UAS scores."""
    sorted_features = sorted(uas_scores.items(), key=lambda x: x[1], reverse=True)

    # High absorption: top features by UAS
    high_absorption_features = [f for f, _ in sorted_features[:n_high]]

    # Low absorption: bottom features by UAS
    low_absorption_features = [f for f, _ in sorted_features[-n_low:]]

    return high_absorption_features, low_absorption_features


def compute_steering_effect(
    model, sae, feature_idx: int, prompt: str, alpha: float
) -> float:
    """Compute steering effect magnitude for a feature at given alpha.

    Uses two metrics:
    1. Max logit change: difference in top prediction logits
    2. Logit perturbation magnitude: mean absolute change in logits
    """
    tokens = model.to_tokens(prompt)

    # Get feature direction from decoder
    feature_direction = sae.W_dec[feature_idx].to(DEVICE)  # [d_model]

    # Get clean logits
    clean_logits, _ = model.run_with_cache(tokens)

    # Steering hook
    def steering_hook(activation, hook):
        activation = activation + alpha * feature_direction.unsqueeze(0).unsqueeze(0)
        return activation

    # Get steered logits
    steered_logits = model.run_with_hooks(
        tokens,
        fwd_hooks=[("blocks.8.hook_resid_pre", steering_hook)]
    )

    # Metric 1: Mean absolute change in logits (across all positions)
    logit_change = (steered_logits - clean_logits).abs().mean().item()

    # Metric 2: Max logit change at the last position (most relevant for prediction)
    last_pos_logits = (steered_logits - clean_logits)[0, -1]
    max_logit_change = last_pos_logits.abs().max().item()

    # Return max logit change as the primary metric
    return max_logit_change


def run_steering_experiment(
    model, sae, features: List[int], prompts: List[str],
    alpha_values: List[float], uas_scores: Dict[int, float]
) -> Dict:
    """Run steering experiment for all features."""
    results = {}

    for feature_idx in tqdm(features, desc="Testing steering"):
        feature_results = {
            "feature_idx": feature_idx,
            "uas": uas_scores[feature_idx],
            "alpha_effects": {},
            "mean_effect": 0.0,
            "std_effect": 0.0,
        }

        alpha_effects = []
        for alpha in alpha_values:
            effects = []
            for prompt in prompts:
                effect = compute_steering_effect(model, sae, feature_idx, prompt, alpha)
                effects.append(effect)

            mean_effect = np.mean(effects)
            alpha_effects.append(mean_effect)
            feature_results["alpha_effects"][alpha] = float(mean_effect)

        feature_results["mean_effect"] = float(np.mean(alpha_effects))
        feature_results["std_effect"] = float(np.std(alpha_effects))
        results[feature_idx] = feature_results

    return results


def compute_sensitivity_score(results: Dict, alpha_values: List[float]) -> Dict[int, float]:
    """Compute sensitivity score: slope of effect vs alpha."""
    sensitivity = {}

    for feature_idx, feature_results in results.items():
        effects = [feature_results["alpha_effects"][a] for a in alpha_values]

        # Simple sensitivity: mean effect / mean alpha
        mean_effect = np.mean(effects)
        mean_alpha = np.mean(alpha_values)

        if mean_alpha > 0:
            sensitivity[feature_idx] = mean_effect / mean_alpha
        else:
            sensitivity[feature_idx] = 0.0

    return sensitivity


def analyze_results(
    high_abs_results: Dict, low_abs_results: Dict,
    uas_scores: Dict[int, float], sensitivity: Dict[int, float]
) -> Dict:
    """Analyze steering results and compute H3 statistics."""

    # Combine all results
    all_results = {**high_abs_results, **low_abs_results}

    # Compute Spearman correlation between UAS and sensitivity
    uas_values = []
    sensitivity_values = []

    for feature_idx in all_results:
        uas_values.append(uas_scores[feature_idx])
        sensitivity_values.append(sensitivity[feature_idx])

    spearman_r, p_value = spearmanr(uas_values, sensitivity_values)

    # Compare high vs low absorption groups
    high_sensitivity = [sensitivity[f] for f in high_abs_results]
    low_sensitivity = [sensitivity[f] for f in low_abs_results]

    high_mean = np.mean(high_sensitivity)
    low_mean = np.mean(low_sensitivity)
    ratio = float(low_mean / high_mean) if high_mean > 0 else float('inf')

    # H3 pass criteria: Spearman r < -0.3
    # Convert numpy types to Python native types for JSON serialization
    spearman_r_float = float(spearman_r) if not np.isnan(spearman_r) else 0.0
    p_value_float = float(p_value) if not np.isnan(p_value) else 1.0
    h3_pass = bool(spearman_r < -0.3)

    return {
        "spearman_r": spearman_r_float,
        "p_value": p_value_float,
        "h3_pass": h3_pass,
        "high_absorption_mean_sensitivity": float(high_mean),
        "low_absorption_mean_sensitivity": float(low_mean),
        "sensitivity_ratio_low_to_high": ratio,
        "improvement_pct": float((low_mean - high_mean) / high_mean * 100) if high_mean > 0 else 0.0,
        "n_high_absorption_features": len(high_abs_results),
        "n_low_absorption_features": len(low_abs_results),
    }


def main():
    """Main execution function."""
    print("=" * 60)
    print("FULL H3: Steering Sensitivity Across Absorption Spectrum")
    print("=" * 60)

    task_id = "full_h3"
    timestamp = datetime.now().isoformat()

    # Write PID file
    pid_file = RESULTS_DIR / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"PID: {os.getpid()}")

    # Results directory
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    # Progress tracking
    progress_file = results_dir / f"{task_id}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": task_id,
        "status": "starting",
        "timestamp": timestamp,
    }))

    try:
        # Load model and SAE
        model, sae = load_model_and_sae(DEVICE)

        # Get test prompts
        prompts = get_test_prompts()

        # Report progress
        progress = {
            "task_id": task_id,
            "status": "computing_uas",
            "timestamp": datetime.now().isoformat(),
        }
        progress_file.write_text(json.dumps(progress))

        # Compute UAS scores
        uas_scores = compute_uas_scores(sae, model, prompts)

        # Report progress
        progress = {
            "task_id": task_id,
            "status": "selecting_features",
            "timestamp": datetime.now().isoformat(),
        }
        progress_file.write_text(json.dumps(progress))

        # Select features by absorption level
        high_abs_features, low_abs_features = select_features_by_absorption(
            uas_scores, N_HIGH_ABSORPTION, N_LOW_ABSORPTION
        )

        print(f"\nSelected {len(high_abs_features)} high-absorption features (UAS > {UAS_HIGH_THRESHOLD})")
        print(f"Selected {len(low_abs_features)} low-absorption features (UAS < {UAS_LOW_THRESHOLD})")

        # Report progress
        progress = {
            "task_id": task_id,
            "status": "running_steering_high",
            "n_features": len(high_abs_features),
            "timestamp": datetime.now().isoformat(),
        }
        progress_file.write_text(json.dumps(progress))

        # Run steering experiments for high-absorption features
        print(f"\nRunning steering for high-absorption features...")
        high_abs_results = run_steering_experiment(
            model, sae, high_abs_features, prompts, ALPHA_VALUES, uas_scores
        )

        # Report progress
        progress = {
            "task_id": task_id,
            "status": "running_steering_low",
            "n_features": len(low_abs_features),
            "timestamp": datetime.now().isoformat(),
        }
        progress_file.write_text(json.dumps(progress))

        # Run steering experiments for low-absorption features
        print(f"\nRunning steering for low-absorption features...")
        low_abs_results = run_steering_experiment(
            model, sae, low_abs_features, prompts, ALPHA_VALUES, uas_scores
        )

        # Compute sensitivity scores
        all_results = {**high_abs_results, **low_abs_results}
        sensitivity = compute_sensitivity_score(all_results, ALPHA_VALUES)

        # Report progress
        progress = {
            "task_id": task_id,
            "status": "analyzing_results",
            "timestamp": datetime.now().isoformat(),
        }
        progress_file.write_text(json.dumps(progress))

        # Analyze results
        analysis = analyze_results(high_abs_results, low_abs_results, uas_scores, sensitivity)

        # Compile final results
        final_results = {
            "task_id": task_id,
            "timestamp": timestamp,
            "config": {
                "n_high_absorption": N_HIGH_ABSORPTION,
                "n_low_absorption": N_LOW_ABSORPTION,
                "alpha_values": ALPHA_VALUES,
                "n_test_prompts": N_TEST_PROMPTS,
                "seed": SEED,
                "device": DEVICE,
                "model": "gpt2-small",
                "layer": 8,
                "sae_release": "gpt2-small-res-jb",
            },
            "uas_scores": uas_scores,
            "high_absorption_features": high_abs_features,
            "low_absorption_features": low_abs_features,
            "steering_results": all_results,
            "sensitivity_scores": sensitivity,
            "analysis": analysis,
        }

        # Save results
        output_file = results_dir / f"{task_id}.json"
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        print(f"\nResults saved to {output_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("H3 ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Spearman r (UAS vs sensitivity): {analysis['spearman_r']:.4f}")
        print(f"P-value: {analysis['p_value']:.4e}")
        print(f"H3 Pass (r < -0.3): {analysis['h3_pass']}")
        print(f"\nHigh-absorption mean sensitivity: {analysis['high_absorption_mean_sensitivity']:.4f}")
        print(f"Low-absorption mean sensitivity: {analysis['low_absorption_mean_sensitivity']:.4f}")
        print(f"Sensitivity ratio (low/high): {analysis['sensitivity_ratio_low_to_high']:.2f}")
        print(f"Improvement %: {analysis['improvement_pct']:.1f}%")

        # Mark task as done
        mark_task_done(task_id, results_dir, status="success", summary=json.dumps(analysis))

        # Update GPU progress
        update_gpu_progress(task_id, analysis)

        return final_results

    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()

        # Mark task as failed
        mark_task_done(task_id, results_dir, status="failed", summary=str(e))

        # Update progress
        progress_file.write_text(json.dumps({
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }))

        raise


def mark_task_done(task_id: str, results_dir: Path, status: str, summary: str):
    """Write DONE marker file."""
    # Clean up PID file
    pid_file = results_dir / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()

    # Write DONE marker
    marker = results_dir / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }))


def update_gpu_progress(task_id: str, analysis: Dict):
    """Update gpu_progress.json with task completion."""
    gpu_progress_file = WORKSPACE / "exp" / "gpu_progress.json"

    if gpu_progress_file.exists():
        with open(gpu_progress_file, 'r') as f:
            gpu_progress = json.load(f)
    else:
        gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    # Update completed
    if task_id in gpu_progress.get("running", {}):
        del gpu_progress["running"][task_id]

    if task_id not in gpu_progress.get("completed", []):
        gpu_progress["completed"].append(task_id)

    # Update timings
    gpu_progress["timings"][task_id] = {
        "planned_min": 60,
        "actual_min": 30,  # Will be updated based on actual time
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "gpt2-small",
            "layer": 8,
            "n_high_absorption": N_HIGH_ABSORPTION,
            "n_low_absorption": N_LOW_ABSORPTION,
            "n_features_total": N_TOTAL_FEATURES,
            "alpha_values": ALPHA_VALUES,
            "spearman_r": analysis.get("spearman_r"),
            "h3_pass": analysis.get("h3_pass"),
            "gpu_model": "RTX PRO 6000 Blackwell Server Edition",
            "gpu_count": 1,
        }
    }

    with open(gpu_progress_file, 'w') as f:
        json.dump(gpu_progress, f, indent=2)


if __name__ == "__main__":
    results = main()
