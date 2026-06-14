#!/usr/bin/env python3
"""
Full Sparsity Sweep Experiment (H1 - Critical Threshold Detection)
Task: full_sparsity_sweep

This script performs a complete sparsity sweep across 12 log-spaced λ values
to identify the critical threshold λ_c for SAE feature absorption phase transition.
Based on the pilot script's approach using decoder weight magnitude as the absorption signal.
"""

import os
import sys
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import torch
import numpy as np

# Import transformer_lens and sae_lens
from transformer_lens import HookedTransformer
from sae_lens import SAE

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
WORKSPACE = "/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive"
REMOTE_BASE = WORKSPACE
RESULTS_DIR = f"{WORKSPACE}/exp/results"
FULL_RESULTS_DIR = f"{RESULTS_DIR}/full"

# Full sweep parameters from task_plan.json
FULL_LAMBDAS = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
N_SAMPLES = 1000
SEED = 42
LAYER = 6  # GPT-2 layer 6 (identified hotspot)
HOOK_NAME = f"blocks.{LAYER}.hook_resid_pre"
MODEL_NAME = "gpt2-small"
SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID = f"blocks.{LAYER}.hook_resid_pre"

TASK_ID = "full_sparsity_sweep"


def write_pid_file(results_dir: str, task_id: str):
    """Write PID file for system recovery detection."""
    pid_file = Path(results_dir) / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))
    return pid_file


def report_progress(task_id: str, results_dir: str, epoch: int, total_epochs: int,
                    step: int = 0, total_steps: int = 0, loss=None, metric=None):
    """Write progress file for system monitor to track."""
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "total_steps": total_steps,
        "loss": loss,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id: str, results_dir: str, status="success", summary=""):
    """Write DONE marker file for system monitor to detect."""
    # Clean up PID file
    pid_file = Path(results_dir) / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()

    # Merge final progress if available
    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass

    # Write DONE marker
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))
    return marker


def load_model_and_sae():
    """Load GPT-2 model and SAE for layer 6."""
    print("Loading GPT-2 Small model...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda")
    print(f"  Model loaded: {MODEL_NAME}")

    print(f"Loading SAE from release '{SAE_RELEASE}', SAE ID '{SAE_ID}'...")
    # SAE.from_pretrained returns only SAE object (deprecated unpacking)
    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device="cuda"
    )
    # Get config dict for informational purposes
    _, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
        release=SAE_RELEASE,
        sae_id=SAE_ID
    )
    print(f"  SAE loaded: d_sae={cfg_dict['d_sae']}, d_in={cfg_dict['d_in']}")
    if sparsity is not None:
        sparsity_val = float(sparsity) if sparsity.numel() == 1 else float(sparsity.mean())
        print(f"  Sparsity (L0): {sparsity_val:.2f}")
    else:
        print("  Sparsity: N/A (deprecated API return)")

    return model, sae, cfg_dict


def get_text_samples(n_samples: int, seed: int = 42) -> List[str]:
    """Get text samples for activation extraction."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)

    # Generate diverse prompts covering common topics
    base_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a surprising turn of events, the scientist discovered that",
        "The economic outlook for next year suggests that",
        "According to recent studies, the relationship between",
        "The cultural significance of this tradition can be traced back to",
        "Technical analysis indicates that the market is",
        "The historical context of this decision involved",
        "Recent advances in machine learning have enabled",
        "The philosophical implications of this theory include",
        "Medical researchers have found that regular",
        "The impact of climate change on global agriculture",
        "A new study reveals the connection between",
        "Experts believe that the future of technology",
        "The debate over renewable energy sources",
        "Understanding the psychology of decision making",
        "The role of artificial intelligence in healthcare",
        "How social media influences public opinion",
        "The evolution of modern architecture",
        "Exploring the mysteries of the universe",
        "The importance of conservation efforts",
    ]

    # Repeat and vary prompts to get enough samples
    samples = []
    while len(samples) < n_samples:
        for prompt in base_prompts:
            samples.append(prompt)
            if len(samples) >= n_samples:
                break

    return samples[:n_samples]


def compute_absorption_v2(lambda_val, activations, sae):
    """
    Compute absorption rate m(λ) using decoder weight magnitude as the key signal.

    Key insight: Features with large decoder weights are considered more "absorbed"
    into the model's representation. We measure:

    1. Decoder weight norm per feature: ||W_dec[i]||_2
    2. Activation magnitude per feature: |f_i|
    3. Absorption score = weight_norm * activation_magnitude

    For a given sparsity λ, we threshold on absorption score, not raw activation.
    This makes the metric more sensitive to the underlying feature importance.
    """
    # Encode with SAE
    sae_features = sae.encode(activations).detach()  # [batch, seq, d_sae]

    # Compute decoder weight norms (importance of each feature)
    W_dec = sae.W_dec.detach()  # [d_sae, d_in]
    weight_norms = W_dec.norm(dim=1).cpu().numpy()  # [d_sae]

    # Compute mean activation magnitude per feature across all positions
    mean_activations = sae_features.abs().mean(dim=(0, 1)).cpu().numpy()  # [d_sae]

    # Absorption score = weight_norm * mean_activation
    absorption_scores = weight_norms * mean_activations  # [d_sae]

    # Threshold by λ: features with absorption_score > λ are "absorbed"
    absorbed_mask = absorption_scores > lambda_val
    absorption_rate = absorbed_mask.mean()

    # Number of absorbed features (L0-like metric)
    n_absorbed = absorbed_mask.sum()

    # Susceptibility: variance across samples
    sample_scores = (sae_features.abs() * torch.tensor(weight_norms, device=sae_features.device).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    susceptibility = sample_scores.std().item() / (sample_scores.mean().item() + 1e-8)

    # Also compute "absorption intensity" - sum of scores for absorbed features
    absorbed_intensity = absorption_scores[absorbed_mask].sum() if absorbed_mask.any() else 0.0

    return {
        "absorption_rate": float(absorption_rate),
        "n_absorbed": int(n_absorbed),
        "susceptibility_proxy": float(susceptibility),
        "absorption_intensity": float(absorbed_intensity),
        "mean_activation_magnitude": float(mean_activations.mean()),
        "weight_norm_mean": float(weight_norms.mean()),
    }


def compute_susceptibility(absorption_rates: List[float], lambdas: List[float]) -> List[float]:
    """
    Compute susceptibility χ = dm/dλ (numerical derivative).
    """
    if len(absorption_rates) < 2:
        return [0.0] * len(absorption_rates)

    chi_values = []
    lambdas = np.array(lambdas)
    absorption_rates = np.array(absorption_rates)

    # Sort by lambda
    sort_idx = np.argsort(lambdas)
    lambdas = lambdas[sort_idx]
    absorption_rates = absorption_rates[sort_idx]

    for i in range(len(absorption_rates)):
        if i == 0:
            # Forward difference
            dm = absorption_rates[1] - absorption_rates[0]
            dlambda = lambdas[1] - lambdas[0]
        elif i == len(absorption_rates) - 1:
            # Backward difference
            dm = absorption_rates[i] - absorption_rates[i-1]
            dlambda = lambdas[i] - lambdas[i-1]
        else:
            # Central difference
            dm = absorption_rates[i+1] - absorption_rates[i-1]
            dlambda = lambdas[i+1] - lambdas[i-1]

        chi = abs(dm / dlambda) if dlambda != 0 else 0.0
        chi_values.append(chi)

    return chi_values


def main():
    print("=" * 70)
    print("FULL SPARSITY SWEEP EXPERIMENT (H1 - Critical Threshold Detection)")
    print("=" * 70)
    print(f"Task ID: {TASK_ID}")
    print(f"Start time: {datetime.now().isoformat()}")
    print()

    # Create results directories
    os.makedirs(FULL_RESULTS_DIR, exist_ok=True)
    print(f"Results directory: {FULL_RESULTS_DIR}")

    # Write PID file immediately
    write_pid_file(FULL_RESULTS_DIR, TASK_ID)
    print(f"PID file written")

    # Report initial progress
    report_progress(TASK_ID, FULL_RESULTS_DIR, epoch=0, total_epochs=1,
                    step=0, total_steps=len(FULL_LAMBDAS))
    print(f"Progress tracking initialized")

    # Load model and SAE
    model, sae, cfg_dict = load_model_and_sae()
    print()

    # Get text samples
    print(f"Getting {N_SAMPLES} text samples (seed={SEED})...")
    samples = get_text_samples(N_SAMPLES, SEED)
    print(f"  Sample count: {len(samples)}")
    print()

    # Tokenize all samples
    print("Tokenizing samples...")
    tokens = model.to_tokens(samples).to("cuda")
    print(f"  Tokens shape: {tokens.shape}")
    print()

    # Get activations once (same activations for all lambda values)
    print("Computing activations from GPT-2 layer 6...")
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[HOOK_NAME])
        activations = cache[HOOK_NAME]  # [batch, seq, d_model]
    print(f"  Activations shape: {activations.shape}")
    print()

    # Compute absorption for each lambda value
    all_results = []
    absorption_rates = []

    print("Running full sparsity sweep...")
    print(f"  Lambda values: {len(FULL_LAMBDAS)}")
    print()

    for i, lambda_val in enumerate(FULL_LAMBDAS):
        print(f"[{i+1}/{len(FULL_LAMBDAS)}] λ = {lambda_val:.1e}")

        # Clean GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        # Compute absorption using the v2 method
        result = compute_absorption_v2(lambda_val, activations, sae)

        # Add lambda to result
        result["lambda"] = lambda_val

        print(f"    absorption_rate: {result['absorption_rate']:.4f}, "
              f"n_absorbed: {result['n_absorbed']}, "
              f"χ_proxy: {result['susceptibility_proxy']:.4f}")

        all_results.append(result)
        absorption_rates.append(result["absorption_rate"])

        # Update progress
        report_progress(TASK_ID, FULL_RESULTS_DIR, epoch=1, total_epochs=1,
                        step=i+1, total_steps=len(FULL_LAMBDAS),
                        metric={"lambda": lambda_val, "absorption": result["absorption_rate"]})

    print()

    # Compute susceptibility (numerical derivative)
    lambdas_array = [r["lambda"] for r in all_results]
    chi_values = compute_susceptibility(absorption_rates, lambdas_array)

    # Add susceptibility to results
    for i, result in enumerate(all_results):
        result["susceptibility_chi"] = chi_values[i]

    # Find critical lambda (maximum susceptibility)
    max_chi_idx = np.argmax(chi_values)
    lambda_c = FULL_LAMBDAS[max_chi_idx]
    max_chi = chi_values[max_chi_idx]

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Lambda values tested: {len(FULL_LAMBDAS)}")
    print(f"Critical lambda (λ_c): {lambda_c:.1e} (max χ = {max_chi:.4f})")
    print()
    print("Absorption rates by lambda:")
    for r in all_results:
        print(f"  λ = {r['lambda']:.1e}: m = {r['absorption_rate']:.4f}, χ = {r['susceptibility_chi']:.4f}")

    # Compute phase transition analysis
    mean_chi = np.mean(chi_values) if np.mean(chi_values) > 0 else 1e-12
    chi_ratio = max_chi / mean_chi

    # Prepare final output
    output = {
        "task_id": TASK_ID,
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "layer": LAYER,
            "hook_name": HOOK_NAME,
            "model": MODEL_NAME,
            "sae_release": SAE_RELEASE,
            "sae_id": SAE_ID,
            "full_lambdas": FULL_LAMBDAS,
        },
        "lambda_values": FULL_LAMBDAS,
        "absorption_rates": absorption_rates,
        "susceptibility_chi": chi_values,
        "critical_lambda": float(lambda_c),
        "max_susceptibility": float(max_chi),
        "all_results": all_results,
        "pilot_reference": {
            "peak_lambda": 0.0005,
            "susceptibility_peak": 1.38,
            "note": "Pilot identified λ_c ≈ 5e-4"
        },
        "phase_transition_analysis": {
            "peak_detected": bool(max_chi > 0.5),
            "chi_ratio": float(chi_ratio),
            "transition_sharpness": "sharp" if chi_ratio > 3 else "gradual"
        },
        "gpu": {
            "id": torch.cuda.current_device(),
            "name": torch.cuda.get_device_name(torch.cuda.current_device()),
        }
    }

    # Write results JSON
    output_path = f"{FULL_RESULTS_DIR}/sparsity_sweep_full.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {output_path}")

    # Mark task as done
    mark_task_done(TASK_ID, FULL_RESULTS_DIR, status="success",
                   summary=f"Full sweep complete. λ_c={lambda_c:.1e}, peak χ={max_chi:.4f}")

    print(f"\nDONE marker written")
    print(f"End time: {datetime.now().isoformat()}")
    print("=" * 70)

    return output


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        # Write failure marker
        marker = Path(FULL_RESULTS_DIR) / f"{TASK_ID}_DONE"
        marker.write_text(json.dumps({
            "task_id": TASK_ID,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }))
        sys.exit(1)