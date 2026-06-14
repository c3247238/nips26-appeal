#!/usr/bin/env python3
"""
Full Dictionary Size Sweep Experiment (H2 - Finite-Size Scaling)

Task: full_dict_size_sweep
Hypothesis: Transition width δλ ∝ N^(-1/ν) where N = dictionary size

NOTE: The gpt2-small-res-jb release only has d_sae=24576 for layer 6.
The gpt2-small-res-jb-feature-splitting release has dictionary size variants
but only for layer 8. This experiment uses layer 8 feature-splitting SAEs to
test finite-size scaling, as they provide the dictionary size variation needed.

Available dictionary sizes via feature-splitting at layer 8:
- 768, 1536, 3072, 6144, 12288, 24576, 49152, 98304
"""

import os
import sys
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import torch
import numpy as np

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

# Dictionary sizes available via feature-splitting at layer 8
# (layer 6 only has d_sae=24576 in the standard release)
DICT_SIZES = [6144, 12288, 24576]  # Subset of available sizes for efficiency

# Sparsity percentile thresholds to test (maps to different effective λ)
SPARSE_PERCENTILES = [90, 92, 94, 95, 96, 97, 98, 99]

N_SAMPLES = 1000
SEED = 42
LAYER = 8  # Using layer 8 (feature-splitting has dict size variants here)
HOOK_NAME = f"blocks.{LAYER}.hook_resid_pre"
MODEL_NAME = "gpt2-small"
SAE_RELEASE_FEATURE_SPLIT = "gpt2-small-res-jb-feature-splitting"

# For scaling collapse analysis
NU_VALUES = [1, 2, 3]  # Critical exponent candidates

TASK_ID = "full_dict_size_sweep"


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
    pid_file = Path(results_dir) / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()

    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass

    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))
    return marker


def load_sae_for_dict_size(release: str, d_sae: int, layer: int) -> Optional[Tuple[SAE, dict]]:
    """
    Load an SAE with a specific dictionary size.
    For feature-splitting release, the SAE ID encodes the dict size.
    """
    # For feature-splitting, the ID format is "blocks.{layer}.hook_resid_pre_{d_sae}"
    sae_id = f"blocks.{layer}.hook_resid_pre_{d_sae}"

    try:
        sae = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device="cuda"
        )
        _, cfg_dict, _ = SAE.from_pretrained_with_cfg_and_sparsity(
            release=release,
            sae_id=sae_id
        )
        return sae, cfg_dict
    except Exception as e:
        print(f"  Warning: Could not load SAE {sae_id}: {e}")
        return None


def compute_absorption_by_percentile(activations: torch.Tensor, sae: SAE,
                                    percentiles: List[float]) -> List[Dict]:
    """
    Compute absorption rates for different sparsity percentile thresholds.

    Absorption score per feature = ||W_dec[i]|| * mean(|f_i|)
    Features with absorption scores above the percentile threshold are "absorbed".
    """
    # Encode with SAE
    with torch.no_grad():
        sae_features = sae.encode(activations).detach()

    # Compute decoder weight norms and mean activations
    W_dec = sae.W_dec.detach()  # [d_sae, d_in]
    weight_norms = W_dec.norm(dim=1)  # [d_sae]

    # Compute mean activation magnitude per feature
    mean_activations = sae_features.abs().mean(dim=(0, 1))  # [d_sae]

    # Absorption score per feature
    absorption_scores = (weight_norms * mean_activations).cpu().numpy()

    results = []
    for pct in percentiles:
        threshold = np.percentile(absorption_scores, pct)
        absorbed_mask = absorption_scores > threshold
        absorption_rate = float(absorbed_mask.mean())

        results.append({
            "percentile": pct,
            "lambda_threshold": float(threshold),
            "absorption_rate": absorption_rate,
            "n_absorbed": int(absorbed_mask.sum()),
            "d_sae": int(sae.cfg.d_sae),
        })

    return results


def compute_scaling_collapse(lambdas: np.ndarray, absorption_rates: np.ndarray,
                              dict_sizes: List[int], nu: float) -> Tuple[np.ndarray, float]:
    """
    Compute rescaled lambda values for scaling collapse analysis.

    The finite-size scaling hypothesis says:
    m(λ, N) = f(λ * N^(1/ν))

    We rescale by N^(-1/ν) to make curves collapse.
    Returns (rescaled_lambdas, collapse_quality)
    """
    rescaled_lambdas = []
    for lam, N in zip(lambdas, dict_sizes):
        rescaled = lam * (N ** (-1.0 / nu))
        rescaled_lambdas.append(rescaled)

    return np.array(rescaled_lambdas)


def compute_collapse_quality(rescaled_lambdas: np.ndarray, absorption_rates: np.ndarray,
                              dict_sizes: List[int]) -> float:
    """
    Compute R²-like metric for scaling collapse quality.

    After rescaling, points from different dict sizes with similar rescaled λ
    should have similar absorption rates. We bin by rescaled λ and compute
    the coefficient of variation within bins.
    """
    if len(rescaled_lambdas) < 4:
        return 0.0

    # Sort by rescaled lambda
    sort_idx = np.argsort(rescaled_lambdas)
    sorted_rescaled = rescaled_lambdas[sort_idx]
    sorted_abs = absorption_rates[sort_idx]
    sorted_sizes = np.array(dict_sizes)[sort_idx]

    # Bin into groups and compute within-bin variance
    n_bins = min(5, len(sorted_abs) // 3)
    if n_bins < 2:
        return 0.0

    bin_size = len(sorted_abs) // n_bins
    bin_variances = []

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(sorted_abs)
        bin_abs = sorted_abs[start:end]

        if len(bin_abs) > 1 and np.mean(bin_abs) > 1e-8:
            var = np.var(bin_abs) / (np.mean(bin_abs) ** 2)
            bin_variances.append(var)

    if not bin_variances:
        return 0.0

    # Average relative variance (lower is better for collapse)
    avg_relative_var = np.mean(bin_variances)
    quality = max(0, 1 / (1 + avg_relative_var))

    return quality


def get_text_samples(n_samples: int, seed: int = 42) -> List[str]:
    """Get text samples for activation extraction."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)

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

    samples = []
    while len(samples) < n_samples:
        for prompt in base_prompts:
            samples.append(prompt)
            if len(samples) >= n_samples:
                break

    return samples[:n_samples]


def main():
    print("=" * 70)
    print("FULL DICTIONARY SIZE SWEEP EXPERIMENT (H2 - Finite-Size Scaling)")
    print("=" * 70)
    print(f"Task ID: {TASK_ID}")
    print(f"Start time: {datetime.now().isoformat()}")
    print()
    print("NOTE: Using layer 8 feature-splitting SAEs because layer 6 only")
    print("      has d_sae=24576 in the standard gpt2-small-res-jb release.")
    print()

    # Create results directories
    os.makedirs(FULL_RESULTS_DIR, exist_ok=True)
    print(f"Results directory: {FULL_RESULTS_DIR}")

    # Write PID file immediately
    write_pid_file(FULL_RESULTS_DIR, TASK_ID)
    print(f"PID file written")

    # Total steps = dict_sizes * percentiles
    total_steps = len(DICT_SIZES) * len(SPARSE_PERCENTILES)
    report_progress(TASK_ID, FULL_RESULTS_DIR, epoch=0, total_epochs=1,
                    step=0, total_steps=total_steps)
    print(f"Progress tracking initialized")

    # Load model
    print("\nLoading GPT-2 Small model...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda")
    print(f"  Model loaded: {MODEL_NAME}")

    # Get text samples and compute activations once
    print(f"\nGetting {N_SAMPLES} text samples (seed={SEED})...")
    samples = get_text_samples(N_SAMPLES, SEED)

    print("Tokenizing samples...")
    tokens = model.to_tokens(samples).to("cuda")

    print(f"Computing activations from GPT-2 layer {LAYER}...")
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[HOOK_NAME])
        activations = cache[HOOK_NAME]
    print(f"  Activations shape: {activations.shape}")
    print()

    # Results storage for each dictionary size
    all_dict_size_results = {}

    # Process each dictionary size
    for i, d_sae in enumerate(DICT_SIZES):
        print(f"\n[{i+1}/{len(DICT_SIZES)}] Loading SAE with d_sae={d_sae}...")

        sae_result = load_sae_for_dict_size(SAE_RELEASE_FEATURE_SPLIT, d_sae, LAYER)

        if sae_result is None:
            print(f"  ERROR: Could not load SAE with d_sae={d_sae}")
            continue

        sae, cfg_dict = sae_result
        actual_d_sae = cfg_dict.get('d_sae', 'unknown')
        print(f"  SAE loaded: d_sae={actual_d_sae}, d_in={cfg_dict.get('d_in', 'unknown')}")

        # Compute absorption for each percentile threshold
        pct_results = compute_absorption_by_percentile(activations, sae, SPARSE_PERCENTILES)

        print(f"  Absorption by percentile threshold:")
        for r in pct_results:
            print(f"    p={r['percentile']:.0f} (threshold={r['lambda_threshold']:.4e}): "
                  f"m={r['absorption_rate']:.4f}, n_absorbed={r['n_absorbed']}")

        all_dict_size_results[d_sae] = {
            "d_sae": actual_d_sae,
            "d_in": cfg_dict.get('d_in', 'unknown'),
            "hook_name": HOOK_NAME,
            "lambda_results": pct_results,
        }

        # Clean up
        del sae
        torch.cuda.empty_cache()
        gc.collect()

        # Update progress
        current_step = (i + 1) * len(SPARSE_PERCENTILES)
        report_progress(TASK_ID, FULL_RESULTS_DIR, epoch=1, total_epochs=1,
                        step=current_step, total_steps=total_steps,
                        metric={"d_sae": d_sae})

    print("\n" + "=" * 70)
    print("SCALING COLLAPSE ANALYSIS")
    print("=" * 70)

    # Prepare data for scaling collapse
    # We use (100 - percentile) as the effective sparsity parameter
    # And absorption_rate as the order parameter

    collapse_data = {}
    for d_sae in DICT_SIZES:
        if d_sae not in all_dict_size_results:
            continue

        results = all_dict_size_results[d_sae]["lambda_results"]
        # Effective lambda = 100 - percentile (higher percentile = lower effective lambda)
        effective_lambdas = np.array([100 - r["percentile"] for r in results])
        absorption_rates = np.array([r["absorption_rate"] for r in results])

        collapse_data[d_sae] = {
            "effective_lambdas": effective_lambdas,
            "absorption_rates": absorption_rates,
        }

    # Test scaling collapse for each nu value
    best_nu = None
    best_collapse_quality = -1
    nu_results = {}

    for nu in NU_VALUES:
        print(f"\nTesting ν = {nu}...")

        all_rescaled = []
        all_absorption = []
        all_dict_size_list = []

        for d_sae in DICT_SIZES:
            if d_sae not in collapse_data:
                continue

            data = collapse_data[d_sae]
            eff_lambdas = data["effective_lambdas"]
            abs_rates = data["absorption_rates"]

            # Rescale: λ_eff = λ * N^(-1/ν)
            rescaled = eff_lambdas * (d_sae ** (-1.0 / nu))

            all_rescaled.extend(rescaled.tolist())
            all_absorption.extend(abs_rates.tolist())
            all_dict_size_list.extend([d_sae] * len(rescaled))

        # Compute collapse quality
        collapse_quality = compute_collapse_quality(
            np.array(all_rescaled),
            np.array(all_absorption),
            all_dict_size_list
        )

        nu_results[nu] = {
            "collapse_quality": collapse_quality,
            "n_points": len(all_rescaled),
        }

        print(f"  Collapse quality: {collapse_quality:.4f}")

        if collapse_quality > best_collapse_quality:
            best_collapse_quality = collapse_quality
            best_nu = nu

    print(f"\nBest ν: {best_nu} with collapse quality {best_collapse_quality:.4f}")

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
            "sae_release": SAE_RELEASE_FEATURE_SPLIT,
            "dict_sizes": DICT_SIZES,
            "sparsity_percentiles": SPARSE_PERCENTILES,
            "nu_values_tested": NU_VALUES,
            "note": "Using layer 8 feature-splitting SAEs (layer 6 only has d_sae=24576)",
        },
        "dict_size_results": all_dict_size_results,
        "scaling_collapse": {
            "nu_results": nu_results,
            "best_nu": best_nu,
            "best_collapse_quality": best_collapse_quality,
            "h2_supported": bool(best_collapse_quality > 0.5),
        },
        "h2_analysis": {
            "prediction": "Transition width δλ ∝ N^(-1/ν)",
            "test_result": "scaling_collapse" if best_collapse_quality > 0.5 else "no_collapse",
            "best_nu": best_nu,
            "confidence": best_collapse_quality,
            "note": "Using layer 8 feature-splitting SAEs instead of layer 6 due to availability",
        },
        "gpu": {
            "id": torch.cuda.current_device(),
            "name": torch.cuda.get_device_name(torch.cuda.current_device()),
        }
    }

    # Write results JSON
    output_path = f"{FULL_RESULTS_DIR}/dict_size_sweep.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {output_path}")

    # Mark task as done
    mark_task_done(TASK_ID, FULL_RESULTS_DIR, status="success",
                   summary=f"Dict size sweep complete. Best ν={best_nu}, quality={best_collapse_quality:.4f}")

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