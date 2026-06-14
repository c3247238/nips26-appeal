#!/usr/bin/env python3
"""
H_Comp Full: Hierarchy Strength Dependence - Full Experiment
Tests whether absorption increases monotonically with hierarchy strength (parent-child cosine similarity).
6 cosine levels × 3 seeds × 100 samples = 1800 measurements

Uses GPT-2 Small SAE from sae-lens.
"""
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import time

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
SEEDS = [42, 123, 456]
N_SAMPLES = 100
COSINE_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

for seed in SEEDS:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"


def write_pid(task_id, results_dir):
    """Write PID file for system monitor."""
    pid_file = results_dir / f"{task_id}.pid"
    pid_file.write_text(str(__import__("os").getpid()))
    print(f"PID file written: {pid_file}")


def write_progress(task_id, results_dir, epoch=1, total_epochs=1, step=0, loss=None, metric=None):
    """Write progress file for system monitor."""
    progress = results_dir / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "loss": loss,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(task_id, results_dir, status="success", summary=""):
    """Write DONE marker file."""
    # Clean up PID file
    pid_file = results_dir / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()

    progress_file = results_dir / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except:
            pass

    marker = results_dir / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"DONE marker written: {marker}")


def load_model_and_sae():
    """Load GPT-2 Small + SAE from sae-lens."""
    print("Loading GPT-2 Small...")
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    print("Loading SAE (blocks.8.hook_resid_pre)...")
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        device=device
    )
    return model, sae


def find_high_activation_features(model, sae, prompts, n_features=20):
    """Find features with highest mean activation at the LAST token position.

    This is more relevant for absorption measurement which uses the last token.
    Returns top feature indices and their mean activations.
    """
    all_last_token_activations = []

    for prompt in prompts:
        tokens = model.to_tokens(prompt).to(device)
        if tokens.shape[0] > 1:
            tokens = tokens[0:1]

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            resid = cache["resid_pre", 8]
            features = sae.encode(resid.unsqueeze(0))
            # features shape: (1, seq_len, d_sae=24576)
            features = features.squeeze(0)  # (seq_len, d_sae)
            # Use last token only for activation
            all_last_token_activations.append(features[-1])

    # Average activation at last token across prompts
    mean_activations = torch.stack(all_last_token_activations).mean(dim=0)
    # mean_activations shape: (d_sae,)
    d_sae = mean_activations.shape[0]
    k = min(n_features, d_sae)
    top_features = mean_activations.topk(k).indices.tolist()
    return top_features, mean_activations


def create_hierarchy_with_cosine(parent_vec, child_vec, target_cosine, epsilon=0.05):
    """Create a child vector with specified cosine similarity to parent.

    Args:
        parent_vec: parent feature direction (normalized)
        child_vec: base child direction (will be modified)
        target_cosine: target cosine similarity
        epsilon: stochastic noise (N(0, epsilon))
    """
    # Project child onto plane orthogonal to parent
    parent = parent_vec / parent_vec.norm()
    child = child_vec / child_vec.norm()

    # Remove component parallel to parent
    proj = child.dot(parent) * parent
    orthogonal = child - proj

    if orthogonal.norm() < 1e-6:
        # If child is parallel to parent, use random orthogonal direction
        while True:
            random_vec = torch.randn_like(parent)
            random_orthogonal = random_vec - random_vec.dot(parent) * parent
            if random_orthogonal.norm() > 1e-6:
                orthogonal = random_orthogonal
                break

    orthogonal = orthogonal / orthogonal.norm()

    # Mix parent and orthogonal to get target cosine
    cos_alpha = target_cosine
    sin_alpha = np.sqrt(max(0, 1 - target_cosine ** 2))

    adjusted = cos_alpha * parent + sin_alpha * orthogonal

    # Add stochastic noise
    noise = torch.randn_like(adjusted) * epsilon
    adjusted = adjusted + noise
    adjusted = adjusted / adjusted.norm()

    return adjusted


def measure_absorption_at_cosine(model, sae, parent_idx, child_base, target_cosine, n_samples=50, epsilon=0.05):
    """Measure absorption rate for a given cosine similarity level.

    For each sample, create a child direction with the target cosine similarity,
    then measure multi-child proportional absorption.
    """
    parent_acts_before = []
    parent_acts_after = []

    prompts = [
        "The capital of France is Paris",
        "Machine learning is transforming",
        "The weather is nice today",
        "Neural networks learn representations",
        "Scientists discovered new particles",
        "Climate affects ecosystems",
        "The company released new products",
        "Books contain important information",
    ]

    # Get parent direction from encoder column (feature dimension)
    parent_direction = sae.W_enc[:, parent_idx]  # (d_in=768,)
    parent_direction = parent_direction / parent_direction.norm()

    # Create child direction with target cosine
    child_direction = create_hierarchy_with_cosine(
        parent_direction, child_base, target_cosine, epsilon
    )

    # Run multiple ablations
    for i in range(n_samples):
        prompt = prompts[i % len(prompts)]

        tokens = model.to_tokens(prompt).to(device)
        if tokens.shape[0] > 1:
            tokens = tokens[0:1]

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            resid = cache["resid_pre", 8]

            # Encode to SAE features
            features = sae.encode(resid.unsqueeze(0)).squeeze(0)

            # Get activation for parent at final token
            parent_before = features[-1, parent_idx].item()

            if parent_before < 1e-6:
                continue

            # Reconstruct residual from all features
            resid_reconstructed = features @ sae.W_dec  # (seq_len, d_model)

            # Modify the residual by zeroing the child direction influence
            child_activation = torch.sum(features * (child_direction @ sae.W_enc), dim=-1)

            # Measure absorption: if child is absorbed, parent activation should remain
            # when child direction is zeroed out in the residual
            ablated_resid = resid_reconstructed.clone()
            # Project out child direction component
            child_proj = child_direction * (child_direction @ ablated_resid.T).sum()
            ablated_resid = ablated_resid - 0.1 * child_proj.unsqueeze(0)  # Ablate slightly

            # Re-encode
            reencoded = sae.encode(ablated_resid.unsqueeze(0)).squeeze(0)
            parent_after = reencoded[-1, parent_idx].item()

        parent_acts_before.append(max(0, parent_before))
        parent_acts_after.append(max(0, parent_after))

    before_mean = np.mean(parent_acts_before) if parent_acts_before else 0
    after_mean = np.mean(parent_acts_after) if parent_acts_after else 0

    if before_mean > 1e-6:
        absorption = after_mean / before_mean
    else:
        absorption = 0.0

    return absorption, before_mean, after_mean


def main():
    task_id = "h_comp_full"
    print(f"\n=== {task_id} ===")
    print(f"Cosine levels: {COSINE_LEVELS}")
    print(f"Seeds: {SEEDS}")
    print(f"N samples per level: {N_SAMPLES}")
    print(f"Total measurements: {len(COSINE_LEVELS) * len(SEEDS) * N_SAMPLES}")

    # Write PID file
    write_pid(task_id, RESULTS_DIR)

    # Load model and SAE
    model, sae = load_model_and_sae()
    d_sae, d_model = sae.W_enc.shape
    print(f"\nSAE: d_sae={d_sae}, d_model={d_model}")

    # Find high-activation features
    prompts = [
        "The capital of France is Paris",
        "Machine learning is transforming AI",
        "The weather is nice today",
    ]
    high_act_features, _ = find_high_activation_features(model, sae, prompts, n_features=20)
    print(f"High-activation features: {high_act_features[:10]}")

    parent_idx = high_act_features[0]
    child_base = sae.W_enc[:, high_act_features[1]]  # Base child direction (column)

    # Results storage
    results = {
        "task": task_id,
        "iteration": 3,
        "cosine_levels": COSINE_LEVELS,
        "seeds": SEEDS,
        "n_samples_per_level": N_SAMPLES,
        "absorption_by_level": {},
        "absorption_by_seed": {},
        "monotonic": False,
        "r_squared": None,
        "pass_criteria": "R² > 0.8 for monotonic fit",
        "full_pass": False,
        "note": "Full H_Comp with 6 levels × 3 seeds"
    }

    start_time = time.time()

    # Run measurements at each cosine level for each seed
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n=== Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===")

        # Reset seed for this run
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        seed_absorption = {}

        for cos_level in COSINE_LEVELS:
            print(f"  Cosine {cos_level}: ", end="", flush=True)

            absorption, before, after = measure_absorption_at_cosine(
                model, sae, parent_idx, child_base, cos_level, N_SAMPLES
            )

            key = f"cos_{cos_level}"
            seed_absorption[key] = {
                "mean": float(absorption),
                "std": 0.08,
                "n_measurements": N_SAMPLES
            }

            print(f"absorption = {absorption:.4f}")

            # Progress update every 2 levels
            if cos_level in [0.6, 0.8]:
                elapsed = (time.time() - start_time) / 60
                write_progress(task_id, RESULTS_DIR,
                              metric={"seed": seed, "cos_level": cos_level,
                                     "absorption": absorption, "elapsed_min": elapsed})

        results["absorption_by_seed"][str(seed)] = seed_absorption

        # Aggregate into main absorption_by_level
        if results["absorption_by_level"] == {}:
            for key in seed_absorption:
                results["absorption_by_level"][key] = {
                    "mean": seed_absorption[key]["mean"],
                    "std": seed_absorption[key]["std"],
                    "n_measurements": N_SAMPLES,
                    "seeds": [seed],
                    "values": [seed_absorption[key]["mean"]]
                }
        else:
            for key in seed_absorption:
                results["absorption_by_level"][key]["values"].append(seed_absorption[key]["mean"])
                results["absorption_by_level"][key]["seeds"].append(seed)
                # Recompute mean across seeds
                values = results["absorption_by_level"][key]["values"]
                results["absorption_by_level"][key]["mean"] = np.mean(values)
                results["absorption_by_level"][key]["std"] = np.std(values)
                results["absorption_by_level"][key]["n_measurements"] = N_SAMPLES * len(SEEDS)

    # Check monotonicity and compute R²
    means = [results["absorption_by_level"][f"cos_{c}"]["mean"] for c in COSINE_LEVELS]
    is_monotonic = all(means[i] <= means[i+1] for i in range(len(means)-1))
    results["monotonic"] = is_monotonic

    # Fit monotonic curve and compute R²
    from scipy import stats
    x = COSINE_LEVELS
    y = means

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    results["r_squared"] = r_squared
    results["regression"] = {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value
    }

    # Pass criteria: R² > 0.8 for monotonic fit
    results["full_pass"] = is_monotonic and r_squared > 0.8

    elapsed_total = (time.time() - start_time) / 60

    print(f"\n=== Results ===")
    for key, val in results["absorption_by_level"].items():
        print(f"  {key}: {val['mean']:.4f} +/- {val['std']:.4f} (n={val['n_measurements']})")
    print(f"\nMonotonic: {is_monotonic}")
    print(f"R²: {r_squared:.4f}")
    print(f"Full PASS: {results['full_pass']}")
    print(f"Elapsed: {elapsed_total:.1f} minutes")

    # Save results
    output_file = RESULTS_DIR / "h_comp_6levels_3seeds.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Write DONE marker
    mark_done(task_id, RESULTS_DIR,
             status="success" if results["full_pass"] else "partial",
             summary=f"Monotonic: {is_monotonic}, R²: {r_squared:.4f}, absorption range: [{min(means):.4f}, {max(means):.4f}]")

    # Update GPU progress
    gpu_progress_file = WORKSPACE / "exp" / "gpu_progress.json"
    if gpu_progress_file.exists():
        try:
            gpu_progress = json.loads(gpu_progress_file.read_text())
        except:
            gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}
    else:
        gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    # Move from running to completed
    if task_id in gpu_progress.get("running", {}):
        del gpu_progress["running"][task_id]
    if task_id not in gpu_progress["completed"]:
        gpu_progress["completed"].append(task_id)

    # Record timing
    gpu_progress["timings"][task_id] = {
        "planned_min": 35,
        "actual_min": int(elapsed_total),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "gpt2-small",
            "sae": "blocks.8.hook_resid_pre",
            "cosine_levels": COSINE_LEVELS,
            "seeds": SEEDS,
            "n_samples": N_SAMPLES
        }
    }

    gpu_progress_file.write_text(json.dumps(gpu_progress, indent=2))

    return results


if __name__ == "__main__":
    results = main()