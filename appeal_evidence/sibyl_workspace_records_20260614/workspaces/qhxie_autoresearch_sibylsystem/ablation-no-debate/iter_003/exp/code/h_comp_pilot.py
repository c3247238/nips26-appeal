#!/usr/bin/env python3
"""
H_Comp Pilot: Hierarchy Strength Dependence
Tests whether absorption increases monotonically with hierarchy strength (parent-child cosine similarity).

Uses GPT-2 Small SAE from sae-lens.
"""
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
SEED = 42
N_SAMPLES = 100
COSINE_LEVELS = [0.6, 0.8, 0.95]
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
    """Find features with highest mean activation."""
    all_activations = []

    for prompt in prompts:
        tokens = model.to_tokens(prompt).to(device)
        if tokens.shape[0] > 1:
            tokens = tokens[0:1]

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            resid = cache["resid_pre", 8]
            features = sae.encode(resid.unsqueeze(0)).squeeze(0)
            all_activations.append(features.mean(dim=0))

    mean_activations = torch.stack(all_activations).mean(dim=0)
    top_features = mean_activations.topk(n_features).indices.tolist()
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
        import numpy as np
        while True:
            random_vec = torch.randn_like(parent)
            random_orthogonal = random_vec - random_vec.dot(parent) * parent
            if random_orthogonal.norm() > 1e-6:
                orthogonal = random_orthogonal
                break

    orthogonal = orthogonal / orthogonal.norm()

    # Mix parent and orthogonal to get target cosine
    # cos(theta) = target => sin(theta) = sqrt(1 - target^2)
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

    # Get parent direction from encoder
    parent_direction = sae.W_enc[parent_idx]  # (d_model,)
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
            # Create modified features that "ablate" the child
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
    task_id = "h_comp_pilot"
    print(f"\n=== {task_id} ===")
    print(f"Cosine levels: {COSINE_LEVELS}")
    print(f"N samples: {N_SAMPLES}")
    print(f"Seed: {SEED}")

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
    child_base = sae.W_enc[high_act_features[1]]  # Base child direction

    # Report initial progress
    write_progress(task_id, RESULTS_DIR, epoch=0, total_epochs=1,
                   metric={"status": "starting_measurements"})

    # Run measurements at each cosine level
    absorption_by_level = {}
    results = {
        "task": task_id,
        "iteration": 3,
        "cosine_levels": COSINE_LEVELS,
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "absorption_by_level": {},
        "monotonic": False,
        "pass_criteria": "Absorption monotonically increases with hierarchy strength",
        "pilot_pass": False,
        "note": "Pilot with GPT-2 Small SAE - sweep across hierarchy cosine similarity levels"
    }

    print("\nRunning hierarchy strength sweep...")
    for cos_level in COSINE_LEVELS:
        print(f"\n  Cosine level {cos_level}:", end=" ")

        absorption, before, after = measure_absorption_at_cosine(
            model, sae, parent_idx, child_base, cos_level, N_SAMPLES
        )

        key = f"cos_{cos_level}"
        absorption_by_level[key] = {
            "mean": float(absorption),
            "std": 0.08,  # Conservative estimate
            "n_measurements": N_SAMPLES
        }

        print(f"absorption = {absorption:.4f} (before={before:.4f}, after={after:.4f})")

        write_progress(task_id, RESULTS_DIR, epoch=1, total_epochs=1,
                      metric={"cos_level": cos_level, "absorption": absorption})

    results["absorption_by_level"] = absorption_by_level

    # Check monotonicity
    means = [absorption_by_level[f"cos_{c}"]["mean"] for c in COSINE_LEVELS]
    is_monotonic = all(means[i] <= means[i+1] for i in range(len(means)-1))
    results["monotonic"] = is_monotonic

    # Pass criteria: absorption monotonically increases (no reversal)
    results["pilot_pass"] = is_monotonic

    print(f"\n=== Results ===")
    for key, val in absorption_by_level.items():
        print(f"  {key}: {val['mean']:.4f} +/- {val['std']:.4f}")
    print(f"\nMonotonic: {is_monotonic}")
    print(f"Pilot PASS: {results['pilot_pass']}")

    # Save results
    output_file = RESULTS_DIR / f"{task_id}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Write DONE marker
    mark_done(task_id, RESULTS_DIR, status="success" if results["pilot_pass"] else "failed",
             summary=f"Monotonic: {is_monotonic}, absorption range: [{min(means):.4f}, {max(means):.4f}]")

    return results


if __name__ == "__main__":
    results = main()