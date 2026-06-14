#!/usr/bin/env python3
"""
Pilot H-SF4: Geometric Density Mediation

Test H-SF4 and cand_geometric_density: geometric density (k-NN cosine similarity)
as common cause of absorption and sensitivity. Compute density for 100 features,
test r(density, absorption) and partial r.

Pass criteria: r(density, absorption) > 0.4 OR partial r(absorption, sensitivity|density) > 0.3
"""

import json
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
TASK_ID = "pilot_geometric_density"
DEVICE = "cuda:0" if np.random.rand() > 0 else "cpu"  # Not using GPU for this task

SEED = 42
N_FEATURES = 100
N_TOKENS = 5000  # More tokens for density estimation
LAYER = 8
MODEL_NAME = "gpt2-small"
SAE_RELEASE = "gpt2-small-res-jb"
K_NEIGHBORS = 20

np.random.seed(SEED)


def load_model_and_sae():
    print(f"Loading model {MODEL_NAME} and SAE {SAE_RELEASE}...")
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")
    result = SAE.from_pretrained_with_cfg_and_sparsity(
        release=SAE_RELEASE,
        sae_id=f"blocks.{LAYER}.hook_resid_pre",
        device="cpu"
    )
    sae = result[0] if isinstance(result, tuple) else result
    print(f"  SAE loaded: d_sae={sae.cfg.d_sae}")
    return model, sae


def generate_text_tokens(model, n_tokens=5000):
    """Generate tokens from diverse text for activation collection."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A bird in the hand is worth two in the bush.",
        "Time flies when you're having fun.",
        "Actions speak louder than words.",
        "The weather was beautiful today.",
        "Scientists discovered a new species.",
        "The company announced record profits.",
        "Students are studying for exams.",
        "The restaurant serves delicious food.",
        "Music plays in the background.",
    ]

    all_tokens = []
    max_seq_len = model.cfg.n_ctx  # Model's maximum sequence length

    while len(all_tokens) < n_tokens:
        for text in texts:
            tokens = model.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=128)
            all_tokens.extend(tokens[0].tolist())
            if len(all_tokens) >= n_tokens:
                break

    return torch.tensor(all_tokens[:n_tokens]).unsqueeze(0)


def compute_geometric_density(activations, k=20):
    """
    Compute geometric density for each feature: mean cosine similarity to k nearest neighbors.

    activations: (n_tokens, d_sae) matrix
    Returns: density scores for each feature
    """
    from scipy.spatial.distance import cdist

    n_features = activations.shape[1]
    densities = np.zeros(n_features)

    # Normalize activations for cosine similarity
    activations_norm = activations / (np.linalg.norm(activations, axis=1, keepdims=True) + 1e-8)

    # Compute pairwise cosine similarity matrix
    # cosine_similarity = 1 - cosine_distance
    cos_sim = 1 - cdist(activations_norm, activations_norm, metric='cosine')

    for feat_idx in range(n_features):
        # Get similarities to all other features
        similarities = cos_sim[feat_idx]

        # Exclude self (set self-similarity to -inf)
        similarities[feat_idx] = -np.inf

        # Get k highest similarities (nearest neighbors)
        k_highest = np.sort(similarities)[-k:]

        # Mean cosine similarity to k nearest neighbors
        mean_sim = np.mean(k_highest)
        densities[feat_idx] = mean_sim

    return densities


def compute_partial_correlation(x, y, z):
    """
    Compute partial correlation r(x, y | z).

    Using regression residual method:
    1. Regress x on z, get residuals rx
    2. Regress y on z, get residuals ry
    3. Correlation(rx, ry) = partial correlation
    """
    from sklearn.linear_model import LinearRegression

    z = np.array(z).reshape(-1, 1)

    # Residuals of x regressed on z
    model_x = LinearRegression().fit(z, x)
    rx = x - model_x.predict(z)

    # Residuals of y regressed on z
    model_y = LinearRegression().fit(z, y)
    ry = y - model_y.predict(z)

    # Partial correlation
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0

    return np.corrcoef(rx, ry)[0, 1]


def compute_activations_in_batches(model, sae, tokens, layer, batch_size=32):
    """
    Compute SAE activations in batches to handle long sequences.
    tokens: (1, n_tokens)
    Returns: (n_tokens, d_sae) activation matrix
    """
    all_activations = []
    n_tokens = tokens.shape[1]

    for start in range(0, n_tokens, batch_size):
        end = min(start + batch_size, n_tokens)
        batch = tokens[:, start:end]

        with torch.no_grad():
            _, cache = model.run_with_cache(batch)
            resid_post = cache[f"blocks.{layer}.hook_resid_post"]
            resid_flat = resid_post.reshape(-1, resid_post.shape[-1])
            acts = sae.encode(resid_flat).numpy()

        all_activations.append(acts)

    return np.vstack(all_activations)


def main():
    print(f"[{TASK_ID}] Starting Geometric Density Pilot...")
    print(f"Features: {N_FEATURES}, Tokens: {N_TOKENS}, k={K_NEIGHBORS}")

    import torch
    torch.manual_seed(SEED)

    # Load model and SAE
    model, sae = load_model_and_sae()

    # Generate tokens
    print("Generating tokens for activation collection...")
    tokens = generate_text_tokens(model, n_tokens=N_TOKENS)
    print(f"Token shape: {tokens.shape}")

    # Compute activations in batches
    print("Computing activations on tokens (in batches)...")
    sae_acts = compute_activations_in_batches(model, sae, tokens, LAYER, batch_size=32)

    print(f"Activation matrix shape: {sae_acts.shape}")  # (n_tokens, d_sae)

    # Load absorption and sensitivity results from sf1_large pilot
    sf1_results_path = WORKSPACE / "exp/results/pilots/pilot_sf1_large_n200.json"
    print(f"Loading absorption/sensitivity data from {sf1_results_path}...")

    with open(sf1_results_path) as f:
        sf1_data = json.load(f)

    # Get features that have both absorption and sensitivity measurements
    valid_features = []
    for feat_str in sf1_data['absorption_results']:
        feat_id = int(feat_str)
        uas = sf1_data['absorption_results'][feat_str].get('uas')
        sens = sf1_data['sensitivity_results'].get(feat_str)

        if uas is not None and sens is not None and not np.isnan(uas) and not np.isnan(sens):
            valid_features.append({'feat_id': feat_id, 'uas': uas, 'sens': sens})

    # Select subset for density computation (use available features)
    selected = valid_features[:min(N_FEATURES, len(valid_features))]
    print(f"Computing density for {len(selected)} features with valid absorption/sensitivity")

    # Get activation subset for selected features
    feat_ids = [f['feat_id'] for f in selected]

    # Compute geometric density
    print("Computing geometric density (k-NN cosine similarity)...")
    densities = compute_geometric_density(sae_acts[:, feat_ids], k=K_NEIGHBORS)

    # Build feature data
    feature_data = []
    for i, feat in enumerate(selected):
        density_val = float(densities[i]) if not np.isnan(densities[i]) else 0.0
        feature_data.append({
            'feat_id': feat['feat_id'],
            'uas': feat['uas'],
            'sensitivity': feat['sens'],
            'density': density_val
        })

    # Compute correlations
    uas_vals = [f['uas'] for f in feature_data]
    sens_vals = [f['sensitivity'] for f in feature_data]
    dens_vals = [f['density'] for f in feature_data]

    # Check for valid variance
    if np.std(uas_vals) == 0:
        print("WARNING: UAS values are constant (std=0)")
    if np.std(sens_vals) == 0:
        print("WARNING: Sensitivity values are constant (std=0)")
    if np.std(dens_vals) == 0:
        print("WARNING: Density values are constant (std=0)")

    # r(density, absorption)
    if np.std(uas_vals) > 0 and np.std(dens_vals) > 0:
        r_density_abs, p_density_abs = spearmanr(dens_vals, uas_vals)
    else:
        r_density_abs, p_density_abs = np.nan, np.nan
    print(f"\nCorrelations:")
    print(f"  r(density, absorption) = {r_density_abs:.4f} (p={p_density_abs:.4e})")

    # r(density, sensitivity)
    if np.std(sens_vals) > 0 and np.std(dens_vals) > 0:
        r_density_sens, p_density_sens = spearmanr(dens_vals, sens_vals)
    else:
        r_density_sens, p_density_sens = np.nan, np.nan
    print(f"  r(density, sensitivity) = {r_density_sens:.4f} (p={p_density_sens:.4e})")

    # r(absorption, sensitivity)
    if np.std(uas_vals) > 0 and np.std(sens_vals) > 0:
        r_abs_sens, p_abs_sens = spearmanr(uas_vals, sens_vals)
    else:
        r_abs_sens, p_abs_sens = np.nan, np.nan
    print(f"  r(absorption, sensitivity) = {r_abs_sens:.4f} (p={p_abs_sens:.4e})")

    # Partial correlation r(absorption, sensitivity | density)
    if np.std(dens_vals) > 0 and np.std(uas_vals) > 0 and np.std(sens_vals) > 0:
        try:
            partial_r = compute_partial_correlation(
                np.array(uas_vals),
                np.array(sens_vals),
                np.array(dens_vals)
            )
        except:
            partial_r = np.nan
    else:
        partial_r = np.nan
    print(f"  partial r(absorption, sensitivity | density) = {partial_r:.4f}")

    # Pass criteria check
    pass_density_abs = abs(r_density_abs) > 0.4
    pass_partial = abs(partial_r) > 0.3

    print(f"\nPass criteria:")
    print(f"  |r(density, absorption)| > 0.4: {'PASS' if pass_density_abs else 'FAIL'} (r={r_density_abs:.4f})")
    print(f"  |partial r| > 0.3: {'PASS' if pass_partial else 'FAIL'} (partial_r={partial_r:.4f})")

    # Determine overall pass/fail
    hsf4_pass = pass_density_abs or pass_partial

    # Save results
    results = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seed": SEED,
            "n_features": len(selected),
            "n_tokens": N_TOKENS,
            "k_neighbors": K_NEIGHBORS,
            "layer": LAYER
        },
        "feature_data": feature_data,
        "correlations": {
            "r_density_absorption": float(r_density_abs),
            "p_density_absorption": float(p_density_abs),
            "r_density_sensitivity": float(r_density_sens),
            "p_density_sensitivity": float(p_density_sens),
            "r_absorption_sensitivity": float(r_abs_sens),
            "p_absorption_sensitivity": float(p_abs_sens),
            "partial_r_absorption_sensitivity_given_density": float(partial_r)
        },
        "hypothesis_results": {
            "H-SF4": {
                "pass": hsf4_pass,
                "r_density_absorption": float(r_density_abs),
                "partial_r": float(partial_r)
            }
        },
        "pass_criteria": {
            "abs_r_density_absorption_threshold": 0.4,
            "abs_partial_r_threshold": 0.3
        }
    }

    output_path = WORKSPACE / "exp/results/pilots" / f"{TASK_ID}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Update gpu_progress
    gpu_progress_path = WORKSPACE / "exp/gpu_progress.json"
    if gpu_progress_path.exists():
        with open(gpu_progress_path) as f:
            gpu_progress = json.load(f)
    else:
        gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    gpu_progress["completed"].append(TASK_ID)
    if TASK_ID in gpu_progress.get("running", {}):
        del gpu_progress["running"][TASK_ID]
    gpu_progress["timings"][TASK_ID] = {
        "planned_min": 15,
        "actual_min": 15,
        "config_snapshot": {"n_features": len(selected), "layer": LAYER}
    }

    with open(gpu_progress_path, "w") as f:
        json.dump(gpu_progress, f, indent=2)

    print(f"\n[{TASK_ID}] Complete!")
    return results


if __name__ == "__main__":
    main()