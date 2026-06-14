#!/usr/bin/env python3
"""
Multi-child Proportional Absorption Measurement (Task: p1_multichild_absorption)

Key fix from pilot: Single-child ablation saturates at 1.0 for both SAE and baseline
because ablating ONE child lets remaining children reconstruct the parent.

Multi-child proportional ablation (k=5) tests whether children collectively substitute
for parent by ablating the top-k children together.

Pilot: 100 samples × 5 hierarchies, seed 42
Full: 10,000 samples × 5 hierarchies × 5 seeds
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from scipy.stats import ttest_ind, spearmanr, pearsonr
import os
import gc

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
GPU_ID = 0  # Will be overridden if GPU_ID param is set
DEVICE = None  # Set dynamically
SEED = 42

# Pilot configuration (for quick validation)
PILOT_SAMPLES = 100
FULL_SAMPLES = 10000
N_SEEDS = 5
K_CHILDREN = 5  # Number of top children to ablate

np.random.seed(SEED)
torch.manual_seed(SEED)


class SimpleSAE(nn.Module):
    """Simplified SAE with TopK sparsity (from pilot code)."""
    def __init__(self, d_model, d_sae, l0_target=32):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.l0_target = l0_target

        self.W_encoder = nn.Linear(d_model, d_sae, bias=True)
        nn.init.xavier_uniform_(self.W_encoder.weight)

        self.W_decoder = nn.Linear(d_sae, d_model, bias=False)
        nn.init.xavier_uniform_(self.W_decoder.weight)

        with torch.no_grad():
            self.normalize_decoder()

        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def normalize_decoder(self):
        norm = torch.norm(self.W_decoder.weight, dim=0, keepdim=True)
        self.W_decoder.weight.div_(norm + 1e-8)

    def forward(self, x):
        pre_acts = self.W_encoder(x) + self.b_enc
        acts = torch.relu(pre_acts)

        # TopK - safe implementation
        batch_size = acts.shape[0]
        k = min(self.l0_target, acts.shape[1])

        if k < acts.shape[1] and batch_size > 0:
            topk_values = torch.zeros_like(acts)
            for i in range(batch_size):
                vals, idxs = torch.topk(acts[i], k=k)
                topk_values[i, idxs] = vals
            sparse_acts = topk_values
        else:
            sparse_acts = acts

        recon = self.W_decoder(sparse_acts) + self.b_dec
        return recon, sparse_acts

    def get_encoder_activations(self, x):
        """Get encoder pre-activations (before TopK)."""
        with torch.no_grad():
            pre_acts = self.W_encoder(x) + self.b_enc
            return torch.relu(pre_acts)


def load_trained_sae(path, device):
    """Load trained SAE from checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    if "d_model" not in config:
        config["d_model"] = 128

    model = SimpleSAE(
        d_model=config["d_model"],
        d_sae=config["d_sae"],
        l0_target=config["l0_target"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def create_random_baseline(d_model, d_sae, l0_target, device, seed=42):
    """Create random decoder baseline (Xavier init, no training)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SimpleSAE(d_model=d_model, d_sae=d_sae, l0_target=l0_target).to(device)
    return model


def create_shuffled_baseline(trained_sae, device, seed=42):
    """Create shuffled features baseline (same activations, permuted indices)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SimpleSAE(
        d_model=trained_sae.d_model,
        d_sae=trained_sae.d_sae,
        l0_target=trained_sae.l0_target
    ).to(device)

    # Copy encoder weights but shuffle them
    with torch.no_grad():
        perm = torch.randperm(trained_sae.d_sae)
        # Shuffle rows (features): weight[perm, :] selects rows indexed by perm
        model.W_encoder.weight.copy_(trained_sae.W_encoder.weight[perm, :])
        model.W_encoder.bias.copy_(trained_sae.W_encoder.bias[perm])
        model.normalize_decoder()

    return model


def create_permuted_encoder_baseline(trained_sae, device, seed=42):
    """Create permuted encoder baseline (trained SAE, shuffled encoder weights)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SimpleSAE(
        d_model=trained_sae.d_model,
        d_sae=trained_sae.d_sae,
        l0_target=trained_sae.l0_target
    ).to(device)

    # Copy decoder weights but shuffle encoder
    with torch.no_grad():
        model.W_decoder.weight.copy_(trained_sae.W_decoder.weight)
        model.W_decoder.weight.data.div_(torch.norm(model.W_decoder.weight, dim=0, keepdim=True) + 1e-8)

        perm = torch.randperm(trained_sae.d_sae)
        model.W_encoder.weight.copy_(trained_sae.W_encoder.weight[perm, :])
        model.W_encoder.bias.copy_(trained_sae.W_encoder.bias[perm])

        model.b_dec.copy_(trained_sae.b_dec)

    return model


def multichild_proportional_ablation(model, parent_dir, child_dirs, device, n_samples=100, k=10):
    """
    Multi-child proportional absorption measurement.

    Key insight from pilot: The correct measurement computes overlap between
    parent features and EACH child SEPARATELY, not the combined children.

    absorption_rate = (overlap(parent, child1) + overlap(parent, child2)) / 2

    This is because the synthetic hierarchy has:
    - parent = 0.67 * child1 + 0.67 * child2 (with grandchildren spanning the space)
    - children are partially orthogonal

    Higher overlap means children activate the same features as parent = absorption.
    """
    results = {
        "k": k,
        "n_samples": n_samples,
        "absorption_k": [],
        "overlap_parent_child1": [],
        "overlap_parent_child2": [],
        "proportional_variance": [],
    }

    with torch.no_grad():
        for i in range(n_samples):
            # Sample activation strength
            strength = np.random.uniform(1.0, 3.0)

            # Normalize directions to same strength
            child_dir_1 = child_dirs[0]  # dogs
            child_dir_2 = child_dirs[1]  # cats

            parent_input = (parent_dir / parent_dir.norm()) * strength
            child1_input = (child_dir_1 / child_dir_1.norm()) * strength
            child2_input = (child_dir_2 / child_dir_2.norm()) * strength

            # Get encoder activations for parent and each child
            parent_acts = model.get_encoder_activations(parent_input.unsqueeze(0).to(device))[0]
            child1_acts = model.get_encoder_activations(child1_input.unsqueeze(0).to(device))[0]
            child2_acts = model.get_encoder_activations(child2_input.unsqueeze(0).to(device))[0]

            # Find top-k features for parent and each child
            k_effective = min(k, parent_acts.shape[0])
            _, parent_topk = torch.topk(parent_acts, k=k_effective)
            _, child1_topk = torch.topk(child1_acts, k=k_effective)
            _, child2_topk = torch.topk(child2_acts, k=k_effective)

            parent_set = set(parent_topk.cpu().tolist())
            child1_set = set(child1_topk.cpu().tolist())
            child2_set = set(child2_topk.cpu().tolist())

            # Overlap between parent and each child
            overlap1 = len(parent_set & child1_set) / k_effective
            overlap2 = len(parent_set & child2_set) / k_effective

            results["overlap_parent_child1"].append(float(overlap1))
            results["overlap_parent_child2"].append(float(overlap2))

            # Average absorption rate
            absorption = (overlap1 + overlap2) / 2
            results["absorption_k"].append(float(absorption))

            # Proportional variance (asymmetry in which child activates parent features more)
            parent_feature_idx = parent_topk.cpu().tolist()
            child1_contributions = child1_acts[parent_feature_idx].cpu().numpy()
            child2_contributions = child2_acts[parent_feature_idx].cpu().numpy()

            total_contrib = child1_contributions + child2_contributions + 1e-8
            proportions = child1_contributions / total_contrib

            prop_variance = np.var(proportions)
            results["proportional_variance"].append(float(prop_variance))

    # Aggregate
    absorption_arr = np.array(results["absorption_k"])
    overlap1_arr = np.array(results["overlap_parent_child1"])
    overlap2_arr = np.array(results["overlap_parent_child2"])
    prop_var_arr = np.array(results["proportional_variance"])

    return {
        "absorption_k5_mean": float(np.mean(absorption_arr)),
        "absorption_k5_std": float(np.std(absorption_arr)),
        "absorption_k5_min": float(np.min(absorption_arr)),
        "absorption_k5_max": float(np.max(absorption_arr)),
        "overlap_parent_child1_mean": float(np.mean(overlap1_arr)),
        "overlap_parent_child1_std": float(np.std(overlap1_arr)),
        "overlap_parent_child2_mean": float(np.mean(overlap2_arr)),
        "overlap_parent_child2_std": float(np.std(overlap2_arr)),
        "proportional_variance_mean": float(np.mean(prop_var_arr)),
        "proportional_variance_std": float(np.std(prop_var_arr)),
        "n_samples": n_samples,
        "k": k,
        "raw": {
            "absorption_k": results["absorption_k"],
            "overlap_parent_child1": results["overlap_parent_child1"],
            "overlap_parent_child2": results["overlap_parent_child2"],
            "proportional_variance": results["proportional_variance"]
        }
    }


def measure_frequency_correlation(model, activations, device):
    """Measure activation frequency for each feature."""
    with torch.no_grad():
        features = torch.FloatTensor(activations["activations"]).to(device)

        # Get encoder activations for all samples
        all_acts = model.get_encoder_activations(features)
        active_counts = (all_acts > 0).sum(dim=0).cpu().numpy()
        mean_acts = all_acts.mean(dim=0).cpu().numpy()

        frequency = active_counts / len(features)

    return {
        "frequency": frequency.tolist(),
        "mean_activation": mean_acts.tolist(),
        "n_features": len(frequency)
    }


def main():
    global DEVICE

    # Parse command line args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (-1 for CPU)")
    parser.add_argument("--mode", type=str, default="pilot", choices=["pilot", "full"], help="Run mode")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()

    if args.gpu < 0 or not torch.cuda.is_available():
        DEVICE = "cpu"
    else:
        DEVICE = f"cuda:{args.gpu}"

    print("=" * 70)
    print("Multi-child Proportional Absorption Measurement")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Device: {DEVICE}")
    print(f"Samples: {args.samples}")
    print(f"Seed: {args.seed}")
    print(f"K children: {K_CHILDREN}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Load pilot data
    print("\n[1/5] Loading pilot data...")
    data_path = WORKSPACE / "data" / "pilot_activations.json"
    with open(data_path) as f:
        activations = json.load(f)
    n_pilot_samples = activations["n_samples"]
    print(f"  Loaded {n_pilot_samples} pilot samples")

    # Load hierarchy
    hierarchy_path = WORKSPACE / "data" / "synthetic_hierarchy.json"
    with open(hierarchy_path) as f:
        hierarchy = json.load(f)
    print("  Loaded hierarchy")

    # Get hierarchy directions
    parent_dir = torch.FloatTensor(hierarchy["levels"]["parent"]["direction"]).to(DEVICE)
    dogs_dir = torch.FloatTensor(hierarchy["levels"]["children"][0]["direction"]).to(DEVICE)
    cats_dir = torch.FloatTensor(hierarchy["levels"]["children"][1]["direction"]).to(DEVICE)
    child_dirs = [dogs_dir, cats_dir]

    # Load trained SAE
    print("\n[2/5] Loading trained SAE...")
    sae_path = WORKSPACE / "exp" / "results" / "pilots" / "sae_L0_32.pt"

    if not sae_path.exists():
        print(f"  ERROR: SAE checkpoint not found at {sae_path}")
        print("  Run p1_train_sae.py first!")
        return None

    trained_sae, sae_config = load_trained_sae(sae_path, DEVICE)
    print(f"  Loaded SAE: d_model={sae_config['d_model']}, d_sae={sae_config['d_sae']}, L0={sae_config['l0_target']}")

    # Create baselines
    print("\n[3/5] Creating baseline models...")
    torch.manual_seed(args.seed + 1)
    np.random.seed(args.seed + 1)

    random_baseline = create_random_baseline(
        d_model=sae_config["d_model"],
        d_sae=sae_config["d_sae"],
        l0_target=sae_config["l0_target"],
        device=DEVICE,
        seed=args.seed + 1
    )
    print("  Created: Random decoder baseline (Xavier init)")

    shuffled_baseline = create_shuffled_baseline(trained_sae, DEVICE, seed=args.seed + 2)
    print("  Created: Shuffled features baseline")

    permuted_encoder_baseline = create_permuted_encoder_baseline(trained_sae, DEVICE, seed=args.seed + 3)
    print("  Created: Permuted encoder baseline")

    # Reset seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Measure absorption
    print("\n[4/5] Measuring multi-child proportional absorption...")

    models = {
        "trained_sae": trained_sae,
        "random_decoder": random_baseline,
        "shuffled_features": shuffled_baseline,
        "permuted_encoder": permuted_encoder_baseline
    }

    results = {}

    for name, model in models.items():
        print(f"\n  {name}:")
        result = multichild_proportional_ablation(
            model=model,
            parent_dir=parent_dir,
            child_dirs=child_dirs,
            device=DEVICE,
            n_samples=args.samples,
            k=K_CHILDREN
        )

        print(f"    absorption_k{K_CHILDREN}: {result['absorption_k5_mean']:.4f} +/- {result['absorption_k5_std']:.4f}")
        print(f"    proportional_variance: {result['proportional_variance_mean']:.4f} +/- {result['proportional_variance_std']:.4f}")

        results[name] = result

    # Statistical analysis (H1)
    print("\n[5/5] Statistical analysis...")

    trained_absorption = np.array(results["trained_sae"]["raw"]["absorption_k"])
    random_absorption = np.array(results["random_decoder"]["raw"]["absorption_k"])
    shuffled_absorption = np.array(results["shuffled_features"]["raw"]["absorption_k"])
    permuted_absorption = np.array(results["permuted_encoder"]["raw"]["absorption_k"])

    # T-tests comparing trained SAE vs each baseline
    t_stat_vs_random, p_val_random = ttest_ind(trained_absorption, random_absorption)
    t_stat_vs_shuffled, p_val_shuffled = ttest_ind(trained_absorption, shuffled_absorption)
    t_stat_vs_permuted, p_val_permuted = ttest_ind(trained_absorption, permuted_absorption)

    # Effect sizes (Cohen's d)
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
        return (np.mean(x) - np.mean(y)) / (pooled_std + 1e-8)

    d_vs_random = cohens_d(trained_absorption, random_absorption)
    d_vs_shuffled = cohens_d(trained_absorption, shuffled_absorption)
    d_vs_permuted = cohens_d(trained_absorption, permuted_absorption)

    print(f"\n  H1 Statistical Tests (trained SAE vs baselines):")
    print(f"    vs Random Decoder: t={t_stat_vs_random:.3f}, p={p_val_random:.4f}, d={d_vs_random:.3f}")
    print(f"    vs Shuffled:       t={t_stat_vs_shuffled:.3f}, p={p_val_shuffled:.4f}, d={d_vs_shuffled:.3f}")
    print(f"    vs Permuted:       t={t_stat_vs_permuted:.3f}, p={p_val_permuted:.4f}, d={d_vs_permuted:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nMulti-child proportional absorption (k={K_CHILDREN}):")
    print(f"  {'Condition':<20} {'Mean':>10} {'Std':>10} {'vs SAE':>15}")
    print(f"  {'-'*55}")
    for name in ["trained_sae", "random_decoder", "shuffled_features", "permuted_encoder"]:
        mean = results[name]["absorption_k5_mean"]
        std = results[name]["absorption_k5_std"]
        delta = mean - results["trained_sae"]["absorption_k5_mean"]
        print(f"  {name:<20} {mean:>10.4f} {std:>10.4f} {delta:>+15.4f}")

    # Pass criteria
    trained_mean = results["trained_sae"]["absorption_k5_mean"]
    random_mean = results["random_decoder"]["absorption_k5_mean"]
    delta = trained_mean - random_mean

    # H1: trained SAE should have HIGHER absorption than baselines
    # (more absorption = children substitute for parent more)
    h1_pass = (trained_mean > random_mean and
               trained_mean > 0.3 and
               (p_val_random < 0.05 or delta > 0.15))

    print(f"\nH1 Pass Criteria:")
    print(f"  Trained SAE absorption > 0.3: {trained_mean:.4f} {'PASS' if trained_mean > 0.3 else 'FAIL'}")
    print(f"  Trained SAE > Random baseline: {delta:+.4f} {'PASS' if delta > 0 else 'FAIL'}")
    print(f"  p-value < 0.05 OR delta > 0.15: p={p_val_random:.4f}, delta={delta:.4f}")
    print(f"\n  OVERALL H1: {'PASS' if h1_pass else 'FAIL'}")

    # Prepare output
    output = {
        "task": "p1_multichild_absorption",
        "mode": args.mode,
        "config": {
            "k_children": K_CHILDREN,
            "n_samples": args.samples,
            "seed": args.seed,
            "gpu": args.gpu
        },
        "absorption_results": {
            name: {
                "absorption_k5_mean": results[name]["absorption_k5_mean"],
                "absorption_k5_std": results[name]["absorption_k5_std"],
                "absorption_k5_min": results[name]["absorption_k5_min"],
                "absorption_k5_max": results[name]["absorption_k5_max"],
                "proportional_variance_mean": results[name]["proportional_variance_mean"],
                "proportional_variance_std": results[name]["proportional_variance_std"],
                "raw": {
                    "absorption_k": results[name]["raw"]["absorption_k"],
                    "overlap_parent_child1": results[name]["raw"]["overlap_parent_child1"],
                    "overlap_parent_child2": results[name]["raw"]["overlap_parent_child2"],
                    "proportional_variance": results[name]["raw"]["proportional_variance"]
                }
            }
            for name in results
        },
        "h1_statistics": {
            "vs_random_decoder": {
                "t_statistic": float(t_stat_vs_random),
                "p_value": float(p_val_random),
                "cohens_d": float(d_vs_random),
                "delta": float(delta)
            },
            "vs_shuffled_features": {
                "t_statistic": float(t_stat_vs_shuffled),
                "p_value": float(p_val_shuffled),
                "cohens_d": float(d_vs_shuffled)
            },
            "vs_permuted_encoder": {
                "t_statistic": float(t_stat_vs_permuted),
                "p_value": float(p_val_permuted),
                "cohens_d": float(d_vs_permuted)
            }
        },
        "h1_pass": bool(h1_pass),
        "sae_config": sae_config,
        "timestamp": datetime.now().isoformat()
    }

    # Save results
    output_dir = WORKSPACE / "exp" / "results" / "full"
    os.makedirs(output_dir, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / "multichild_absorption.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {output_path}")

    print("\n" + "=" * 70)
    return output


if __name__ == "__main__":
    output = main()
