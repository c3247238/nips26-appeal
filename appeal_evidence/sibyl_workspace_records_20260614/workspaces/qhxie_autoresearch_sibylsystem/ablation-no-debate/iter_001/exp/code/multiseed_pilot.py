#!/usr/bin/env python3
"""
Multi-child Multi-seed Pilot: Validate H1 with Multiple Seeds

Address zero-variance concern from pilot (std=0.0 for trained SAE).
Validate with multiple seeds (42, 43, 44) to ensure absorption is not
purely deterministic.

Expected: Non-zero variance across seeds if absorption is not purely geometric.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from scipy.stats import ttest_ind, f_oneway
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 43, 44]
PILOT_SAMPLES = 100
K_CHILDREN = 5

np.random.seed(42)
torch.manual_seed(42)


class SimpleSAE(nn.Module):
    """Simplified SAE with TopK sparsity."""
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

    def get_encoder_activations(self, x):
        with torch.no_grad():
            pre_acts = self.W_encoder(x) + self.b_enc
            return torch.relu(pre_acts)

    def forward(self, x):
        pre_acts = self.W_encoder(x) + self.b_enc
        acts = torch.relu(pre_acts)

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

    def state_dict(self):
        state = {
            'W_encoder.weight': self.W_encoder.weight,
            'W_encoder.bias': self.W_encoder.bias,
            'W_decoder.weight': self.W_decoder.weight,
            'b_enc': self.b_enc,
            'b_dec': self.b_dec
        }
        return state


def train_sae(d_model, d_sae, l0_target, device, train_data, n_steps=500, lr=1e-3, seed=42):
    """Train SAE with given random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SimpleSAE(d_model=d_model, d_sae=d_sae, l0_target=l0_target).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    data_tensor = torch.FloatTensor(train_data).to(device)

    for step in range(n_steps):
        optimizer.zero_grad()

        # Sample batch
        batch_size = min(256, len(data_tensor))
        indices = torch.randint(0, len(data_tensor), (batch_size,))
        batch = data_tensor[indices]

        # Forward
        recon, sparse_acts = model(batch)

        # Loss
        recon_loss = ((recon - batch) ** 2).mean()

        # L0 penalty
        l0_penalty = (sparse_acts > 0).float().mean()

        loss = recon_loss + 0.01 * l0_penalty

        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"    Step {step+1}/{n_steps}, loss={loss.item():.4f}")

    return model


def create_random_baseline(d_model, d_sae, l0_target, device, seed=42):
    """Create random decoder baseline (Xavier init, no training)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SimpleSAE(d_model=d_model, d_sae=d_sae, l0_target=l0_target).to(device)
    return model


def multichild_proportional_ablation(model, parent_dir, child_dirs, device, n_samples=100, k=5):
    """
    Multi-child proportional absorption measurement.
    """
    results = {
        "absorption_k": [],
        "overlap_parent_child1": [],
        "overlap_parent_child2": []
    }

    with torch.no_grad():
        for i in range(n_samples):
            strength = np.random.uniform(1.0, 3.0)

            child_dir_1 = child_dirs[0]
            child_dir_2 = child_dirs[1]

            parent_input = (parent_dir / parent_dir.norm()) * strength
            child1_input = (child_dir_1 / child_dir_1.norm()) * strength
            child2_input = (child_dir_2 / child_dir_2.norm()) * strength

            parent_acts = model.get_encoder_activations(parent_input.unsqueeze(0).to(device))[0]
            child1_acts = model.get_encoder_activations(child1_input.unsqueeze(0).to(device))[0]
            child2_acts = model.get_encoder_activations(child2_input.unsqueeze(0).to(device))[0]

            k_effective = min(k, parent_acts.shape[0])
            _, parent_topk = torch.topk(parent_acts, k=k_effective)
            _, child1_topk = torch.topk(child1_acts, k=k_effective)
            _, child2_topk = torch.topk(child2_acts, k=k_effective)

            parent_set = set(parent_topk.cpu().tolist())
            child1_set = set(child1_topk.cpu().tolist())
            child2_set = set(child2_topk.cpu().tolist())

            overlap1 = len(parent_set & child1_set) / k_effective
            overlap2 = len(parent_set & child2_set) / k_effective

            results["overlap_parent_child1"].append(float(overlap1))
            results["overlap_parent_child2"].append(float(overlap2))

            absorption = (overlap1 + overlap2) / 2
            results["absorption_k"].append(float(absorption))

    absorption_arr = np.array(results["absorption_k"])

    return {
        "absorption_mean": float(np.mean(absorption_arr)),
        "absorption_std": float(np.std(absorption_arr)),
        "absorption_min": float(np.min(absorption_arr)),
        "absorption_max": float(np.max(absorption_arr)),
        "n_samples": n_samples,
        "k": k,
        "raw": results["absorption_k"]
    }


def main():
    print("=" * 70)
    print("Multi-seed Validation: H1 Absorption Stability")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Samples per seed: {PILOT_SAMPLES}")
    print(f"K children: {K_CHILDREN}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Load hierarchy
    print("\n[1/4] Loading hierarchy...")
    hierarchy_path = WORKSPACE / "data" / "synthetic_hierarchy.json"
    with open(hierarchy_path) as f:
        hierarchy = json.load(f)

    parent_dir = torch.FloatTensor(hierarchy["levels"]["parent"]["direction"]).to(DEVICE)
    dogs_dir = torch.FloatTensor(hierarchy["levels"]["children"][0]["direction"]).to(DEVICE)
    cats_dir = torch.FloatTensor(hierarchy["levels"]["children"][1]["direction"]).to(DEVICE)
    child_dirs = [dogs_dir, cats_dir]
    print(f"  Parent direction dim: {parent_dir.shape[0]}")

    # Generate training data
    print("\n[2/4] Generating training data...")
    d_model = parent_dir.shape[0]
    d_sae = 4096
    l0_target = 32

    n_train = 5000
    train_data = []
    for _ in range(n_train):
        # Mix of parent and child directions
        mix = np.random.rand()
        if mix < 0.33:
            direction = parent_dir.cpu().numpy()
        elif mix < 0.67:
            direction = dogs_dir.cpu().numpy()
        else:
            direction = cats_dir.cpu().numpy()

        strength = np.random.uniform(0.5, 2.0)
        sample = direction / (np.linalg.norm(direction) + 1e-8) * strength
        sample += np.random.randn(d_model) * 0.1  # Add noise
        train_data.append(sample)

    train_data = np.array(train_data)
    print(f"  Generated {n_train} training samples, dim={d_model}")

    # Train SAE with each seed
    print("\n[3/4] Training SAEs with different seeds...")
    trained_saes = {}
    trained_absorptions = []
    random_absorptions = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}:")

        # Train SAE
        sae = train_sae(
            d_model, d_sae, l0_target, DEVICE,
            train_data, n_steps=500, lr=1e-3, seed=seed
        )
        trained_saes[seed] = sae

        # Measure absorption
        result = multichild_proportional_ablation(
            model=sae,
            parent_dir=parent_dir,
            child_dirs=child_dirs,
            device=DEVICE,
            n_samples=PILOT_SAMPLES,
            k=K_CHILDREN
        )
        trained_absorptions.append(result['absorption_mean'])
        print(f"    Trained SAE absorption: {result['absorption_mean']:.4f} +/- {result['absorption_std']:.4f}")

        # Random baseline for this seed
        random_sae = create_random_baseline(d_model, d_sae, l0_target, DEVICE, seed=seed + 100)
        random_result = multichild_proportional_ablation(
            model=random_sae,
            parent_dir=parent_dir,
            child_dirs=child_dirs,
            device=DEVICE,
            n_samples=PILOT_SAMPLES,
            k=K_CHILDREN
        )
        random_absorptions.append(random_result['absorption_mean'])
        print(f"    Random baseline absorption: {random_result['absorption_mean']:.4f} +/- {random_result['absorption_std']:.4f}")

    # Statistical analysis
    print("\n[4/4] Statistical analysis...")

    trained_arr = np.array(trained_absorptions)
    random_arr = np.array(random_absorptions)

    # Variance across seeds
    trained_variance = np.var(trained_arr)
    random_variance = np.var(random_arr)

    print(f"\nMulti-seed absorption summary:")
    print(f"  {'Seed':<8} {'Trained SAE':<20} {'Random Baseline':<20}")
    print(f"  {'-'*50}")
    for i, seed in enumerate(SEEDS):
        print(f"  {seed:<8} {trained_arr[i]:.4f}{'':>15} {random_arr[i]:.4f}{'':>15}")
    print(f"  {'-'*50}")
    print(f"  {'Mean':<8} {np.mean(trained_arr):.4f}{'':>15} {np.mean(random_arr):.4f}{'':>15}")
    print(f"  {'Std':<8} {np.std(trained_arr):.4f}{'':>15} {np.std(random_arr):.4f}{'':>15}")
    print(f"  {'Var':<8} {trained_variance:.6f}{'':>12} {random_variance:.6f}{'':>12}")

    # Check pass criteria
    print(f"\nPass Criteria:")
    non_zero_variance = trained_variance > 1e-6
    print(f"  Non-zero variance (var > 1e-6): {trained_variance:.6f} - {'PASS' if non_zero_variance else 'FAIL'}")

    all_trained_above_threshold = all(trained_arr > 0.3)
    print(f"  All seeds absorption > 0.3: {all_trained_above_threshold} - {'PASS' if all_trained_above_threshold else 'FAIL'}")

    majority_random_below = sum(random_arr < 0.2) >= len(SEEDS) * 0.67
    print(f"  Majority ({sum(random_arr < 0.2)}/{len(SEEDS)}) random < 0.2: {'PASS' if majority_random_below else 'FAIL'}")

    h1_pass = non_zero_variance and all_trained_above_threshold
    print(f"\n  OVERALL H1 Multi-seed: {'PASS' if h1_pass else 'FAIL'}")

    # One-way ANOVA to check if variance is significant
    # Collect raw data for ANOVA
    all_trained_raw = []
    all_random_raw = []

    for seed_idx, seed in enumerate(SEEDS):
        result = multichild_proportional_ablation(
            model=trained_saes[seed],
            parent_dir=parent_dir,
            child_dirs=child_dirs,
            device=DEVICE,
            n_samples=PILOT_SAMPLES,
            k=K_CHILDREN
        )
        all_trained_raw.extend(result['raw'])

        random_sae = create_random_baseline(d_model, d_sae, l0_target, DEVICE, seed=seed + 100)
        random_result = multichild_proportional_ablation(
            model=random_sae,
            parent_dir=parent_dir,
            child_dirs=child_dirs,
            device=DEVICE,
            n_samples=PILOT_SAMPLES,
            k=K_CHILDREN
        )
        all_random_raw.extend(random_result['raw'])

    t_stat, p_val = ttest_ind(all_trained_raw, all_random_raw)
    print(f"\n  T-test (trained vs random): t={t_stat:.3f}, p={p_val:.2e}")

    # Prepare output
    output = {
        "task": "multichild_multiseed_pilot",
        "hypothesis": "H1: Multi-child absorption stable across seeds",
        "config": {
            "seeds": SEEDS,
            "n_samples_per_seed": PILOT_SAMPLES,
            "k_children": K_CHILDREN,
            "d_model": d_model,
            "d_sae": d_sae,
            "l0_target": l0_target,
            "training_steps": 500
        },
        "results": {
            "by_seed": [
                {
                    "seed": int(seed),
                    "trained_absorption_mean": float(trained_arr[i]),
                    "random_absorption_mean": float(random_arr[i])
                }
                for i, seed in enumerate(SEEDS)
            ],
            "summary": {
                "trained_mean": float(np.mean(trained_arr)),
                "trained_std": float(np.std(trained_arr)),
                "trained_variance": float(trained_variance),
                "random_mean": float(np.mean(random_arr)),
                "random_std": float(np.std(random_arr)),
                "random_variance": float(random_variance),
                "delta_mean": float(np.mean(trained_arr) - np.mean(random_arr))
            }
        },
        "statistics": {
            "ttest": {
                "t_statistic": float(t_stat),
                "p_value": float(p_val)
            },
            "anova": {
                "f_statistic": None,  # Not computed for 2 groups
                "comment": "Using t-test for 2-group comparison"
            }
        },
        "pass_criteria": {
            "non_zero_variance": bool(non_zero_variance),
            "variance_threshold": 1e-6,
            "all_trained_above_0.3": bool(all_trained_above_threshold),
            "majority_random_below_0.2": bool(majority_random_below),
            "overall_pass": bool(h1_pass)
        },
        "timestamp": datetime.now().isoformat()
    }

    # Save results
    output_dir = WORKSPACE / "exp" / "results" / "new_pilots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "multiseed_pilot.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    print("\n" + "=" * 70)
    return output


if __name__ == "__main__":
    output = main()
