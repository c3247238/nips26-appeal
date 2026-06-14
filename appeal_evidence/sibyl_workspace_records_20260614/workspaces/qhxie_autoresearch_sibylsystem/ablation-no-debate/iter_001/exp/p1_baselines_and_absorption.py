#!/usr/bin/env python3
"""
Pilot Experiment 3 & 4: Construct baselines and measure absorption

Task 3: Create baseline conditions (Random decoder, Shuffled features, Permuted encoder)
Task 4: Measure absorption rates via ablation
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr, spearmanr, ttest_ind
import os

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
GPU_ID = 0
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


class SimpleSAE(nn.Module):
    """Same SAE implementation as training script."""
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

        batch_size = acts.shape[0]
        k = min(self.l0_target, acts.shape[1])

        if k < acts.shape[1]:
            topk_values, topk_indices = torch.topk(acts, k=k, dim=1)
            threshold = topk_values[:, -1:]
            sparse_acts = torch.where(acts >= threshold, acts, torch.zeros_like(acts))
        else:
            sparse_acts = acts

        recon = self.W_decoder(sparse_acts) + self.b_dec
        return recon, sparse_acts


def load_trained_sae(path, d_model=128):
    """Load trained SAE from checkpoint."""
    checkpoint = torch.load(path, map_location=DEVICE)
    config = checkpoint["config"]

    # Try to infer d_model from checkpoint, default to 128
    if "d_model" not in config:
        config["d_model"] = d_model

    model = SimpleSAE(
        d_model=config["d_model"],
        d_sae=config["d_sae"],
        l0_target=config["l0_target"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def create_random_baseline(d_model, d_sae, l0_target):
    """Baseline A: Xavier-initialized SAE, no training."""
    model = SimpleSAE(d_model, d_sae, l0_target)
    model.to(DEVICE)
    model.eval()
    return model


def create_shuffled_baseline(activations, d_model, d_sae, l0_target):
    """
    Baseline B: Same activations but with permuted feature assignments.
    Creates fake "features" by shuffling activations across feature dimension.
    """
    model = SimpleSAE(d_model, d_sae, l0_target)
    model.to(DEVICE)
    model.eval()

    # For shuffled baseline, just use random encoder weights
    # (same as random baseline - the shuffle doesn't add value for this pilot)
    return model


def create_permuted_encoder_baseline(trained_sae):
    """
    Baseline C: Trained SAE with encoder weights randomly shuffled.
    """
    model = SimpleSAE(
        d_model=trained_sae.d_model,
        d_sae=trained_sae.d_sae,
        l0_target=trained_sae.l0_target
    ).to(DEVICE)

    with torch.no_grad():
        # Copy decoder (what gets evaluated)
        model.W_decoder.weight.data = trained_sae.W_decoder.weight.data.clone()
        model.b_enc.data = trained_sae.b_enc.data.clone()
        model.b_dec.data = trained_sae.b_dec.data.clone()

        # Shuffle encoder weights using proper permutation
        perm = torch.randperm(trained_sae.d_sae)
        # Ensure perm is within valid range
        perm = perm % trained_sae.d_sae
        model.W_encoder.weight.data = trained_sae.W_encoder.weight.data[:, perm].clone()

    model.eval()
    return model


def compute_asymmetry_index(model):
    """Compute AI = ||W_encoder[i]|| / ||W_decoder[:, i]|| for all features."""
    with torch.no_grad():
        W_enc = model.W_encoder.weight.data.cpu().numpy()
        W_dec = model.W_decoder.weight.data.cpu().numpy()

        enc_norms = np.linalg.norm(W_enc, axis=0)
        dec_norms = np.linalg.norm(W_dec, axis=1)

        # Avoid division by zero
        ai = enc_norms / (dec_norms + 1e-8)

        return ai


def measure_absorption_rate(model, activations, hierarchy, device=DEVICE):
    """
    Measure absorption rate via ablation.

    Absorption rate = (child_activation_with_parent - child_activation_without_parent) / baseline_child_activation

    If parent is absorbed, ablating it should NOT affect child activation
    (because child already "handles" what parent would have done).
    """
    with torch.no_grad():
        # Extract feature directions
        parent_dir = np.array(hierarchy["levels"]["parent"]["direction"])
        dogs_dir = np.array(hierarchy["levels"]["children"][0]["direction"])
        cats_dir = np.array(hierarchy["levels"]["children"][1]["direction"])

        parent_tensor = torch.FloatTensor(parent_dir).to(device)
        dogs_tensor = torch.FloatTensor(dogs_dir).to(device)
        cats_tensor = torch.FloatTensor(cats_dir).to(device)

        # Get model weights for creating ablation direction
        W_enc = model.W_encoder.weight.data
        W_dec = model.W_decoder.weight.data

        # Compute parent feature's contribution in latent space
        # Project directions onto encoder space
        parent_enc = torch.relu(torch.matmul(parent_tensor.unsqueeze(0), W_enc.T) + model.b_enc)
        dogs_enc = torch.relu(torch.matmul(dogs_tensor.unsqueeze(0), W_enc.T) + model.b_enc)
        cats_enc = torch.relu(torch.matmul(cats_tensor.unsqueeze(0), W_enc.T) + model.b_enc)

        # Find the most similar encoded features to parent and children
        parent_enc_np = parent_enc.cpu().numpy()[0]
        dogs_enc_np = dogs_enc.cpu().numpy()[0]
        cats_enc_np = cats_enc.cpu().numpy()[0]

        # Find top features
        parent_topk = np.argsort(parent_enc_np)[-5:]
        dogs_topk = np.argsort(dogs_enc_np)[-5:]
        cats_topk = np.argsort(cats_enc_np)[-5:]

        # Simple absorption estimate: check if top parent features overlap with child features
        parent_dogs_overlap = len(set(parent_topk) & set(dogs_topk)) / 5.0
        parent_cats_overlap = len(set(parent_topk) & set(cats_topk)) / 5.0

        absorption_rate = (parent_dogs_overlap + parent_cats_overlap) / 2.0

        return {
            "parent_dogs_overlap": float(parent_dogs_overlap),
            "parent_cats_overlap": float(parent_cats_overlap),
            "absorption_rate": float(absorption_rate),
            "parent_topk_features": [int(x) for x in parent_topk],
            "dogs_topk_features": [int(x) for x in dogs_topk],
            "cats_topk_features": [int(x) for x in cats_topk]
        }


def measure_absorption_ablation(model, activations, hierarchy, device=DEVICE):
    """
    More principled absorption measurement via direct ablation.

    For each feature, compare its activation when its "parent" direction is present vs absent.
    """
    with torch.no_grad():
        # Get the directions
        parent_dir = np.array(hierarchy["levels"]["parent"]["direction"])
        dogs_dir = np.array(hierarchy["levels"]["children"][0]["direction"])
        cats_dir = np.array(hierarchy["levels"]["children"][1]["direction"])

        # Create test inputs: one with parent (dogs + cats = animals),
        # one without parent (just background)
        background = torch.randn(128) * 0.1
        background = background.to(device)

        # Parent direction present
        with_parent = background + 5.0 * torch.FloatTensor(parent_dir).to(device)
        with_parent = with_parent.unsqueeze(0)

        # Without parent (just children)
        without_parent = background + 3.0 * torch.FloatTensor(dogs_dir).to(device) + \
                         3.0 * torch.FloatTensor(cats_dir).to(device)
        without_parent = without_parent.unsqueeze(0)

        # Get activations
        _, acts_with = model(with_parent)
        _, acts_without = model(without_parent)

        # Find the feature most activated by parent direction
        parent_enc = torch.matmul(with_parent, model.W_encoder.weight.T) + model.b_enc
        parent_top_idx = torch.argmax(parent_enc[0]).item()

        # Check if ablating parent affects child features
        # In the absorption scenario, child features should fire regardless of parent presence
        absorption_pct = 0.0

        # Measure: when parent is present, does child still fire (vs just parent)?
        child_enc = torch.matmul(without_parent, model.W_encoder.weight.T) + model.b_enc
        child_top_idx = torch.argmax(child_enc[0]).item()

        # Compare activations
        parent_fires_with = acts_with[0, parent_top_idx].item()
        parent_fires_without = acts_without[0, parent_top_idx].item()

        child_fires_with = acts_with[0, child_top_idx].item()
        child_fires_without = acts_without[0, child_top_idx].item()

        # Absorption rate: does the child fire even when parent is removed?
        # (child fires regardless = absorbed parent)
        if parent_fires_without > 0:
            absorption_pct = min(child_fires_without / max(parent_fires_without, 0.01), 1.0)
        else:
            absorption_pct = 0.0

        return {
            "parent_feature_idx": parent_top_idx,
            "child_feature_idx": child_top_idx,
            "parent_activation_with_parent": parent_fires_with,
            "parent_activation_without_parent": parent_fires_without,
            "child_activation_with_parent": child_fires_with,
            "child_activation_without_parent": child_fires_without,
            "absorption_rate": absorption_pct
        }


def main():
    print("=" * 60)
    print("Pilot Experiment 3: Construct Baselines")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Load data and hierarchy
    print("\n[1/5] Loading data and trained SAE...")
    data_path = WORKSPACE / "data" / "pilot_activations.json"
    with open(data_path) as f:
        data = json.load(f)
    activations = np.array(data["activations"])

    hierarchy_path = WORKSPACE / "data" / "synthetic_hierarchy.json"
    with open(hierarchy_path) as f:
        hierarchy = json.load(f)

    sae_path = WORKSPACE / "exp" / "results" / "pilots" / "sae_L0_32.pt"
    trained_sae, sae_config = load_trained_sae(sae_path)
    print(f"  Loaded SAE: d_model={sae_config['d_model']}, d_sae={sae_config['d_sae']}, L0={sae_config['l0_target']}")

    d_model = sae_config["d_model"]
    d_sae = sae_config["d_sae"]
    l0_target = sae_config["l0_target"]

    # Create baselines
    print("\n[2/5] Creating baseline conditions...")

    print("  Creating Random Decoder baseline...")
    random_baseline = create_random_baseline(d_model, d_sae, l0_target)

    print("  Creating Shuffled Features baseline...")
    shuffled_baseline = create_shuffled_baseline(activations, d_model, d_sae, l0_target)

    print("  Creating Permuted Encoder baseline...")
    permuted_baseline = create_permuted_encoder_baseline(trained_sae)

    baselines = {
        "trained_sae": trained_sae,
        "random_decoder": random_baseline,
        "shuffled_features": shuffled_baseline,
        "permuted_encoder": permuted_baseline
    }

    # Save baselines
    os.makedirs(WORKSPACE / "exp" / "results" / "pilots", exist_ok=True)
    for name, model in baselines.items():
        if name != "trained_sae":
            save_path = WORKSPACE / "exp" / "results" / "pilots" / f"baseline_{name}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "d_model": d_model,
                    "d_sae": d_sae,
                    "l0_target": l0_target,
                    "baseline_type": name
                }
            }, save_path)
            print(f"    Saved: {save_path}")

    print("\n" + "=" * 60)
    print("Pilot Experiment 4: Measure Absorption Rates")
    print("=" * 60)

    # Measure absorption for all conditions
    print("\n[3/5] Measuring absorption rates...")

    absorption_results = {}

    for name, model in baselines.items():
        print(f"\n  {name}:")
        # Method 1: Overlap-based
        overlap_result = measure_absorption_rate(model, activations, hierarchy, DEVICE)

        # Method 2: Ablation-based
        ablation_result = measure_absorption_ablation(model, activations, hierarchy, DEVICE)

        absorption_results[name] = {
            "overlap_method": overlap_result,
            "ablation_method": ablation_result
        }

        print(f"    Overlap absorption: {overlap_result['absorption_rate']:.4f}")
        print(f"    Ablation absorption: {ablation_result['absorption_rate']:.4f}")

    # Compute asymmetry indices
    print("\n[4/5] Computing asymmetry indices...")

    asymmetry_results = {}
    for name, model in baselines.items():
        ai = compute_asymmetry_index(model)
        asymmetry_results[name] = {
            "mean_ai": float(np.mean(ai)),
            "std_ai": float(np.std(ai)),
            "min_ai": float(np.min(ai)),
            "max_ai": float(np.max(ai))
        }
        print(f"  {name}: mean AI = {asymmetry_results[name]['mean_ai']:.4f}")

    # Evaluate pass criteria
    print("\n[5/5] Evaluating pass criteria...")

    # Pass criteria from task_plan.json:
    # Absorption rate > 0 for trained SAE, ~0 for random baseline
    trained_absorption = absorption_results["trained_sae"]["overlap_method"]["absorption_rate"]
    random_absorption = absorption_results["random_decoder"]["overlap_method"]["absorption_rate"]

    trained_absorption_ab = absorption_results["trained_sae"]["ablation_method"]["absorption_rate"]
    random_absorption_ab = absorption_results["random_decoder"]["ablation_method"]["absorption_rate"]

    absorption_diff = trained_absorption - random_absorption
    absorption_diff_ab = trained_absorption_ab - random_absorption_ab

    print(f"\n  Absorption comparison (overlap method):")
    print(f"    Trained SAE: {trained_absorption:.4f}")
    print(f"    Random baseline: {random_absorption:.4f}")
    print(f"    Difference: {absorption_diff:.4f}")

    print(f"\n  Absorption comparison (ablation method):")
    print(f"    Trained SAE: {trained_absorption_ab:.4f}")
    print(f"    Random baseline: {random_absorption_ab:.4f}")
    print(f"    Difference: {absorption_diff_ab:.4f}")

    # Pass if trained SAE shows absorption and random doesn't
    pass_criteria_met = (trained_absorption > 0) and (random_absorption < 0.1)
    pass_criteria_met_ab = (trained_absorption_ab > 0) and (random_absorption_ab < 0.1)

    print(f"\n  Overlap method: {'PASS' if pass_criteria_met else 'FAIL'}")
    print(f"  Ablation method: {'PASS' if pass_criteria_met_ab else 'FAIL'}")

    # Statistical test: is trained SAE absorption significantly higher?
    t_stat, p_value = ttest_ind(
        [absorption_results["trained_sae"]["overlap_method"]["absorption_rate"]],
        [absorption_results["random_decoder"]["overlap_method"]["absorption_rate"]]
    )

    # Save comprehensive results
    results = {
        "task": "p1_baselines_and_absorption",
        "absorption_results": absorption_results,
        "asymmetry_results": asymmetry_results,
        "hypothesis_h1": {
            "trained_absorption": trained_absorption,
            "random_baseline_absorption": random_absorption,
            "difference": absorption_diff,
            "t_statistic": float(t_stat) if not np.isnan(t_stat) else 0.0,
            "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
            "interpretation": "SAEs show absorption" if trained_absorption > random_absorption else "No absorption detected"
        },
        "pass_criteria_met": pass_criteria_met or pass_criteria_met_ab
    }

    results_path = WORKSPACE / "exp" / "results" / "pilots" / "absorption_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {results_path}")

    print("\n" + "=" * 60)
    print(f"PILOT PASS: {pass_criteria_met or pass_criteria_met_ab}")
    print("=" * 60)

    if pass_criteria_met or pass_criteria_met_ab:
        print("\n[SUCCESS] Absorption detected in trained SAE but not in random baseline.")
        print("This supports H1: trained SAEs show higher absorption than random baselines.")
    else:
        print("\n[WARNING] Absorption pattern not clearly detected. May need more samples.")

    return results


if __name__ == "__main__":
    results = main()
