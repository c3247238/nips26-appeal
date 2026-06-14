#!/usr/bin/env python3
"""
Pilot Experiment 3 & 4: Measure absorption on trained SAE vs random baseline
Simplified version focusing on core measurement.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
import os

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
GPU_ID = 0
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Reset CUDA to clear any corruption
if torch.cuda.is_available():
    torch.cuda.empty_cache()


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

    def forward(self, x):
        pre_acts = self.W_encoder(x) + self.b_enc
        acts = torch.relu(pre_acts)

        # TopK - safe implementation
        batch_size = acts.shape[0]
        k = min(self.l0_target, acts.shape[1])

        if k < acts.shape[1] and batch_size > 0:
            # Get top k values for each row
            topk_values = torch.zeros_like(acts)
            for i in range(batch_size):
                vals, idxs = torch.topk(acts[i], k=k)
                topk_values[i, idxs] = vals
            sparse_acts = topk_values
        else:
            sparse_acts = acts

        recon = self.W_decoder(sparse_acts) + self.b_dec
        return recon, sparse_acts


def load_trained_sae(path):
    """Load trained SAE from checkpoint."""
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]

    if "d_model" not in config:
        config["d_model"] = 128

    model = SimpleSAE(
        d_model=config["d_model"],
        d_sae=config["d_sae"],
        l0_target=config["l0_target"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def compute_asymmetry_index(model):
    """Compute AI = ||W_encoder[i]|| / ||W_decoder[:, i]||."""
    W_enc = model.W_encoder.weight.detach().cpu().numpy()
    W_dec = model.W_decoder.weight.detach().cpu().numpy()

    enc_norms = np.linalg.norm(W_enc, axis=0)
    dec_norms = np.linalg.norm(W_dec, axis=1)

    ai = enc_norms / (dec_norms + 1e-8)
    return ai


def measure_absorption_overlap(model, hierarchy, device=DEVICE):
    """
    Measure absorption via feature overlap.
    If parent features overlap with child features, absorption is high.
    """
    with torch.no_grad():
        # Get hierarchy directions
        parent_dir = torch.FloatTensor(hierarchy["levels"]["parent"]["direction"]).to(device)
        dogs_dir = torch.FloatTensor(hierarchy["levels"]["children"][0]["direction"]).to(device)
        cats_dir = torch.FloatTensor(hierarchy["levels"]["children"][1]["direction"]).to(device)

        # Project directions to latent space
        parent_enc = torch.relu(torch.matmul(parent_dir.unsqueeze(0), model.W_encoder.weight.T) + model.b_enc)[0]
        dogs_enc = torch.relu(torch.matmul(dogs_dir.unsqueeze(0), model.W_encoder.weight.T) + model.b_enc)[0]
        cats_enc = torch.relu(torch.matmul(cats_dir.unsqueeze(0), model.W_encoder.weight.T) + model.b_enc)[0]

        # Find top-k features
        k = min(10, model.d_sae)
        _, parent_topk = torch.topk(parent_enc, k=k)
        _, dogs_topk = torch.topk(dogs_enc, k=k)
        _, cats_topk = torch.topk(cats_enc, k=k)

        parent_topk = set(parent_topk.cpu().tolist())
        dogs_topk = set(dogs_topk.cpu().tolist())
        cats_topk = set(cats_topk.cpu().tolist())

        # Overlap percentage
        parent_dogs_overlap = len(parent_topk & dogs_topk) / k
        parent_cats_overlap = len(parent_topk & cats_topk) / k
        absorption_rate = (parent_dogs_overlap + parent_cats_overlap) / 2

        return {
            "absorption_rate": float(absorption_rate),
            "parent_dogs_overlap": float(parent_dogs_overlap),
            "parent_cats_overlap": float(parent_cats_overlap),
            "parent_topk_features": list(parent_topk),
            "dogs_topk_features": list(dogs_topk),
            "cats_topk_features": list(cats_topk)
        }


def measure_absorption_ablation(model, hierarchy, device=DEVICE):
    """
    Measure absorption via ablation test.
    Test whether ablating parent affects child features.
    """
    with torch.no_grad():
        parent_dir = torch.FloatTensor(hierarchy["levels"]["parent"]["direction"]).to(device)
        dogs_dir = torch.FloatTensor(hierarchy["levels"]["children"][0]["direction"]).to(device)
        cats_dir = torch.FloatTensor(hierarchy["levels"]["children"][1]["direction"]).to(device)

        # Test input: children present
        children_input = 3.0 * dogs_dir + 3.0 * cats_dir
        children_input = children_input.unsqueeze(0)

        # Test input: children + parent present
        full_input = 5.0 * parent_dir + children_input[0]
        full_input = full_input.unsqueeze(0)

        # Get activations
        _, acts_children = model(children_input)
        _, acts_full = model(full_input)

        # Find top features for children
        child_enc = torch.relu(torch.matmul(children_input, model.W_encoder.weight.T) + model.b_enc)
        top_child_idx = torch.argmax(child_enc[0]).item()

        # Compare child feature activation with/without parent
        child_with_parent = acts_full[0, top_child_idx].item()
        child_without_parent = acts_children[0, top_child_idx].item()

        # If child fires regardless of parent presence, absorption is high
        if child_without_parent > 0:
            absorption_rate = min(child_with_parent / (child_without_parent + 1e-8), 1.0)
        else:
            absorption_rate = 0.0

        return {
            "absorption_rate": float(absorption_rate),
            "top_child_feature": top_child_idx,
            "child_activation_with_parent": float(child_with_parent),
            "child_activation_without_parent": float(child_without_parent)
        }


def main():
    print("=" * 60)
    print("Pilot Experiment 3 & 4: Absorption Measurement")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Load data and trained SAE
    print("\n[1/4] Loading data and hierarchy...")
    data_path = WORKSPACE / "data" / "pilot_activations.json"
    with open(data_path) as f:
        data = json.load(f)
    n_samples = data["n_samples"]
    print(f"  Loaded {n_samples} samples")

    hierarchy_path = WORKSPACE / "data" / "synthetic_hierarchy.json"
    with open(hierarchy_path) as f:
        hierarchy = json.load(f)
    print("  Loaded hierarchy")

    # Load trained SAE
    print("\n[2/4] Loading trained SAE...")
    sae_path = WORKSPACE / "exp" / "results" / "pilots" / "sae_L0_32.pt"

    if not sae_path.exists():
        print(f"  ERROR: SAE checkpoint not found at {sae_path}")
        print("  Run p1_train_sae.py first!")
        return None

    trained_sae, sae_config = load_trained_sae(sae_path)
    print(f"  Loaded SAE: d_model={sae_config['d_model']}, d_sae={sae_config['d_sae']}, L0={sae_config['l0_target']}")

    # Create random baseline
    print("\n[3/4] Creating random baseline...")
    torch.manual_seed(SEED + 1)  # Different seed for random baseline
    np.random.seed(SEED + 1)
    random_baseline = SimpleSAE(
        d_model=sae_config["d_model"],
        d_sae=sae_config["d_sae"],
        l0_target=sae_config["l0_target"]
    ).to(DEVICE)
    print("  Created random baseline (Xavier init, no training)")

    # Reset seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Measure absorption
    print("\n[4/4] Measuring absorption rates...")

    models = {
        "trained_sae": trained_sae,
        "random_baseline": random_baseline
    }

    results = {}

    for name, model in models.items():
        print(f"\n  {name}:")

        # Overlap method
        overlap_result = measure_absorption_overlap(model, hierarchy, DEVICE)
        print(f"    Overlap method: {overlap_result['absorption_rate']:.4f}")

        # Ablation method
        ablation_result = measure_absorption_ablation(model, hierarchy, DEVICE)
        print(f"    Ablation method: {ablation_result['absorption_rate']:.4f}")

        # Asymmetry index
        ai = compute_asymmetry_index(model)
        ai_mean = float(np.mean(ai))
        ai_std = float(np.std(ai))
        print(f"    Asymmetry index: {ai_mean:.4f} +/- {ai_std:.4f}")

        results[name] = {
            "overlap_method": overlap_result,
            "ablation_method": ablation_result,
            "asymmetry_index": {
                "mean": ai_mean,
                "std": ai_std,
                "min": float(np.min(ai)),
                "max": float(np.max(ai))
            }
        }

    # Compare results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    trained_ov = results["trained_sae"]["overlap_method"]["absorption_rate"]
    random_ov = results["random_baseline"]["overlap_method"]["absorption_rate"]

    trained_ab = results["trained_sae"]["ablation_method"]["absorption_rate"]
    random_ab = results["random_baseline"]["ablation_method"]["absorption_rate"]

    print(f"\nOverlap absorption rate:")
    print(f"  Trained SAE: {trained_ov:.4f}")
    print(f"  Random baseline: {random_ov:.4f}")
    print(f"  Difference: {trained_ov - random_ov:.4f}")

    print(f"\nAblation absorption rate:")
    print(f"  Trained SAE: {trained_ab:.4f}")
    print(f"  Random baseline: {random_ab:.4f}")
    print(f"  Difference: {trained_ab - random_ab:.4f}")

    # Pass criteria: trained SAE shows absorption, random doesn't
    overlap_pass = trained_ov > random_ov and trained_ov > 0.1
    ablation_pass = trained_ab > random_ab and trained_ab > 0.1

    print(f"\nPass criteria (H1 - trained SAE shows absorption, random doesn't):")
    print(f"  Overlap method: {'PASS' if overlap_pass else 'FAIL'}")
    print(f"  Ablation method: {'PASS' if ablation_pass else 'FAIL'}")

    overall_pass = overlap_pass or ablation_pass

    # Hypothesis evaluation
    hypothesis_h1 = {
        "description": "Trained SAEs show higher absorption than random baselines",
        "trained_absorption_overlap": trained_ov,
        "random_absorption_overlap": random_ov,
        "trained_absorption_ablation": trained_ab,
        "random_absorption_ablation": random_ab,
        "overlap_pass": overlap_pass,
        "ablation_pass": ablation_pass,
        "supported": overall_pass
    }

    # Save results
    os.makedirs(WORKSPACE / "exp" / "results" / "pilots", exist_ok=True)
    output = {
        "task": "p1_absorption_pilot",
        "results": results,
        "hypothesis_h1": hypothesis_h1,
        "overall_pass": overall_pass,
        "pilot_samples": n_samples,
        "sae_config": sae_config
    }

    results_path = WORKSPACE / "exp" / "results" / "pilots" / "absorption_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {results_path}")

    print("\n" + "=" * 60)
    print(f"PILOT PASS: {overall_pass}")
    print("=" * 60)

    if overall_pass:
        print("\n[SUCCESS] Absorption detected in trained SAE but not in random baseline.")
        print("This supports H1: trained SAEs show genuine absorption patterns.")
    else:
        print("\n[INFO] No clear absorption difference detected in pilot.")
        print("This may indicate the synthetic hierarchy needs refinement.")

    return output


if __name__ == "__main__":
    results = main()
