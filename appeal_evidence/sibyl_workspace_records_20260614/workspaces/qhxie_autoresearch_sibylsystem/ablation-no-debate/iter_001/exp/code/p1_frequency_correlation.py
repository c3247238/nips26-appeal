#!/usr/bin/env python3
"""
H2 Analysis: Absorption vs Feature Activation Frequency

Hypothesis H2: Absorption rate inversely correlates with feature activation frequency.
Lower-frequency features should be more absorbed due to competitive exclusion.

Spearman correlation: rho < -0.3, p < 0.01
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
import os

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
DEVICE = "cpu"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


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


def load_trained_sae(path, device):
    """Load trained SAE from checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    config["d_model"] = config.get("d_model", 128)

    model = SimpleSAE(
        d_model=config["d_model"],
        d_sae=config["d_sae"],
        l0_target=config["l0_target"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def compute_feature_frequency(model, activations, device):
    """Compute activation frequency for each feature across all samples."""
    with torch.no_grad():
        features = torch.FloatTensor(activations["activations"]).to(device)
        acts = model.get_encoder_activations(features)
        active_counts = (acts > 0).sum(dim=0).cpu().numpy()
        frequency = active_counts / len(features)
        mean_acts = acts.mean(dim=0).cpu().numpy()
    return frequency, mean_acts


def compute_absorption_per_feature(model, parent_dir, child_dirs, device, n_samples=1000, k=10):
    """Compute absorption rate per feature, averaged across many samples."""
    with torch.no_grad():
        n_features = model.d_sae
        feature_absorption = np.zeros(n_features)

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

            # For each feature, compute overlap with parent
            k_eff = min(k, parent_acts.shape[0])
            _, parent_topk = torch.topk(parent_acts, k=k_eff)
            parent_set = set(parent_topk.cpu().tolist())

            # Accumulate absorption for parent features
            for idx in parent_topk.cpu().tolist():
                child1_overlap = 1.0 if idx in set(torch.topk(child1_acts, k=k_eff)[1].cpu().tolist()) else 0.0
                child2_overlap = 1.0 if idx in set(torch.topk(child2_acts, k=k_eff)[1].cpu().tolist()) else 0.0
                feature_absorption[idx] += (child1_overlap + child2_overlap) / 2

        # Average across samples
        feature_absorption /= n_samples

    return feature_absorption


def main():
    print("=" * 70)
    print("H2 Analysis: Absorption vs Feature Activation Frequency")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Load data
    print("\n[1/4] Loading data...")
    data_path = WORKSPACE / "data" / "pilot_activations.json"
    with open(data_path) as f:
        activations = json.load(f)
    print(f"  Loaded {activations['n_samples']} samples")

    hierarchy_path = WORKSPACE / "data" / "synthetic_hierarchy.json"
    with open(hierarchy_path) as f:
        hierarchy = json.load(f)

    parent_dir = torch.FloatTensor(hierarchy["levels"]["parent"]["direction"]).to(DEVICE)
    dogs_dir = torch.FloatTensor(hierarchy["levels"]["children"][0]["direction"]).to(DEVICE)
    cats_dir = torch.FloatTensor(hierarchy["levels"]["children"][1]["direction"]).to(DEVICE)
    child_dirs = [dogs_dir, cats_dir]

    # Load trained SAE
    print("\n[2/4] Loading trained SAE...")
    sae_path = WORKSPACE / "exp" / "results" / "pilots" / "sae_L0_32.pt"
    trained_sae, sae_config = load_trained_sae(sae_path, DEVICE)
    print(f"  SAE: d_model={sae_config['d_model']}, d_sae={sae_config['d_sae']}, L0={sae_config['l0_target']}")

    # Compute feature frequencies
    print("\n[3/4] Computing feature frequencies...")
    frequency, mean_acts = compute_feature_frequency(trained_sae, activations, DEVICE)

    # Compute absorption per feature
    print("\n[4/4] Computing absorption per feature...")
    absorption_per_feature = compute_absorption_per_feature(trained_sae, parent_dir, child_dirs, DEVICE, n_samples=1000, k=10)

    # Compute correlations
    print("\n[5/5] Computing correlations...")

    # Only consider features that activate at least once
    active_mask = frequency > 0
    freq_active = frequency[active_mask]
    absorp_active = absorption_per_feature[active_mask]

    # Spearman correlation (rank-based, robust to outliers)
    rho, p_spearman = spearmanr(freq_active, absorp_active)

    # Pearson correlation (linear)
    r, p_pearson = pearsonr(freq_active, absorp_active)

    # H2 pass criteria: rho < -0.3, p < 0.01
    h2_pass = rho < -0.3 and p_spearman < 0.01

    print(f"\n  Feature Statistics:")
    print(f"    Total features: {len(frequency)}")
    print(f"    Active features: {active_mask.sum()}")
    print(f"    Frequency range: [{freq_active.min():.4f}, {freq_active.max():.4f}]")
    print(f"    Absorption range: [{absorp_active.min():.4f}, {absorp_active.max():.4f}]")

    print(f"\n  Correlations:")
    print(f"    Spearman rho: {rho:.4f} (p={p_spearman:.4e})")
    print(f"    Pearson r: {r:.4f} (p={p_pearson:.4e})")

    print(f"\n  H2 Pass Criteria:")
    print(f"    rho < -0.3: {rho:.4f} {'PASS' if rho < -0.3 else 'FAIL'}")
    print(f"    p < 0.01: {p_spearman:.4e} {'PASS' if p_spearman < 0.01 else 'FAIL'}")
    print(f"\n  OVERALL H2: {'PASS' if h2_pass else 'FAIL'}")

    # Prepare output
    output = {
        "task": "p1_frequency_correlation",
        "h2_pass": bool(h2_pass),
        "correlations": {
            "spearman_rho": float(rho),
            "spearman_p": float(p_spearman),
            "pearson_r": float(r),
            "pearson_p": float(p_pearson)
        },
        "feature_statistics": {
            "total_features": int(len(frequency)),
            "active_features": int(active_mask.sum()),
            "frequency_mean": float(freq_active.mean()),
            "frequency_std": float(freq_active.std()),
            "absorption_mean": float(absorp_active.mean()),
            "absorption_std": float(absorp_active.std())
        },
        "data": {
            "frequency": frequency.tolist(),
            "absorption_per_feature": absorption_per_feature.tolist()
        },
        "timestamp": datetime.now().isoformat()
    }

    # Save results
    output_dir = WORKSPACE / "exp" / "results" / "full"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "h2_frequency_correlation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {output_path}")

    print("\n" + "=" * 70)
    return output


if __name__ == "__main__":
    output = main()
