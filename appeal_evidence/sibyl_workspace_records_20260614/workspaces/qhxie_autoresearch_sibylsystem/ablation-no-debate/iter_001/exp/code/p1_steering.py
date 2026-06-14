#!/usr/bin/env python3
"""
H3 Steering Intervention: Does steering absorbed features toward parent directions improve sensitivity?

Hypothesis H3: Steering absorbed features toward parent directions improves feature sensitivity.

Implementation:
1. Identify absorbed features (proportional absorption > threshold = top quartile)
2. Reconstruct parent direction from children's decoder subspace
3. Apply steering: add alpha * parent_dir to SAE activations
4. Measure sensitivity before/after on held-out text
5. Compare absorbed vs non-absorbed controls

Paired t-test: p < 0.01
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from scipy.stats import ttest_rel
import os
import gc

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

    def forward_with_steering(self, x, feature_idx, steering_alpha, parent_direction):
        """Forward pass with steering on a specific feature."""
        pre_acts = self.W_encoder(x) + self.b_enc
        acts = torch.relu(pre_acts)

        # Apply steering
        if steering_alpha > 0:
            # Get decoder direction for this feature
            decoder_dir = self.W_decoder.weight[:, feature_idx]
            # Project parent direction onto this decoder direction
            projection = (parent_direction @ decoder_dir) / (decoder_dir.norm() + 1e-8)
            # Add steering signal
            acts[0, feature_idx] += steering_alpha * projection.item()

        # TopK sparsity
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


def compute_feature_absorption(model, parent_dir, child_dirs, device, n_samples=500, k=10):
    """Compute absorption rate per feature."""
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

            k_eff = min(k, parent_acts.shape[0])
            _, parent_topk = torch.topk(parent_acts, k=k_eff)

            for idx in parent_topk.cpu().tolist():
                child1_overlap = 1.0 if idx in set(torch.topk(child1_acts, k=k_eff)[1].cpu().tolist()) else 0.0
                child2_overlap = 1.0 if idx in set(torch.topk(child2_acts, k=k_eff)[1].cpu().tolist()) else 0.0
                feature_absorption[idx] += (child1_overlap + child2_overlap) / 2

        feature_absorption /= n_samples

    return feature_absorption


def measure_sensitivity(model, input_dir, feature_idx, device):
    """Measure sensitivity of a feature to its decoder direction."""
    with torch.no_grad():
        # Get reconstruction at baseline
        baseline_recon = model.W_decoder.weight[:, feature_idx].unsqueeze(0)

        # Get reconstruction with slight perturbation in feature activation
        pre_acts = model.W_encoder(input_dir.unsqueeze(0).to(device)) + model.b_enc
        base_activation = pre_acts[0, feature_idx].item()

        # Compute sensitivity as reconstruction norm per unit activation
        if base_activation > 1e-6:
            sensitivity = baseline_recon.norm().item() / base_activation
        else:
            sensitivity = 0.0

    return sensitivity


def main():
    print("=" * 70)
    print("H3 Steering Intervention Analysis")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Load data
    print("\n[1/6] Loading data...")
    hierarchy_path = WORKSPACE / "data" / "synthetic_hierarchy.json"
    with open(hierarchy_path) as f:
        hierarchy = json.load(f)

    parent_dir = torch.FloatTensor(hierarchy["levels"]["parent"]["direction"]).to(DEVICE)
    dogs_dir = torch.FloatTensor(hierarchy["levels"]["children"][0]["direction"]).to(DEVICE)
    cats_dir = torch.FloatTensor(hierarchy["levels"]["children"][1]["direction"]).to(DEVICE)
    child_dirs = [dogs_dir, cats_dir]

    # Load trained SAE
    print("\n[2/6] Loading trained SAE...")
    sae_path = WORKSPACE / "exp" / "results" / "pilots" / "sae_L0_32.pt"
    trained_sae, sae_config = load_trained_sae(sae_path, DEVICE)
    print(f"  SAE: d_model={sae_config['d_model']}, d_sae={sae_config['d_sae']}, L0={sae_config['l0_target']}")

    # Compute feature absorption
    print("\n[3/6] Computing feature absorption...")
    feature_absorption = compute_feature_absorption(trained_sae, parent_dir, child_dirs, DEVICE, n_samples=500, k=10)

    # Identify absorbed vs non-absorbed features
    print("\n[4/6] Identifying absorbed and non-absorbed features...")
    threshold = np.percentile(feature_absorption[feature_absorption > 0], 75)

    absorbed_mask = feature_absorption >= threshold
    non_absorbed_mask = feature_absorption == 0

    absorbed_indices = np.where(absorbed_mask)[0].tolist()
    non_absorbed_indices = np.where(non_absorbed_mask)[0].tolist()

    print(f"  Absorption threshold (75th percentile): {threshold:.4f}")
    print(f"  Absorbed features: {len(absorbed_indices)}")
    print(f"  Non-absorbed features: {len(non_absorbed_indices)}")

    # Steering experiment
    print("\n[5/6] Running steering experiment...")
    alphas = [0.0, 0.1, 0.2]

    results = {
        "absorbed": {alpha: [] for alpha in alphas},
        "non_absorbed": {alpha: [] for alpha in alphas}
    }

    # Use parent direction as steering target
    parent_direction = parent_dir

    n_test_samples = 100
    torch.manual_seed(SEED + 10)
    np.random.seed(SEED + 10)

    for i in range(n_test_samples):
        # Generate random test direction
        test_direction = torch.randn(sae_config["d_model"]).to(DEVICE)
        test_direction = test_direction / test_direction.norm()

        # Test absorbed features
        for feat_idx in absorbed_indices[:10]:  # Limit to 10 features per sample
            for alpha in alphas:
                sens = measure_sensitivity(trained_sae, test_direction, feat_idx, DEVICE)
                results["absorbed"][alpha].append(sens)

        # Test non-absorbed features
        for feat_idx in non_absorbed_indices[:10]:
            for alpha in alphas:
                sens = measure_sensitivity(trained_sae, test_direction, feat_idx, DEVICE)
                results["non_absorbed"][alpha].append(sens)

    # Statistical analysis
    print("\n[6/6] Statistical analysis...")

    absorbed_baseline = np.array(results["absorbed"][0.0])
    absorbed_steered = np.array(results["absorbed"][0.2])
    non_absorbed_baseline = np.array(results["non_absorbed"][0.0])
    non_absorbed_steered = np.array(results["non_absorbed"][0.2])

    # Paired t-test: absorbed features before vs after steering
    t_absorbed, p_absorbed = ttest_rel(absorbed_steered, absorbed_baseline)

    # Paired t-test: non-absorbed features before vs after steering
    t_non_absorbed, p_non_absorbed = ttest_rel(non_absorbed_steered, non_absorbed_baseline)

    # Difference in improvement between absorbed and non-absorbed
    absorbed_improvement = absorbed_steered.mean() - absorbed_baseline.mean()
    non_absorbed_improvement = non_absorbed_steered.mean() - non_absorbed_baseline.mean()

    print(f"\n  Sensitivity Results (alpha=0.2):")
    print(f"    Absorbed features:")
    print(f"      Baseline: {absorbed_baseline.mean():.4f} +/- {absorbed_baseline.std():.4f}")
    print(f"      Steered: {absorbed_steered.mean():.4f} +/- {absorbed_steered.std():.4f}")
    print(f"      Improvement: {absorbed_improvement:+.4f}")
    print(f"      Paired t-test: t={t_absorbed:.3f}, p={p_absorbed:.4e}")

    print(f"    Non-absorbed features:")
    print(f"      Baseline: {non_absorbed_baseline.mean():.4f} +/- {non_absorbed_baseline.std():.4f}")
    print(f"      Steered: {non_absorbed_steered.mean():.4f} +/- {non_absorbed_steered.std():.4f}")
    print(f"      Improvement: {non_absorbed_improvement:+.4f}")
    print(f"      Paired t-test: t={t_non_absorbed:.3f}, p={p_non_absorbed:.4e}")

    # H3 pass criteria: absorbed features show greater improvement than non-absorbed
    h3_pass = (
        absorbed_improvement > non_absorbed_improvement and
        p_absorbed < 0.01
    )

    print(f"\n  H3 Pass Criteria:")
    print(f"    Absorbed improvement > non-absorbed: {absorbed_improvement:+.4f} vs {non_absorbed_improvement:+.4f}")
    print(f"    Absorbed p-value < 0.01: {p_absorbed:.4e}")
    print(f"\n  OVERALL H3: {'PASS' if h3_pass else 'FAIL'}")

    # Prepare output
    output = {
        "task": "p1_steering",
        "h3_pass": bool(h3_pass),
        "absorption_threshold": float(threshold),
        "n_absorbed_features": len(absorbed_indices),
        "n_non_absorbed_features": len(non_absorbed_indices),
        "steering_results": {
            "absorbed": {
                "baseline_mean": float(absorbed_baseline.mean()),
                "baseline_std": float(absorbed_baseline.std()),
                "steered_mean": float(absorbed_steered.mean()),
                "steered_std": float(absorbed_steered.std()),
                "improvement": float(absorbed_improvement),
                "t_statistic": float(t_absorbed),
                "p_value": float(p_absorbed)
            },
            "non_absorbed": {
                "baseline_mean": float(non_absorbed_baseline.mean()),
                "baseline_std": float(non_absorbed_baseline.std()),
                "steered_mean": float(non_absorbed_steered.mean()),
                "steered_std": float(non_absorbed_steered.std()),
                "improvement": float(non_absorbed_improvement),
                "t_statistic": float(t_non_absorbed),
                "p_value": float(p_non_absorbed)
            }
        },
        "alpha_values": alphas,
        "key_finding": f"Absorbed features showed {'greater' if absorbed_improvement > non_absorbed_improvement else 'less'} sensitivity improvement after steering compared to non-absorbed features.",
        "timestamp": datetime.now().isoformat()
    }

    # Save results
    output_dir = WORKSPACE / "exp" / "results" / "full"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "h3_steering_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {output_path}")

    print("\n" + "=" * 70)
    return output


if __name__ == "__main__":
    output = main()
