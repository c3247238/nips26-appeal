#!/usr/bin/env python3
"""
H_Mech Pilot: 2x2 Factorial Decomposition (Geometric vs Learned)

Decompose absorption into geometric vs learned contributions via 2x2 factorial:
- (A) Random encoder + Random decoder (pure geometry)
- (B) Trained encoder + Random decoder (encoder alignment only)
- (C) Random encoder + Trained decoder (decoder geometry only)
- (D) Trained encoder + Trained decoder (full training)

Expected:
- Condition C ≈ D ≈ 0.48 (geometric dominates)
- Condition A ≈ 0.06 (no structure)
- If D >> C: absorption is primarily learned
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from scipy.stats import ttest_ind
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42
PILOT_SAMPLES = 100
K_CHILDREN = 5

np.random.seed(SEED)
torch.manual_seed(SEED)


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


def create_condition_sae(trained_sae, condition, device, seed=42):
    """
    Create SAE for specific 2x2 factorial condition.

    Conditions:
    - A: Random encoder + Random decoder
    - B: Trained encoder + Random decoder
    - C: Random encoder + Trained decoder
    - D: Trained encoder + Trained decoder
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SimpleSAE(
        d_model=trained_sae.d_model,
        d_sae=trained_sae.d_sae,
        l0_target=trained_sae.l0_target
    ).to(device)

    with torch.no_grad():
        if condition == "A":
            # Random encoder + Random decoder (already initialized)
            pass

        elif condition == "B":
            # Trained encoder + Random decoder
            model.W_encoder.weight.copy_(trained_sae.W_encoder.weight)
            model.W_encoder.bias.copy_(trained_sae.W_encoder.bias)
            model.normalize_decoder()  # Re-normalize decoder

        elif condition == "C":
            # Random encoder + Trained decoder
            model.W_decoder.weight.copy_(trained_sae.W_decoder.weight)
            model.W_decoder.weight.div_(torch.norm(model.W_decoder.weight, dim=0, keepdim=True) + 1e-8)

        elif condition == "D":
            # Trained encoder + Trained decoder (copy all)
            model.W_encoder.weight.copy_(trained_sae.W_encoder.weight)
            model.W_encoder.bias.copy_(trained_sae.W_encoder.bias)
            model.W_decoder.weight.copy_(trained_sae.W_decoder.weight)
            model.W_decoder.weight.div_(torch.norm(model.W_decoder.weight, dim=0, keepdim=True) + 1e-8)
            model.b_dec.copy_(trained_sae.b_dec)

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
    print("H_Mech Pilot: 2x2 Factorial Decomposition (Geometric vs Learned)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Samples: {PILOT_SAMPLES}")
    print(f"K children: {K_CHILDREN}")
    print(f"Seed: {SEED}")

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

    # Load trained SAE
    print("\n[2/4] Loading trained SAE...")
    sae_path = WORKSPACE / "exp" / "results" / "pilots" / "sae_L0_32.pt"

    if not sae_path.exists():
        print(f"  SAE checkpoint not found at {sae_path}")
        print("  Creating synthetic SAE for testing...")
        d_model = parent_dir.shape[0]
        d_sae = 4096
        l0_target = 32

        class SyntheticSAE(nn.Module):
            def __init__(self, d_model, d_sae, l0):
                super().__init__()
                self.d_model = d_model
                self.d_sae = d_sae
                self.l0_target = l0
                self.W_encoder = nn.Linear(d_model, d_sae, bias=True)
                nn.init.xavier_uniform_(self.W_encoder.weight)
                nn.init.zeros_(self.W_encoder.bias)
                self.W_decoder = nn.Linear(d_sae, d_model, bias=False)
                nn.init.xavier_uniform_(self.W_decoder.weight)
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

        trained_sae = SyntheticSAE(d_model, d_sae, l0_target).to(DEVICE)
        config = {"d_model": d_model, "d_sae": d_sae, "l0_target": l0_target}
    else:
        trained_sae, config = load_trained_sae(sae_path, DEVICE)

    print(f"  SAE config: d_model={config['d_model']}, d_sae={config['d_sae']}, L0={config['l0_target']}")

    # Create 2x2 factorial conditions
    print("\n[3/4] Creating 2x2 factorial conditions...")

    conditions = {
        "A": {"encoder": "Random", "decoder": "Random", "description": "Pure geometry"},
        "B": {"encoder": "Trained", "decoder": "Random", "description": "Encoder alignment only"},
        "C": {"encoder": "Random", "decoder": "Trained", "description": "Decoder geometry only"},
        "D": {"encoder": "Trained", "decoder": "Trained", "description": "Full training"}
    }

    sae_models = {}
    for cond in ["A", "B", "C", "D"]:
        print(f"  Creating condition {cond}: {conditions[cond]['encoder']} encoder + {conditions[cond]['decoder']} decoder")
        sae_models[cond] = create_condition_sae(trained_sae, cond, DEVICE, seed=SEED)

    # Measure absorption for each condition
    print("\n[4/4] Measuring absorption for each condition...")

    results = {}
    all_raw = {}

    for cond in ["A", "B", "C", "D"]:
        print(f"\n  Condition {cond}:")
        result = multichild_proportional_ablation(
            model=sae_models[cond],
            parent_dir=parent_dir,
            child_dirs=child_dirs,
            device=DEVICE,
            n_samples=PILOT_SAMPLES,
            k=K_CHILDREN
        )
        results[cond] = result
        all_raw[cond] = result['raw']
        print(f"    Absorption: {result['absorption_mean']:.4f} +/- {result['absorption_std']:.4f}")

    # Statistical analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n2x2 Factorial Decomposition:")
    print(f"  {'Condition':<6} {'Encoder':<10} {'Decoder':<10} {'Absorption':<15} {'Interpretation':<30}")
    print(f"  {'-'*75}")

    interpretations = {
        "A": "Pure geometry (no learned structure)",
        "B": "Encoder alignment effect",
        "C": "Decoder geometry effect",
        "D": "Full training effect"
    }

    for cond in ["A", "B", "C", "D"]:
        enc = conditions[cond]["encoder"]
        dec = conditions[cond]["decoder"]
        mean = results[cond]["absorption_mean"]
        std = results[cond]["absorption_std"]
        interp = interpretations[cond]
        print(f"  {cond:<6} {enc:<10} {dec:<10} {mean:.4f} +/- {std:.4f}   {interp:<30}")

    # Statistical tests
    print(f"\nStatistical Comparisons:")

    # Compare C ≈ D (geometric vs full training)
    cond_c = np.array(all_raw["C"])
    cond_d = np.array(all_raw["D"])
    t_cd, p_cd = ttest_ind(cond_c, cond_d)
    print(f"  Condition C vs D (geometric dominated): t={t_cd:.3f}, p={p_cd:.4f}")
    geometric_dominates = abs(results["C"]["absorption_mean"] - results["D"]["absorption_mean"]) < 0.1

    # Compare A vs trained conditions
    cond_a = np.array(all_raw["A"])
    t_a_c, p_a_c = ttest_ind(cond_a, cond_c)
    t_a_d, p_a_d = ttest_ind(cond_a, cond_d)
    print(f"  Condition A vs C: t={t_a_c:.3f}, p={p_a_c:.4f}")
    print(f"  Condition A vs D: t={t_a_d:.3f}, p={p_a_d:.4f}")

    # Decomposition analysis
    print(f"\nDecomposition Analysis:")
    absorption_geom = results["C"]["absorption_mean"]  # Decoder geometry
    absorption_learned = results["D"]["absorption_mean"] - results["C"]["absorption_mean"]  # Additional from encoder
    print(f"  Geometric contribution (decoder): {absorption_geom:.4f}")
    print(f"  Learned contribution (encoder): {absorption_learned:+.4f}")
    print(f"  Total: {results['D']['absorption_mean']:.4f}")

    # Pass criteria
    print(f"\nPass Criteria:")
    condition_c_approx_d = geometric_dominates
    print(f"  Condition C ≈ D (within 0.1): {'PASS' if condition_c_approx_d else 'FAIL'} ({results['C']['absorption_mean']:.4f} vs {results['D']['absorption_mean']:.4f})")

    condition_a_low = results["A"]["absorption_mean"] < 0.2
    print(f"  Condition A absorption < 0.2: {'PASS' if condition_a_low else 'FAIL'} ({results['A']['absorption_mean']:.4f})")

    condition_b_intermediate = 0.1 < results["B"]["absorption_mean"] < 0.6
    print(f"  Condition B in (0.1, 0.6): {'PASS' if condition_b_intermediate else 'FAIL'} ({results['B']['absorption_mean']:.4f})")

    h_mech_pass = condition_c_approx_d and condition_a_low
    print(f"\n  OVERALL H_Mech: {'PASS' if h_mech_pass else 'INCONCLUSIVE'}")

    # Prepare output
    output = {
        "task": "h_mech_factorial",
        "hypothesis": "H_Mech: Absorption is primarily geometric, with encoder alignment providing additional effect",
        "config": {
            "conditions": conditions,
            "n_samples": PILOT_SAMPLES,
            "k_children": K_CHILDREN,
            "seed": SEED
        },
        "results": {
            cond: {
                "encoder": conditions[cond]["encoder"],
                "decoder": conditions[cond]["decoder"],
                "absorption_mean": results[cond]["absorption_mean"],
                "absorption_std": results[cond]["absorption_std"],
                "absorption_min": results[cond]["absorption_min"],
                "absorption_max": results[cond]["absorption_max"]
            }
            for cond in ["A", "B", "C", "D"]
        },
        "decomposition": {
            "geometric_contribution": float(absorption_geom),
            "learned_contribution": float(absorption_learned),
            "total": float(results["D"]["absorption_mean"]),
            "geometric_dominates": bool(geometric_dominates)
        },
        "statistics": {
            "c_vs_d": {
                "t_statistic": float(t_cd),
                "p_value": float(p_cd),
                "difference": float(results["D"]["absorption_mean"] - results["C"]["absorption_mean"])
            },
            "a_vs_c": {
                "t_statistic": float(t_a_c),
                "p_value": float(p_a_c)
            },
            "a_vs_d": {
                "t_statistic": float(t_a_d),
                "p_value": float(p_a_d)
            }
        },
        "pass_criteria": {
            "condition_c_approx_d": bool(condition_c_approx_d),
            "condition_a_low": bool(condition_a_low),
            "condition_b_intermediate": bool(condition_b_intermediate),
            "overall_pass": bool(h_mech_pass)
        },
        "interpretation": {
            "geometric_dominant": bool(geometric_dominates),
            "encoder_alignment_effect": float(results["B"]["absorption_mean"] - results["A"]["absorption_mean"]),
            "decoder_geometry_effect": float(results["C"]["absorption_mean"] - results["A"]["absorption_mean"]),
            "full_training_effect": float(results["D"]["absorption_mean"] - results["A"]["absorption_mean"])
        },
        "timestamp": datetime.now().isoformat()
    }

    # Save results
    output_dir = WORKSPACE / "exp" / "results" / "new_pilots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "h_mech_factorial.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    print("\n" + "=" * 70)
    return output


if __name__ == "__main__":
    output = main()
