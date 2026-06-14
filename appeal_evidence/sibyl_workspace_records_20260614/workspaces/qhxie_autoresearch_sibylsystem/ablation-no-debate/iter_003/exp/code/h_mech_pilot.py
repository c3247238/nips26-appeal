#!/usr/bin/env python3
"""
H_Mech Pilot: 2x2 factorial using real GPT-2 Small SAE.
Tests whether encoder alignment drives absorption (Condition B ≈ D) and decoder is irrelevant (Condition C ≈ A).

Uses sae-lens to load real SAE and measure multi-child proportional absorption.
"""
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from transformer_lens import HookedTransformer
from sae_lens import SAE

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
SEED = 42
N_SAMPLES = 100
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_sae():
    """Load GPT-2 Small + SAE from sae-lens."""
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    print("Loading SAE (blocks.8.hook_resid_pre)...")
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        device=device
    )
    return model, sae


def find_high_activation_features(model, sae, prompts, n_features=10):
    """Find features with highest mean activation across prompts."""
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


def measure_absorption(model, sae, parent_idx, child_indices, n_samples=50):
    """Measure multi-child proportional absorption using real SAE.

    For each sample:
    1. Get residual stream for a prompt
    2. Encode to SAE features
    3. Record parent feature activation at final token
    4. Ablate child features in SAE space
    5. Decode ablated features back to residual space
    6. Re-encode and measure parent activation
    7. Compute absorption = after / before
    """
    parent_acts_before = []
    parent_acts_after = []

    prompts = [
        "The capital of France is Paris, a beautiful city in Europe",
        "Machine learning models can learn complex patterns from data",
        "The weather today is sunny and warm with clear skies",
        "Scientists discovered new particles in recent experiments",
        "The book was written by a famous author about history",
        "Neural networks use gradients to learn representations",
        "Climate change affects ecosystems around the world",
        "The company released new products for customers",
    ]

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

            # Parent activation before ablation (at final token)
            parent_before = features[-1, parent_idx].item()

            # Ablate child features
            ablated = features.clone()
            ablated[:, child_indices] = 0

            # Decode to residual space
            ablated_resid = ablated @ sae.W_dec

            # Re-encode
            reencoded = sae.encode(ablated_resid.unsqueeze(0)).squeeze(0)
            parent_after = reencoded[-1, parent_idx].item()

        parent_acts_before.append(max(0, parent_before))
        parent_acts_after.append(max(0, parent_after))

    before_mean = np.mean(parent_acts_before)
    after_mean = np.mean(parent_acts_after)

    if before_mean > 1e-6:
        absorption = after_mean / before_mean
    else:
        absorption = 0.0

    return absorption, before_mean, after_mean


def create_variant_sae(sae, variant_type):
    """Create variant SAE with modified encoder/decoder.

    variant_type:
        - original: unchanged SAE
        - random_encoder: random encoder, original decoder
        - random_decoder: original encoder, random decoder
        - random_both: random encoder and decoder
    """
    d_sae, d_model = sae.W_enc.shape

    if variant_type == "original":
        return sae.W_enc.clone(), sae.W_dec.clone()
    elif variant_type == "random_encoder":
        W_enc = torch.randn(d_sae, d_model, device=device)
        W_enc = W_enc / W_enc.norm(dim=1, keepdim=True)
        return W_enc, sae.W_dec.clone()
    elif variant_type == "random_decoder":
        W_dec = torch.randn(d_model, d_sae, device=device)
        W_dec = W_dec / W_dec.norm(dim=0, keepdim=True)
        return sae.W_enc.clone(), W_dec
    elif variant_type == "random_both":
        W_enc = torch.randn(d_sae, d_model, device=device)
        W_enc = W_enc / W_enc.norm(dim=1, keepdim=True)
        W_dec = torch.randn(d_model, d_sae, device=device)
        W_dec = W_dec / W_dec.norm(dim=0, keepdim=True)
        return W_enc, W_dec


def run_condition(model, sae, condition, variant_type, parent_idx, child_indices, n_samples):
    """Run one condition of the 2x2 factorial."""
    W_enc, W_dec = create_variant_sae(sae, variant_type)

    # Create temporary SAE wrapper with variant weights
    class VariantSAE:
        def __init__(self, W_enc, W_dec):
            self.W_enc = W_enc
            self.W_dec = W_dec

        def encode(self, x):
            h = torch.nn.functional.relu(x @ self.W_enc)
            return h

        def decode(self, h):
            return h @ self.W_dec

    variant_sae = VariantSAE(W_enc, W_dec)

    absorption, before, after = measure_absorption(
        model, variant_sae, parent_idx, child_indices, n_samples
    )

    enc_type = "trained" if variant_type in ["original", "random_decoder"] else "random"
    dec_type = "trained" if variant_type in ["original", "random_encoder"] else "random"

    return {
        "condition": condition,
        "encoder": enc_type,
        "decoder": dec_type,
        "absorption_rate": float(absorption),
        "parent_activation_before": float(before),
        "parent_activation_after": float(after),
    }


def main():
    results = {
        "task_id": "h_mech_pilot",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "model": "gpt2-small",
            "sae": "blocks.8.hook_resid_pre",
        },
        "conditions": [],
    }

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

    # Select parent and child features
    parent_idx = high_act_features[0]
    child_indices = high_act_features[1:6]
    print(f"Parent feature: {parent_idx}")
    print(f"Child features: {child_indices}")

    print("\nRunning 2x2 factorial conditions...")

    conditions = [
        ("A", "random_both"),
        ("B", "random_decoder"),
        ("C", "random_encoder"),
        ("D", "original"),
    ]

    absorption_rates = {}
    for cond_name, variant_type in conditions:
        enc_type = "trained" if variant_type in ["original", "random_decoder"] else "random"
        dec_type = "trained" if variant_type in ["original", "random_encoder"] else "random"

        print(f"  Condition {cond_name}: Encoder={enc_type}, Decoder={dec_type}", end=" ... ")

        result = run_condition(
            model, sae, cond_name, variant_type,
            parent_idx, child_indices, N_SAMPLES
        )

        results["conditions"].append(result)
        absorption_rates[cond_name] = result["absorption_rate"]
        print(f"absorption={result['absorption_rate']:.4f}")

    # Summary
    b_vs_d = abs(absorption_rates["B"] - absorption_rates["D"])
    c_vs_a = absorption_rates["C"] - absorption_rates["A"]

    results["summary"] = {
        "condition_A_random_random": float(absorption_rates["A"]),
        "condition_B_trained_random": float(absorption_rates["B"]),
        "condition_C_random_trained": float(absorption_rates["C"]),
        "condition_D_trained_trained": float(absorption_rates["D"]),
        "encoder_driven_check": b_vs_d < 0.1,
        "decoder_irrelevant_check": c_vs_a < 0.05,
        "b_vs_d_delta": float(b_vs_d),
        "c_vs_a_delta": float(c_vs_a),
        "pilot_pass": b_vs_d < 0.1 and c_vs_a < 0.05,
    }

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj.item()
        return obj

    results = convert(results)

    # Save
    output_file = RESULTS_DIR / "h_mech_pilot_seed42.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n=== H_Mech Pilot Results ===")
    for cond, rate in absorption_rates.items():
        print(f"  {cond}: {rate:.4f}")
    print(f"\n|B-D| = {b_vs_d:.4f} (threshold 0.1): {results['summary']['encoder_driven_check']}")
    print(f"C-A = {c_vs_a:.4f} (threshold 0.05): {results['summary']['decoder_irrelevant_check']}")
    print(f"Pilot PASS: {results['summary']['pilot_pass']}")

    return results


if __name__ == "__main__":
    main()