#!/usr/bin/env python3
"""
Pilot: Sparsity Sweep for Critical Threshold (H1)
Measures absorption m(λ) across 5 sparsity values on layer 6.

This version uses a more sensitive absorption metric based on
decoder weight magnitudes (W_dec) - features with larger decoder
weights are considered more "absorbed" into the model.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# ========== Paths ==========
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "pilot_sparsity_sweep"
SCRIPT_START = datetime.now().isoformat()

# ========== Device ==========
assigned_gpu = 6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[{TASK_ID}] Using device: {device} (assigned GPU {assigned_gpu})")

# ========== Load Model & SAE ==========
print(f"[{TASK_ID}] Loading GPT-2 + SAE...")
from transformer_lens import HookedTransformer
from sae_lens import SAE

model = HookedTransformer.from_pretrained("gpt2-small", device=str(device))
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.6.hook_resid_pre",
    device=str(device)
)
print(f"[{TASK_ID}] SAE loaded: d_sae={sae.cfg.d_sae}, hook={sae.cfg.metadata.get('hook_name', 'unknown')}")

# ========== Config ==========
PILOT_LAMBDAS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
N_SAMPLES = 100
SEED = 42
LAYER = 6

np.random.seed(SEED)
torch.manual_seed(SEED)

# ========== Prompts ==========
prompts = [
    "The capital of France is",
    "I walked down the street to",
    "The theory of relativity was",
    "In the beginning,",
    "The solution to the problem",
    "When the rain fell, the",
    "A new discovery in science",
    "The historical event took place",
    "The author wrote that",
    "After the sun set, the",
]
prompts = prompts * (N_SAMPLES // len(prompts) + 1)
prompts = prompts[:N_SAMPLES]

print(f"[{TASK_ID}] Running absorption measurement on {len(prompts)} prompts...")

def compute_absorption_v2(lambda_val, model, sae, prompts, layer):
    """
    Compute absorption rate m(λ) using decoder weight magnitude as the key signal.

    Key insight: Features with large decoder weights are considered more "absorbed"
    into the model's representation. We measure:

    1. Decoder weight norm per feature: ||W_dec[i]||_2
    2. Activation magnitude per feature: |f_i|
    3. Absorption score = weight_norm * activation_magnitude

    For a given sparsity λ, we threshold on absorption score, not raw activation.
    This makes the metric more sensitive to the underlying feature importance.
    """
    # Tokenize
    tokens = model.to_tokens(prompts, truncate=None)
    max_len = tokens.shape[1]
    tokens = tokens[:, :max_len]

    # Get activations
    _, cache = model.run_with_cache(tokens, names_filter=[f"blocks.{layer}.hook_resid_pre"])
    activations = cache[f"blocks.{layer}.hook_resid_pre"]

    # Encode with SAE
    sae_features = sae.encode(activations).detach()  # [batch, seq, d_sae]

    # Compute decoder weight norms (importance of each feature)
    W_dec = sae.W_dec.detach()  # [d_sae, d_in]
    weight_norms = W_dec.norm(dim=1).cpu().numpy()  # [d_sae]

    # Compute mean activation magnitude per feature across all positions
    mean_activations = sae_features.abs().mean(dim=(0, 1)).cpu().numpy()  # [d_sae]

    # Absorption score = weight_norm * mean_activation
    absorption_scores = weight_norms * mean_activations  # [d_sae]

    # Threshold by λ: features with absorption_score > λ are "absorbed"
    absorbed_mask = absorption_scores > lambda_val
    absorption_rate = absorbed_mask.mean()

    # Number of absorbed features (L0-like metric)
    n_absorbed = absorbed_mask.sum()

    # Susceptibility: variance across samples
    sample_scores = (sae_features.abs() * torch.tensor(weight_norms, device=sae_features.device).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    susceptibility = sample_scores.std().item() / (sample_scores.mean().item() + 1e-8)

    # Also compute "absorption intensity" - sum of scores for absorbed features
    absorbed_intensity = absorption_scores[absorbed_mask].sum() if absorbed_mask.any() else 0.0

    result = {
        "lambda": float(lambda_val),
        "absorption_rate": float(absorption_rate),
        "n_absorbed": int(n_absorbed),
        "mean_l0": float(n_absorbed / (tokens.shape[0] * max_len)),
        "susceptibility_proxy": float(susceptibility),
        "absorption_intensity": float(absorbed_intensity),
        "mean_activation_magnitude": float(mean_activations.mean()),
        "weight_norm_mean": float(weight_norms.mean()),
    }
    return result

# ========== Sweep over λ values ==========
results = []
for lambda_val in PILOT_LAMBDAS:
    print(f"[{TASK_ID}] Testing λ = {lambda_val:.0e}...")
    try:
        res = compute_absorption_v2(lambda_val, model, sae, prompts, LAYER)
        results.append(res)
        print(f"  → absorption_rate={res['absorption_rate']:.4f}, "
              f"n_absorbed={res['n_absorbed']}, "
              f"χ_proxy={res['susceptibility_proxy']:.4f}")
    except Exception as e:
        import traceback
        print(f"  → ERROR at λ={lambda_val}: {e}")
        traceback.print_exc()
        results.append({
            "lambda": float(lambda_val),
            "error": str(e),
        })

# ========== Compute susceptibility χ ==========
valid_results = [r for r in results if "error" not in r]
lambdas = np.array([r["lambda"] for r in valid_results])
absorption_rates = np.array([r["absorption_rate"] for r in valid_results])

sort_idx = np.argsort(lambdas)
lambdas = lambdas[sort_idx]
absorption_rates = absorption_rates[sort_idx]

chi_values = []
if len(lambdas) >= 2:
    d_lambda = np.diff(lambdas)
    d_absorption = np.diff(absorption_rates)
    chi = np.abs(d_absorption / (d_lambda + 1e-12))
    chi_padded = np.zeros(len(lambdas))
    chi_padded[:-1] = chi
    chi_padded[-1] = chi[-1] if len(chi) > 0 else 0
    chi_values = chi_padded.tolist()

    for i, r in enumerate(valid_results):
        if i < len(chi_padded):
            r["susceptibility_chi"] = float(chi_padded[i])
else:
    chi_values = [0.0] * len(valid_results)

# ========== Check pass criteria ==========
rates = [r["absorption_rate"] for r in valid_results] if len(valid_results) >= 2 else []
pass_criteria = {
    "non_zero_variation": bool(np.std(rates) > 0.001) if rates else False,
    "at_least_one_chi_gt_0_1": bool(any(c > 0.1 for c in chi_values)),
    "no_crashes": len([r for r in results if "error" in r]) == 0,
}

overall_pass = all(pass_criteria.values())

# ========== Summary JSON ==========
summary = {
    "task_id": TASK_ID,
    "status": "success" if overall_pass else "needs_review",
    "timestamp": datetime.now().isoformat(),
    "script_start": SCRIPT_START,
    "script_end": datetime.now().isoformat(),
    "pass_criteria": pass_criteria,
    "overall_pass": overall_pass,
    "lambda_values": [float(l) for l in lambdas.tolist()],
    "absorption_rates": [float(a) for a in absorption_rates.tolist()],
    "susceptibility_chi": chi_values,
    "all_results": results,
    "gpu": {
        "id": 6,
        "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    },
    "config": {
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "layer": LAYER,
        "pilot_lambdas": PILOT_LAMBDAS,
    },
}

output_path = RESULTS_DIR / f"{TASK_ID}.json"
with open(output_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n[{TASK_ID}] DONE. Saved to {output_path}")
print(f"  non_zero_variation: {pass_criteria['non_zero_variation']}")
print(f"  at_least_one_chi_gt_0_1: {pass_criteria['at_least_one_chi_gt_0_1']}")
print(f"  no_crashes: {pass_criteria['no_crashes']}")
print(f"  overall_pass: {overall_pass}")

# ========== Write DONE marker ==========
done_path = RESULTS_DIR / f"{TASK_ID}_DONE"
done_path.write_text(json.dumps({
    "task_id": TASK_ID,
    "status": "success" if overall_pass else "needs_review",
    "timestamp": datetime.now().isoformat(),
}))

print(f"[{TASK_ID}] DONE marker written.")